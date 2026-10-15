#!/usr/bin/env python3
"""Dense, resumable killfeed OCR using the existing recognition implementation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2

from parse_killfeed import DEFAULT_ROW_BANDS_JSON, parse_killfeed_image

MANIFEST_SCHEMA = "valorant-killfeed-dense-run.v1"
OBSERVATION_SCHEMA = "valorant-killfeed-observation.v1"
TIMELINE_SCHEMA = "valorant-killfeed-event-timeline.v1"
RECEIPT_SCHEMA = "valorant-killfeed-validation-receipt.v1"
DEFAULT_RECT = (80, 1300, 600, 180)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def timestamp_grid(start: float, end: float, interval: float, origin: float = 0.0) -> list[float]:
    if interval <= 0 or end < start:
        raise ValueError("interval must be positive and end must not precede start")
    first_index = max(0, int((start - origin) // interval))
    while origin + first_index * interval < start - 1e-9:
        first_index += 1
    out: list[float] = []
    index = first_index
    while True:
        value = origin + index * interval
        if value > end + 1e-9:
            break
        out.append(round(value, 6))
        index += 1
    return out


def frame_at(video: Path, timestamp: float) -> Any:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot decode frame at {timestamp:.3f}s")
    return frame


def row_payload(event: Any, observation_id: str, order: int) -> dict[str, Any]:
    data = asdict(event) if hasattr(event, "__dataclass_fields__") else dict(event)
    active = bool(data.get("active", True))
    observed_order = data.get("row_band_index")
    if observed_order is None:
        observed_order = order
    return {
        "row_ref": f"{observation_id}:row:{observed_order}",
        "order": int(observed_order),
        "row_band_index": int(observed_order),
        "active": active,
        "killer": data.get("killer", ""),
        "victim": data.get("victim", ""),
        "raw_left": data.get("raw_left", ""),
        "raw_right": data.get("raw_right", ""),
        "row_color": data.get("row_color", "inactive" if not active else "unknown"),
        "orientation": data.get("orientation"),
        "probable_enemy_kill": bool(data.get("probable_enemy_kill", False)),
        "weapon": data.get("weapon"),
        "recognition_status": "inactive" if not active else (
            "ambiguous" if data.get("killer") in (None, "", "?") or data.get("victim") in (None, "", "?") else "recognized"
        ),
    }


def observation_status(rows: list[dict[str, Any]]) -> str:
    active = [row for row in rows if row["active"]]
    if not active:
        return "not_detected"
    if any(row["recognition_status"] == "ambiguous" for row in active):
        return "ambiguous"
    return "detected"


def build_observation(
    timestamp: float,
    frame: Any,
    events: Iterable[Any],
    *,
    source_sha256: str,
    config_sha256: str,
    elapsed: float,
) -> dict[str, Any]:
    observation_id = f"kf:{timestamp:.6f}"
    frame_bytes = frame.tobytes()
    rows = [row_payload(event, observation_id, index) for index, event in enumerate(events)]
    return {
        "schema": OBSERVATION_SCHEMA,
        "observation_id": observation_id,
        "source_timestamp": timestamp,
        "source_video_sha256": source_sha256,
        "configuration_sha256": config_sha256,
        "frame_sha256": digest_bytes(frame_bytes),
        "frame_shape": list(frame.shape),
        "status": observation_status(rows),
        "rows": rows,
        "processing_seconds": round(elapsed, 6),
        "created_at": utc_now(),
    }


def error_observation(timestamp: float, source_sha256: str, config_sha256: str, error: Exception) -> dict[str, Any]:
    return {
        "schema": OBSERVATION_SCHEMA,
        "observation_id": f"kf:{timestamp:.6f}",
        "source_timestamp": timestamp,
        "source_video_sha256": source_sha256,
        "configuration_sha256": config_sha256,
        "frame_sha256": None,
        "status": "error",
        "rows": [],
        "error": f"{type(error).__name__}: {error}",
        "created_at": utc_now(),
    }


def load_checkpoint(path: Path, source_sha256: str, config_sha256: str) -> dict[float, dict[str, Any]]:
    observations: dict[float, dict[str, Any]] = {}
    if not path.exists():
        return observations
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_video_sha256") != source_sha256 or row.get("configuration_sha256") != config_sha256:
                raise ValueError(f"incompatible checkpoint record at line {number}")
            observations[float(row["source_timestamp"])] = row
    return observations


def append_checkpoint(path: Path, observation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(observation, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("killer"), row.get("victim"), row.get("raw_left"), row.get("raw_right"),
        row.get("row_color"), row.get("recognition_status"), row.get("row_band_index"),
        row.get("orientation"),
    )


def compact_observations(observations: list[dict[str, Any]], interval: float) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    open_runs: dict[int, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    def close(order: int) -> None:
        run = open_runs.pop(order, None)
        if run:
            runs.append(run)

    for observation in observations:
        timestamp = float(observation["source_timestamp"])
        active_rows = {int(row["order"]): row for row in observation.get("rows", []) if row.get("active")}
        samples.append({
            "observation_ref": observation["observation_id"],
            "source_timestamp": timestamp,
            "status": observation.get("status"),
            "row_refs": [row["row_ref"] for row in observation.get("rows", []) if row.get("active")],
        })
        for order in list(open_runs):
            if order not in active_rows:
                close(order)
        for order, row in active_rows.items():
            current = open_runs.get(order)
            continuous = current and abs(timestamp - float(current["last_observed"]) - interval) < 1e-6
            if not current or not continuous or tuple(current["signature"]) != row_signature(row):
                close(order)
                current = {
                    "run_id": f"kfr:{len(runs) + len(open_runs):06d}",
                    "order": order,
                    "row_band_index": row.get("row_band_index"),
                    "killer": row.get("killer"),
                    "victim": row.get("victim"),
                    "raw_left": row.get("raw_left"),
                    "raw_right": row.get("raw_right"),
                    "row_color": row.get("row_color"),
                    "orientation": row.get("orientation"),
                    "recognition_status": row.get("recognition_status"),
                    "first_observed": timestamp,
                    "last_observed": timestamp,
                    "observation_count": 0,
                    "row_refs": [],
                    "observation_refs": [],
                    "signature": list(row_signature(row)),
                }
                open_runs[order] = current
            current["last_observed"] = timestamp
            current["observation_count"] += 1
            current["row_refs"].append(row["row_ref"])
            current["observation_refs"].append(observation["observation_id"])
    for order in list(open_runs):
        close(order)
    runs.sort(key=lambda run: (run["first_observed"], run["order"], run["run_id"]))
    for index, run in enumerate(runs):
        run["run_id"] = f"kfr:{index:06d}"
        run.pop("signature", None)
    return {"schema": TIMELINE_SCHEMA, "sampling_interval_seconds": interval, "samples": samples, "runs": runs}


def finalize(out_dir: Path, manifest: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(observations, key=lambda row: float(row["source_timestamp"]))
    requested = manifest["sampling"]["timestamps"]
    found = [float(row["source_timestamp"]) for row in ordered]
    if found != requested:
        raise ValueError("checkpoint does not exactly cover requested grid")
    raw_lines = b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode() for row in ordered)
    raw_path = out_dir / "observations.jsonl.gz"
    fd, temp_name = tempfile.mkstemp(prefix=".observations.", dir=out_dir)
    os.close(fd)
    try:
        with open(temp_name, "wb") as raw_handle:
            # Do not let the random atomic-temp basename leak into the gzip
            # header.  A fixed mtime alone is insufficient for byte-stable
            # regeneration when GzipFile derives ``filename`` from fileobj.
            with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as handle:
                handle.write(raw_lines)
        os.replace(temp_name, raw_path)
    finally:
        if os.path.exists(temp_name): os.unlink(temp_name)
    timeline = compact_observations(ordered, float(manifest["sampling"]["interval_seconds"]))
    timeline.update({
        "source_video_sha256": manifest["source"]["sha256"],
        "configuration_sha256": manifest["configuration_sha256"],
        "coverage": {
            "requested_samples": len(requested),
            "processed_samples": len(ordered),
            "successful_ocr_samples": sum(row.get("status") != "error" for row in ordered),
            "grid_complete": True,
            "elimination_recall": None,
            "elimination_recall_scope": "requires independent ground truth",
        },
    })
    timeline_path = out_dir / "timeline.json"
    atomic_write(timeline_path, (json.dumps(timeline, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "valid": True,
        "source_video_sha256": manifest["source"]["sha256"],
        "configuration_sha256": manifest["configuration_sha256"],
        "raw_sha256": digest_file(raw_path),
        "timeline_sha256": digest_file(timeline_path),
        "requested_samples": len(requested),
        "processed_samples": len(ordered),
        "successful_ocr_samples": sum(row.get("status") != "error" for row in ordered),
        "error_samples": sum(row.get("status") == "error" for row in ordered),
        "run_count": len(timeline["runs"]),
        "validated_at": utc_now(),
    }
    receipt_path = out_dir / "validation.json"
    atomic_write(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def process(
    video: Path,
    out_dir: Path,
    start: float,
    end: float,
    *,
    interval: float = 1.0,
    origin: float = 0.0,
    rect: tuple[int, int, int, int] = DEFAULT_RECT,
    row_bands: Path = DEFAULT_ROW_BANDS_JSON,
    ocr_engine: str = "easyocr",
    frame_reader: Callable[[Path, float], Any] = frame_at,
    parser: Callable[..., Any] = parse_killfeed_image,
) -> dict[str, Any]:
    video = video.resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    source_hash = digest_file(video)
    timestamps = timestamp_grid(start, end, interval, origin)
    config = {
        "schema": MANIFEST_SCHEMA,
        "source": {"path": str(video), "sha256": source_hash},
        "sampling": {"start": start, "end": end, "interval_seconds": interval, "origin_seconds": origin, "timestamps": timestamps},
        "recognition": {"owner": "valorant-ocr-benchmark", "ocr_engine": ocr_engine, "killfeed_rect": list(rect), "row_bands": str(row_bands.resolve()), "row_bands_sha256": digest_file(row_bands.resolve()), "weapons_enabled": False},
    }
    config_hash = digest_json(config)
    manifest = {**config, "configuration_sha256": config_hash, "created_at": utc_now()}
    atomic_write(out_dir / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    checkpoint = out_dir / "checkpoint.jsonl"
    existing = load_checkpoint(checkpoint, source_hash, config_hash)
    for timestamp in timestamps:
        if timestamp in existing:
            continue
        started = time.perf_counter()
        try:
            frame = frame_reader(video, timestamp)
            events = parser(frame, row_bands_json=row_bands, weapons_enabled=False, ocr_engine=ocr_engine,
                            now=timestamp, killfeed_rect=rect, omit_inactive_if_count_ge=None, warm_ocr=True)
            observation = build_observation(timestamp, frame, events, source_sha256=source_hash,
                                            config_sha256=config_hash, elapsed=time.perf_counter() - started)
        except Exception as error:
            observation = error_observation(timestamp, source_hash, config_hash, error)
        append_checkpoint(checkpoint, observation)
        existing[timestamp] = observation
    return finalize(out_dir, manifest, list(existing.values()))


def validate(out_dir: Path) -> dict[str, Any]:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    receipt = json.loads((out_dir / "validation.json").read_text())
    errors = []
    for name, expected in (("observations.jsonl.gz", receipt["raw_sha256"]), ("timeline.json", receipt["timeline_sha256"])):
        path = out_dir / name
        if not path.exists() or digest_file(path) != expected: errors.append(f"hash mismatch: {name}")
    if manifest.get("configuration_sha256") != receipt.get("configuration_sha256"): errors.append("configuration mismatch")
    return {"ok": not errors, "errors": errors, "receipt": receipt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("process")
    run.add_argument("--video", type=Path, required=True); run.add_argument("--output", type=Path, required=True)
    run.add_argument("--start", type=float, default=0.0); run.add_argument("--end", type=float, required=True)
    run.add_argument("--interval", type=float, default=1.0); run.add_argument("--origin", type=float, default=0.0)
    run.add_argument("--killfeed-rect", default=",".join(map(str, DEFAULT_RECT))); run.add_argument("--row-bands", type=Path, default=DEFAULT_ROW_BANDS_JSON)
    check = sub.add_parser("validate"); check.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "process":
        rect = tuple(int(value) for value in args.killfeed_rect.split(","))
        print(json.dumps(process(args.video, args.output, args.start, args.end, interval=args.interval, origin=args.origin, rect=rect, row_bands=args.row_bands), indent=2))
    else:
        result = validate(args.output); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
