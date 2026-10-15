#!/usr/bin/env python3
"""
Unified killfeed parser: OCR (killer / victim) + optional weapon template matching.

Uses the same ``row_bands_frac`` JSON as ``scripts/match_killfeed_weapon.py`` so each row
line aligns with one weapon pass.

  python parse_killfeed.py --image assets/screenshots/foo.png \\
      --row-bands-json config/killfeed_row_bands.json \\
      --weapon-templates-json config/weapon_templates.json

Per-row OCR does not use EasyOCR's stacked multi-row fast path (that lives in ``process_frame``).
Default engine is EasyOCR; use ``--ocr-engine tesseract`` for a lighter dependency path on static images.

Empty killfeed slots (band over map only): green+red HSV highlight below a threshold → ``active=false``,
no OCR / weapon work. Tune ``--min-band-highlight-px`` / ``--min-band-highlight-frac`` if needed.
Use ``--omit-inactive-bands`` to drop inactive rows from JSON.

Design notes and ``parse_killfeed_image`` integration: ``docs/UNIFIED_PARSE.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import valorant_killfeed_tracker as vkt
from scripts.match_killfeed_weapon import (
    WeaponMatchParams,
    match_weapons_in_roi_bands,
)

# Defaults aligned with ``parse_killfeed.py`` CLI / tuned row-band + HSV gates.
DEFAULT_ROW_BANDS_JSON = ROOT / "config" / "killfeed_row_bands.json"
DEFAULT_WEAPON_TEMPLATES_JSON = ROOT / "config" / "weapon_templates.json"
DEFAULT_HIGHLIGHT_MIN_ABS = 280
DEFAULT_HIGHLIGHT_MIN_FRAC = 0.001


def _bgr_from_image_input(image: Path | str | bytes | np.ndarray) -> np.ndarray:
    """Load BGR from a file path, encoded image bytes, or an HxWx3 BGR ndarray."""
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("BGR image ndarray must have shape HxWx3")
        return image
    if isinstance(image, bytes):
        buf = np.frombuffer(image, dtype=np.uint8)
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if dec is None:
            raise ValueError("image bytes are not a decodable image (PNG/JPEG/WebP/BMP, etc.)")
        return dec
    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")
    return vkt.load_bgr(path)


@contextmanager
def _temporary_killfeed_region(rect: tuple[int, int, int, int] | None):
    """Temporarily set ``valorant_killfeed_tracker.REGION_KILLFEED`` (top, left, width, height)."""
    if rect is None:
        yield
        return
    top, left, width, height = rect
    if min(top, left, width, height) < 0:
        raise ValueError("killfeed_rect values must be non-negative")
    old = dict(vkt.REGION_KILLFEED)
    vkt.REGION_KILLFEED = {"top": top, "left": left, "width": width, "height": height}
    try:
        yield
    finally:
        vkt.REGION_KILLFEED = old


def _maybe_drop_inactive_events(
    events: list[vkt.KillfeedEvent],
    *,
    omit_inactive_if_count_ge: int | None,
) -> list[vkt.KillfeedEvent]:
    """
    When there are several empty row bands (inactive), drop them so callers get a compact list.

    If ``omit_inactive_if_count_ge`` is ``None``, return all events. Otherwise, when the number
    of inactive rows is >= that threshold, return only ``active`` events.
    """
    if omit_inactive_if_count_ge is None:
        return list(events)
    n_inactive = sum(1 for e in events if not e.active)
    if n_inactive >= omit_inactive_if_count_ge:
        return [e for e in events if e.active]
    return list(events)


def parse_killfeed_image(
    image: Path | str | bytes | np.ndarray,
    *,
    row_bands_json: Path | None = None,
    weapon_templates_json: Path | None = None,
    weapons_enabled: bool = True,
    ocr_engine: str | None = None,
    now: float | None = None,
    highlight_min_abs: int = DEFAULT_HIGHLIGHT_MIN_ABS,
    highlight_min_frac: float = DEFAULT_HIGHLIGHT_MIN_FRAC,
    weapon_params: WeaponMatchParams | None = None,
    killfeed_rect: tuple[int, int, int, int] | None = None,
    omit_inactive_if_count_ge: int | None = 2,
    warm_ocr: bool = True,
    timings_out: dict[str, float] | None = None,
) -> list[vkt.KillfeedEvent]:
    """
    High-level API: crop killfeed (if full-frame), run unified OCR + optional weapon templates.

    **Input**

    - ``Path`` / ``str``: path to PNG/JPEG/WebP/BMP, etc.
    - ``bytes``: encoded image (same formats as ``cv2.imdecode``).
    - ``numpy.ndarray``: BGR ``HxWx3`` (full screenshot or killfeed crop).

    **Defaults**

    Row bands and weapon JSON under ``config/``, ``ocr_engine`` from
    ``valorant_killfeed_tracker.DEFAULT_OCR_ENGINE`` (EasyOCR unless overridden), HSV highlight gates
    ``280`` px and ``0.001`` fraction — same as the CLI.

    **Inactive rows**

    If at least ``omit_inactive_if_count_ge`` rows are inactive (empty slots), inactive
    :class:`~valorant_killfeed_tracker.KillfeedEvent` entries are omitted from the returned list.
    Set ``omit_inactive_if_count_ge=None`` to always return one event per band.

    **I/O**

    Does not write parse results, JSON, or debug screenshots. It may **read** config paths
    (row bands / weapon templates) and image paths you pass in. Third-party OCR stacks may
    still use their own caches on disk.

    Returns a list of :class:`~valorant_killfeed_tracker.KillfeedEvent`.
    """
    eng = ocr_engine if ocr_engine is not None else vkt.DEFAULT_OCR_ENGINE
    bands_path = (row_bands_json or DEFAULT_ROW_BANDS_JSON).resolve()
    if not bands_path.is_file():
        raise FileNotFoundError(f"Row bands JSON not found: {bands_path}")

    wjson: Path | None
    if not weapons_enabled:
        wjson = None
    else:
        wpath = (weapon_templates_json or DEFAULT_WEAPON_TEMPLATES_JSON).resolve()
        if not wpath.is_file():
            raise FileNotFoundError(
                f"Weapon templates not found: {wpath} (set weapons_enabled=False to skip)"
            )
        wjson = wpath

    bands = vkt.load_row_bands_json(bands_path)
    wp = weapon_params or WeaponMatchParams()
    t_now = time.time() if now is None else float(now)

    with _temporary_killfeed_region(killfeed_rect):
        bgr = _bgr_from_image_input(image)
        if warm_ocr:
            vkt.warm_easyocr_for_session(eng)
        roi = vkt.crop_killfeed_region_if_possible(bgr)
        events, _boxes, _masks, _band_active, _wh = parse_killfeed_roi_unified(
            roi,
            bands,
            templates_json=wjson,
            weapon_params=wp,
            ocr_engine=eng,
            now=t_now,
            highlight_min_abs=highlight_min_abs,
            highlight_min_frac=highlight_min_frac,
            timings_out=timings_out,
        )

    return _maybe_drop_inactive_events(events, omit_inactive_if_count_ge=omit_inactive_if_count_ge)


def parse_killfeed_roi_unified(
    roi_bgr: object,
    bands: list[tuple[float, float]],
    *,
    templates_json: Path | None,
    weapon_params: WeaponMatchParams,
    ocr_engine: str,
    now: float,
    highlight_min_abs: int = 280,
    highlight_min_frac: float = 0.001,
    timings_out: dict[str, float] | None = None,
) -> tuple[
    list[vkt.KillfeedEvent],
    list[tuple[int, int, int, int, str]],
    dict[str, object],
    list[bool],
    list | None,
]:
    """
    One :class:`KillfeedEvent` per ``row_bands_frac`` slot.

    Slots with almost no green/red killfeed HSV highlight are **inactive** (empty row over map):
    no OCR, no weapon match, ``killer``/``victim`` empty, ``active=False``.

    If ``timings_out`` is a dict, it is cleared and filled with ``*_ms`` keys (setup, weapons, ocr, total).
    """
    if timings_out is not None:
        timings_out.clear()

    t_setup0 = time.perf_counter()
    boxes, masks = vkt.fixed_row_boxes_from_bands(roi_bgr, bands)
    gm, rm = masks["green"], masks["red"]
    frame_w = roi_bgr.shape[1]
    band_active = [
        vkt.killfeed_row_band_is_active(
            gm, rm, y, y + h, frame_w, min_abs=highlight_min_abs, min_frac=highlight_min_frac
        )
        for (_x, y, _w, h, _c) in boxes
    ]
    setup_ms = (time.perf_counter() - t_setup0) * 1000.0

    t_w0 = time.perf_counter()
    weapon_hits = (
        match_weapons_in_roi_bands(
            roi_bgr, bands, templates_json, weapon_params, band_active=band_active
        )
        if templates_json is not None
        else None
    )
    weapons_ms = (time.perf_counter() - t_w0) * 1000.0
    weapons = weapon_hits if weapon_hits is not None else [None] * len(boxes)

    frame_h, frame_w = roi_bgr.shape[:2]
    events: list[vkt.KillfeedEvent] = []
    t_ocr0 = time.perf_counter()
    for i, (x, y, w, h, row_color) in enumerate(boxes):
        on = band_active[i] if i < len(band_active) else True
        if not on:
            events.append(
                vkt.KillfeedEvent(
                    killer="",
                    victim="",
                    row_color="inactive",
                    probable_enemy_kill=False,
                    raw_left="",
                    raw_right="",
                    t=now,
                    weapon=None,
                    weapon_score=None,
                    weapon_margin=None,
                    weapon_vs=None,
                    row_band_index=i,
                    active=False,
                )
            )
            continue

        ex, ey, ew, eh = vkt.expand_row_box(x, y, w, h, frame_w, frame_h)
        crop = roi_bgr[ey : ey + eh, ex : ex + ew]
        row_key = vkt.frame_safe_row_cache_key(
            crop, ex >> 2, ey >> 2, ew >> 2, eh >> 2, row_color, (ocr_engine or vkt.DEFAULT_OCR_ENGINE).lower()
        )
        killer, victim, raw_l, raw_r = vkt.parse_killer_victim_from_row_crop(
            crop, ocr_engine=ocr_engine, row_cache_key=row_key
        )
        wh = weapons[i] if i < len(weapons) else None
        events.append(
            vkt.KillfeedEvent(
                killer=killer or "?",
                victim=victim or "?",
                row_color=row_color,
                probable_enemy_kill=row_color == "red",
                raw_left=raw_l,
                raw_right=raw_r,
                t=now,
                weapon=wh.weapon if wh else None,
                weapon_score=wh.score if wh else None,
                weapon_margin=wh.margin if wh else None,
                weapon_vs=wh.vs_weapon if wh else None,
                row_band_index=i,
                active=True,
            )
        )
    ocr_ms = (time.perf_counter() - t_ocr0) * 1000.0
    if timings_out is not None:
        timings_out["setup_ms"] = setup_ms
        timings_out["weapons_ms"] = weapons_ms
        timings_out["ocr_ms"] = ocr_ms
        timings_out["total_ms"] = setup_ms + weapons_ms + ocr_ms

    return events, boxes, masks, band_active, weapon_hits


def draw_unified_debug(
    roi_bgr: object,
    boxes: list[tuple[int, int, int, int, str]],
    events: list[vkt.KillfeedEvent],
) -> object:
    vis = roi_bgr.copy()
    for x, y, w, h, _c in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (96, 96, 96), 1)
    for e in events:
        if e.row_band_index is None or e.row_band_index >= len(boxes):
            continue
        x, y, w, h, row_color = boxes[e.row_band_index][:5]
        if not e.active:
            col = (128, 128, 128)
            ex, ey, ew, eh = vkt.expand_row_box(x, y, w, h, roi_bgr.shape[1], roi_bgr.shape[0])
            cv2.rectangle(vis, (ex, ey), (ex + ew, ey + eh), col, 1)
            cv2.putText(
                vis,
                "(empty slot)",
                (ex, max(ey - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                col,
                1,
                cv2.LINE_AA,
            )
            continue
        col = (0, 255, 0) if row_color == "green" else (0, 0, 255)
        ex, ey, ew, eh = vkt.expand_row_box(x, y, w, h, roi_bgr.shape[1], roi_bgr.shape[0])
        cv2.rectangle(vis, (ex, ey), (ex + ew, ey + eh), col, 2)
        wtxt = f" {e.weapon}" if e.weapon else ""
        k, v = e.killer or "?", e.victim or "?"
        label = f"{k[:14]}->{v[:14]}{wtxt}"[:70]
        cv2.putText(
            vis,
            label,
            (ex, max(ey - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            col,
            1,
            cv2.LINE_AA,
        )
    return vis


def main() -> None:
    p = argparse.ArgumentParser(description="Parse killfeed: names (OCR) + weapons (optional templates).")
    p.add_argument("--image", type=Path, required=True, help="Screenshot (full monitor or killfeed crop).")
    p.add_argument(
        "--row-bands-json",
        type=Path,
        default=ROOT / "config" / "killfeed_row_bands.json",
        help="row_bands_frac JSON (default: config/killfeed_row_bands.json).",
    )
    p.add_argument(
        "--weapon-templates-json",
        type=Path,
        default=ROOT / "config" / "weapon_templates.json",
        help="Weapon template map JSON. Pass a non-existent path with --no-weapons to disable.",
    )
    p.add_argument("--no-weapons", action="store_true", help="Skip template matching (names only).")
    p.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "killfeed_parse.json",
        help="Write parsed rows here (default: killfeed_parse.json). Skipped with --no-write / --print-json.",
    )
    p.add_argument("--debug-out", type=Path, default=None, help="Write combined debug BGR PNG (skipped with --no-write / --print-json).")
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write --output-json or any debug PNGs; print a text summary to stdout only.",
    )
    p.add_argument(
        "--print-json",
        action="store_true",
        help="Print one JSON array to stdout only; no disk writes. Ignores --output-json and --debug-out.",
    )
    p.add_argument(
        "--ocr-engine",
        type=str,
        choices=["tesseract", "easyocr", "both"],
        default=vkt.DEFAULT_OCR_ENGINE,
    )
    p.add_argument(
        "--killfeed-rect",
        type=str,
        default=None,
        metavar="TOP,LEFT,WIDTH,HEIGHT",
        help="Override killfeed rect for cropping full screenshots.",
    )
    p.add_argument(
        "--min-band-highlight-px",
        type=int,
        default=280,
        metavar="N",
        help="Min green+red HSV pixels in a row band to treat it as an active killfeed row "
        "(below this: empty slot, skip OCR/weapons).",
    )
    p.add_argument(
        "--min-band-highlight-frac",
        type=float,
        default=0.001,
        metavar="0..1",
        help="Min highlight pixels as fraction of band area (combined with --min-band-highlight-px).",
    )
    p.add_argument(
        "--omit-inactive-bands",
        action="store_true",
        help="Write only active rows to JSON (default: all bands with active=false for empty slots).",
    )
    args = p.parse_args()

    if args.killfeed_rect:
        parts = [int(x.strip()) for x in args.killfeed_rect.split(",")]
        if len(parts) != 4 or any(x < 0 for x in parts):
            raise SystemExit("--killfeed-rect: need TOP,LEFT,WIDTH,HEIGHT")
        vkt.REGION_KILLFEED = {
            "top": parts[0],
            "left": parts[1],
            "width": parts[2],
            "height": parts[3],
        }

    img_path = args.image.resolve()
    if not img_path.is_file():
        raise SystemExit(f"Not a file: {img_path}")

    bands_path = args.row_bands_json.resolve()
    if not bands_path.is_file():
        raise SystemExit(f"Not a file: {bands_path}")
    bands = vkt.load_row_bands_json(bands_path)

    wjson = None if args.no_weapons else args.weapon_templates_json.resolve()
    if wjson is not None and not wjson.is_file():
        raise SystemExit(f"Weapon templates not found: {wjson} (use --no-weapons)")

    vkt.warm_easyocr_for_session(args.ocr_engine)

    full = cv2.imread(str(img_path))
    if full is None:
        raise SystemExit(f"Cannot read image: {img_path}")
    roi = vkt.crop_killfeed_region_if_possible(full)

    wp = WeaponMatchParams()
    now = time.time()
    events, boxes, masks, band_active, weapon_hits = parse_killfeed_roi_unified(
        roi,
        bands,
        templates_json=wjson,
        weapon_params=wp,
        ocr_engine=args.ocr_engine,
        now=now,
        highlight_min_abs=args.min_band_highlight_px,
        highlight_min_frac=args.min_band_highlight_frac,
    )

    json_rows = [e for e in events if e.active] if args.omit_inactive_bands else events

    no_write = bool(args.print_json or args.no_write)
    if args.print_json:
        print(json.dumps([asdict(e) for e in json_rows], ensure_ascii=False))
        return

    if not no_write:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in json_rows], f, indent=2, ensure_ascii=False)

    n_act = sum(1 for e in events if e.active)
    if no_write:
        print(
            f"Parsed {len(json_rows)} row(s) in output set; "
            f"{n_act} active / {len(events)} bands (no files written)"
        )
    else:
        print(
            f"Wrote {args.output_json} ({len(json_rows)} row(s) in file; "
            f"{n_act} active / {len(events)} bands)"
        )
    for e in events:
        if not e.active:
            print(f"  [inactive] band={e.row_band_index}  (empty slot)")
            continue
        wpart = f"  [{e.weapon} {e.weapon_score:.3f}]" if e.weapon else ""
        side = "enemy" if e.probable_enemy_kill else "ally"
        print(f"  [{side}] band={e.row_band_index}  {e.killer} -> {e.victim}{wpart}")

    if not no_write and args.debug_out is not None:
        vis = draw_unified_debug(roi, boxes, events)
        if weapon_hits is not None:
            for wh in weapon_hits:
                if wh is None:
                    continue
                cv2.rectangle(vis, (wh.x, wh.y), (wh.x + wh.w, wh.y + wh.h), (0, 255, 0), 1)
                cv2.putText(
                    vis,
                    f"{wh.weapon} {wh.score:.2f}",
                    (wh.x, max(10, wh.y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        vkt.draw_status_footer(vis, vkt.describe_ocr_compute_backend(args.ocr_engine))
        args.debug_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.debug_out.resolve()), vis)
        print(f"Debug image -> {args.debug_out.resolve()}")

    # Optional: same HSV masks as tracker debug
    if not no_write and args.debug_out is not None:
        dbg_dir = args.debug_out.parent / "parse_masks"
        vkt.save_killfeed_debug_images(
            dbg_dir,
            img_path.stem,
            annotated_bgr=None,
            masks=masks,
            footer_line=None,
            mode_tag="parse_killfeed masks",
        )


if __name__ == "__main__":
    main()
