import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import mss
import numpy as np
import pytesseract
import re
from typing import Optional

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import torch

    _EASYOCR_USE_GPU = torch.cuda.is_available()
except Exception:
    torch = None  # type: ignore
    _EASYOCR_USE_GPU = False


def cuda_gpu_name() -> str:
    """Human-readable CUDA device name, or empty if unavailable."""
    if not _EASYOCR_USE_GPU or torch is None:
        return ""
    try:
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return "CUDA"


def describe_ocr_compute_backend(ocr_engine: str) -> str:
    """Short summary: which OCR runs where (Tesseract=CPU; EasyOCR=GPU if CUDA)."""
    eng = (ocr_engine or DEFAULT_OCR_ENGINE).lower()
    bits: list[str] = []
    if eng in ("tesseract", "both"):
        bits.append("Tesseract=CPU")
    if eng in ("easyocr", "both"):
        if easyocr is None:
            bits.append("EasyOCR=missing")
        elif _EASYOCR_USE_GPU:
            name = cuda_gpu_name()
            bits.append(f"EasyOCR=GPU ({name})" if name else "EasyOCR=GPU")
        else:
            bits.append("EasyOCR=CPU (no CUDA)")
    return " | ".join(bits) if bits else "unknown"


def draw_status_footer(bgr: np.ndarray, line: str) -> None:
    """Overlay small status text on bottom of preview (BGR)."""
    if bgr is None or bgr.size == 0 or not line:
        return
    # OpenCV putText is unreliable with non-Latin device names
    safe = "".join(c if 32 <= ord(c) < 127 else "?" for c in line)[:100]
    h, w = bgr.shape[:2]
    y = min(h - 6, h - 1)
    x = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    for dx, dy in ((1, 1),):
        cv2.putText(bgr, safe, (x + dx, y + dy), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(bgr, safe, (x, y), font, scale, (0, 255, 255), thick, cv2.LINE_AA)


def opencv_highgui_available() -> bool:
    """False when using opencv-python-headless (e.g. after pip install easyocr)."""
    try:
        cv2.namedWindow("__vf_gui_probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__vf_gui_probe")
        return True
    except cv2.error:
        return False


# If Tesseract isn’t in PATH, set it manually
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- CONFIG ---
# Full-screen capture rect for the killfeed column only (MSS / crop on static images).
# Keep HEIGHT small: a tall box pulls in the combat report / death recap modal below the strip
# and produces bogus killer→victim rows (names from that UI). ~4 killfeed lines ≈ 140–200 px tall.
REGION_KILLFEED = {"top": 80, "left": 1300, "width": 600, "height": 180}

GREEN_LOW = np.array([35, 40, 80])
GREEN_HIGH = np.array([90, 255, 255])
RED_LOW_1 = np.array([0, 60, 80])
RED_HIGH_1 = np.array([10, 255, 255])
RED_LOW_2 = np.array([170, 60, 80])
RED_HIGH_2 = np.array([179, 255, 255])

MIN_WIDTH_ROW = 120
MIN_HEIGHT_ROW = 20
MIN_GREEN_PIXELS = 120
ASPECT_RATIO_MIN = 2.0
ASPECT_RATIO_MAX = 25.0

# Horizontal split: center is weapon/icons — OCR sides only (fractions of row width).
ROW_LEFT_END = 0.40
ROW_RIGHT_START = 0.60

# Text-focused inner windows inside each side (avoid portraits/icons at edges).
LEFT_TEXT_START_FRAC = 0.05
LEFT_TEXT_END_FRAC = 0.92
RIGHT_TEXT_START_FRAC = 0.08
RIGHT_TEXT_END_FRAC = 0.86
SPLIT_SEARCH_START_FRAC = 0.30
SPLIT_SEARCH_END_FRAC = 0.70
SPLIT_GAP_PADDING_FRAC = 0.04

REFRESH_DELAY = 0.2
DUPLICATE_PAIR_COOLDOWN = 3.0  # same killer→victim within this window = one event

OVERLAY_TXT = "overlay_stats.txt"
EVENTS_JSON = "killfeed_events.json"
MAX_STORED_EVENTS = 80

OCR_PSM_LINE = "--psm 7"  # single text line
OCR_UPSCALE = 2
# Include space for EasyOCR allowlist (multi-word Riot display names). Tesseract ``-c`` is built
# without spaces in the value so Windows ``pytesseract``/``shlex.split`` does not break the argv.
_OCR_CHARS_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_#"
OCR_CHAR_WHITELIST = _OCR_CHARS_ALNUM + " "
TESSERACT_CHAR_WHITELIST = _OCR_CHARS_ALNUM
DEFAULT_OCR_ENGINE = "easyocr"  # tesseract | easyocr | both

# EasyOCR: one readtext() per killfeed row (left+right names) instead of two — much faster live.
EASYOCR_ONE_PASS = True
# LRU cache of OCR results for identical row geometry (skips repeat work while killfeed is static).
ROW_OCR_CACHE_MAX = 64

# EasyOCR speed (defaults were ~2–3s/frame on GPU: huge canvas + 2x upscale + per-row detect).
# Product goal ~200–500 ms/frame often includes weapon + names; we only OCR names here (no weapon icons).
# Default "craft": DBNet (dbnet18) needs JIT deformable-conv; on Windows that usually needs
# MSVC + CUDA toolkit (CUDA_HOME) or the runtime errors with tensors on CUDA.
EASYOCR_DETECT_NETWORK = "craft"  # "dbnet18" often faster when DCN extensions build OK
_easyocr_detect_network = EASYOCR_DETECT_NETWORK  # may downgrade to craft after a failed readtext
_easyocr_dcn_fallback_logged = False
EASYOCR_CANVAS_SIZE = 640  # EasyOCR default 2560 makes detection very slow
EASYOCR_MAG_RATIO = 1.0
EASYOCR_MIN_SIZE = 10
EASYOCR_ROW_MAX_WIDTH = 560
EASYOCR_ROW_MAX_HEIGHT = 56
EASYOCR_ROW_MIN_HEIGHT = 22  # upscale tiny strips slightly for recognition
# One detector pass for all rows in the frame (vertical stack).
EASYOCR_STACK_ROWS = True

# Color box often covers only left side; expand to include full row (victim on right).
ROW_EXPAND_LEFT = 8
ROW_EXPAND_RIGHT = 240
ROW_EXPAND_TOP = 2
ROW_EXPAND_BOTTOM = 2

TIMINGS_JSONL = "analysis_timings.jsonl"
LIVE_LOG_EVERY_FRAMES = 10  # avoid huge logs in real-time
SCREENSHOT_DIR = "temp/screenshots"
SCREENSHOT_COOLDOWN = 1.0  # seconds between full-screen captures
# Live mode: save full-monitor PNG when new killfeed rows are detected (overridable via CLI).
SAVE_FULLSCREEN_ON_KILLFEED = True

_easyocr_reader = None
_easyocr_reader_sig: tuple | None = None
_row_ocr_cache: OrderedDict[tuple, tuple[str, str, str, str]] = OrderedDict()


def frame_safe_row_cache_key(crop: np.ndarray, *geometry: object) -> tuple:
    """Return a cache identity that cannot leak OCR text across different pixels."""
    if crop is None or crop.size == 0:
        content_hash = "empty"
        shape: tuple[int, ...] = ()
    else:
        contiguous = np.ascontiguousarray(crop)
        content_hash = hashlib.sha256(contiguous.tobytes()).hexdigest()
        shape = tuple(int(value) for value in contiguous.shape)
    recognition = (
        DEFAULT_OCR_ENGINE,
        EASYOCR_ONE_PASS,
        EASYOCR_STACK_ROWS,
        _easyocr_detect_network,
        _EASYOCR_USE_GPU,
        EASYOCR_CANVAS_SIZE,
        EASYOCR_MAG_RATIO,
        EASYOCR_MIN_SIZE,
    )
    return (*geometry, shape, content_hash, recognition)


@dataclass
class KillfeedEvent:
    killer: str
    victim: str
    row_color: str
    probable_enemy_kill: bool
    raw_left: str
    raw_right: str
    t: float
    weapon: str | None = None
    weapon_score: float | None = None
    weapon_margin: float | None = None
    weapon_vs: str | None = None
    row_band_index: int | None = None
    active: bool = True


def normalize_ocr_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9#\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def upscale_for_ocr(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    if w < 2 or h < 2:
        return img
    return cv2.resize(img, (w * OCR_UPSCALE, h * OCR_UPSCALE), interpolation=cv2.INTER_CUBIC)


def prepare_easyocr_row_image(bgr: np.ndarray) -> np.ndarray:
    """Resize killfeed row for fast EasyOCR (avoid 2x upscale + huge detector input)."""
    if bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    if h < EASYOCR_ROW_MIN_HEIGHT:
        s = EASYOCR_ROW_MIN_HEIGHT / max(h, 1)
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
        h, w = bgr.shape[:2]
    scale = min(
        1.0,
        EASYOCR_ROW_MAX_WIDTH / max(w, 1),
        EASYOCR_ROW_MAX_HEIGHT / max(h, 1),
    )
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr


def _easyocr_row_band(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    pad_y = max(1, int(h * 0.10))
    band = crop[pad_y : max(pad_y + 1, h - pad_y), :]
    return band if band.size > 0 else crop


def _easyocr_readtext_kwargs() -> dict:
    return {
        "decoder": "greedy",
        "batch_size": 16,
        "detail": 1,
        "paragraph": False,
        "width_ths": 0.85,
        "height_ths": 0.85,
        "canvas_size": EASYOCR_CANVAS_SIZE,
        "mag_ratio": EASYOCR_MAG_RATIO,
        "min_size": EASYOCR_MIN_SIZE,
        "allowlist": OCR_CHAR_WHITELIST,
        "text_threshold": 0.65,
        "low_text": 0.35,
        "link_threshold": 0.45,
        "max_candidates": 40,
    }


def _easyocr_inference_context():
    if torch is not None:
        return torch.inference_mode()
    return nullcontext()


def ocr_line_region(bgr: np.ndarray) -> str:
    """OCR a thin strip (one name)."""
    if bgr is None or bgr.size == 0:
        return ""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Try two preprocess variants and keep the one with more alphanumeric signal.
    var1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    var2 = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        7,
    )
    # Valorant nicknames are often bright/white; isolate bright low-saturation text.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 70, 255]))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    var3 = white_mask

    cfg = f"{OCR_PSM_LINE} -c tessedit_char_whitelist={TESSERACT_CHAR_WHITELIST}"
    cand = []
    for prepared in (var1, var2, var3):
        prepared = upscale_for_ocr(prepared)
        text = pytesseract.image_to_string(prepared, config=cfg).strip()
        text = normalize_ocr_text(text)
        score = sum(ch.isalnum() for ch in text)
        cand.append((score, text))

    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1] if cand else ""


def _easyocr_invalidate_reader() -> None:
    global _easyocr_reader, _easyocr_reader_sig
    _easyocr_reader = None
    _easyocr_reader_sig = None


def get_easyocr_reader() -> Optional["easyocr.Reader"]:
    global _easyocr_reader, _easyocr_reader_sig
    if easyocr is None:
        return None
    sig = (_easyocr_detect_network, _EASYOCR_USE_GPU)
    if _easyocr_reader is not None and _easyocr_reader_sig != sig:
        _easyocr_reader = None
    if _easyocr_reader is None:
        if _EASYOCR_USE_GPU and torch is not None:
            torch.backends.cudnn.benchmark = True
        _easyocr_reader = easyocr.Reader(
            ["en"],
            gpu=_EASYOCR_USE_GPU,
            verbose=False,
            detect_network=_easyocr_detect_network,
            quantize=True,
            cudnn_benchmark=True,
        )
        _easyocr_reader_sig = sig
    return _easyocr_reader


def easyocr_readtext_rgb(rgb: np.ndarray) -> list:
    """Run EasyOCR readtext; if DBNet deformable-conv is missing, fall back to CRAFT once."""
    global _easyocr_dcn_fallback_logged, _easyocr_detect_network
    reader = get_easyocr_reader()
    if reader is None:
        return []
    kw = _easyocr_readtext_kwargs()
    try:
        with _easyocr_inference_context():
            return reader.readtext(rgb, **kw)
    except RuntimeError as e:
        msg = str(e).lower()
        if _easyocr_detect_network != "craft" and (
            "deform_conv" in msg or "deformable" in msg or "dcn" in msg
        ):
            if not _easyocr_dcn_fallback_logged:
                print(
                    "EasyOCR: DBNet needs compiled deformable-conv (CUDA_HOME + MSVC on Windows); "
                    "switching to CRAFT detector.",
                    file=sys.stderr,
                )
                _easyocr_dcn_fallback_logged = True
            _easyocr_detect_network = "craft"
            _easyocr_invalidate_reader()
            reader = get_easyocr_reader()
            if reader is None:
                return []
            with _easyocr_inference_context():
                return reader.readtext(rgb, **kw)
        raise


def _row_cache_get(key: tuple) -> tuple[str, str, str, str] | None:
    global _row_ocr_cache
    if key not in _row_ocr_cache:
        return None
    _row_ocr_cache.move_to_end(key)
    return _row_ocr_cache[key]


def _row_cache_put(key: tuple, value: tuple[str, str, str, str]) -> None:
    global _row_ocr_cache
    _row_ocr_cache[key] = value
    _row_ocr_cache.move_to_end(key)
    while len(_row_ocr_cache) > ROW_OCR_CACHE_MAX:
        _row_ocr_cache.popitem(last=False)


def ocr_line_region_easyocr(bgr: np.ndarray) -> tuple[str, float]:
    if bgr is None or bgr.size == 0:
        return "", 0.0
    if get_easyocr_reader() is None:
        return "", 0.0
    rgb = cv2.cvtColor(prepare_easyocr_row_image(bgr), cv2.COLOR_BGR2RGB)
    results = easyocr_readtext_rgb(rgb)
    if not results:
        return "", 0.0
    parts = []
    confs = []
    for _, text, conf in results:
        cleaned = normalize_ocr_text(text)
        if cleaned:
            parts.append(cleaned)
            confs.append(float(conf))
    if not parts:
        return "", 0.0
    joined = " ".join(parts).strip()
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return joined, avg_conf


def _kv_from_easyocr_detections(
    raw_results: list,
) -> tuple[str, str, str, str]:
    items: list[tuple[float, str, float]] = []
    for bbox, text, conf in raw_results:
        cleaned = normalize_ocr_text(text)
        if len(cleaned) < 2:
            continue
        xs = [float(p[0]) for p in bbox]
        cx = sum(xs) / max(len(xs), 1)
        items.append((cx, cleaned, float(conf)))
    if not items:
        return "", "", "", ""
    return _kv_from_line_items(items)


def easyocr_parse_stacked_bands(bands: list[np.ndarray]) -> list[tuple[str, str, str, str]]:
    """One detector pass for multiple killfeed rows (vertical stack)."""
    reader = get_easyocr_reader()
    if reader is None or not bands:
        return [("", "", "", "")] * len(bands)
    gap = 6
    prepared = [prepare_easyocr_row_image(b) for b in bands]
    heights = [p.shape[0] for p in prepared]
    max_w = max(p.shape[1] for p in prepared)
    total_h = sum(heights) + gap * (len(prepared) - 1)
    canvas = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    y = 0
    spans: list[tuple[int, int]] = []
    for p in prepared:
        h0, w0 = p.shape[:2]
        canvas[y : y + h0, :w0] = p
        spans.append((y, y + h0))
        y += h0 + gap
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    raw = easyocr_readtext_rgb(rgb)
    mids = [(a + b) / 2.0 for a, b in spans]
    row_buckets: list[list[tuple[float, str, float]]] = [[] for _ in bands]
    for bbox, text, conf in raw:
        cleaned = normalize_ocr_text(text)
        if len(cleaned) < 2:
            continue
        cy = sum(float(p[1]) for p in bbox) / 4.0
        cx = sum(float(p[0]) for p in bbox) / 4.0
        bi = min(range(len(spans)), key=lambda i: abs(cy - mids[i]))
        row_buckets[bi].append((cx, cleaned, float(conf)))
    out: list[tuple[str, str, str, str]] = []
    for bucket in row_buckets:
        if not bucket:
            out.append(("", "", "", ""))
        else:
            out.append(_kv_from_line_items(bucket))
    return out


def _kv_from_line_items(items: list[tuple[float, str, float]]) -> tuple[str, str, str, str]:
    items = sorted(items, key=lambda t: t[0])
    if len(items) == 1:
        k = items[0][1]
        return k, "", k, ""
    return items[0][1], items[-1][1], items[0][1], items[-1][1]


def ocr_killfeed_row_easyocr_one_pass(crop: np.ndarray) -> tuple[str, str, str, str]:
    """Single EasyOCR pass on full row; killer = leftmost text box, victim = rightmost."""
    if crop is None or crop.size == 0:
        return "", "", "", ""
    if get_easyocr_reader() is None:
        return "", "", "", ""
    band = _easyocr_row_band(crop)
    rgb = cv2.cvtColor(prepare_easyocr_row_image(band), cv2.COLOR_BGR2RGB)
    results = easyocr_readtext_rgb(rgb)
    return _kv_from_easyocr_detections(results)


_easyocr_warmed_signatures: set[tuple[object, ...]] = set()


def warm_easyocr_for_session(ocr_engine: str) -> None:
    """Prime detector + recognizer and both single-row and stacked paths (blank readtext is not enough)."""
    eng = (ocr_engine or DEFAULT_OCR_ENGINE).lower()
    if eng not in ("easyocr", "both") or easyocr is None:
        return
    if get_easyocr_reader() is None:
        return
    signature = (_easyocr_detect_network, _EASYOCR_USE_GPU)
    if signature in _easyocr_warmed_signatures:
        return
    h, w = 56, 520
    strip = np.full((h, w, 3), 32, dtype=np.uint8)
    cv2.putText(
        strip,
        "PlayerOne",
        (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        strip,
        "PlayerTwo",
        (270, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    try:
        with _easyocr_inference_context():
            _ = ocr_killfeed_row_easyocr_one_pass(strip)
            easyocr_parse_stacked_bands([strip, strip.copy()])
        if torch is not None and _EASYOCR_USE_GPU:
            torch.cuda.synchronize()
        _easyocr_warmed_signatures.add(signature)
    except Exception:
        pass


def score_ocr_text(text: str, conf: float = 0.0) -> float:
    alnum = sum(ch.isalnum() for ch in text)
    length_penalty = max(0, len(text) - 18) * 0.2
    return alnum + conf * 5.0 - length_penalty


def pick_better_ocr_string(a: str, b: str, conf_a: float = 0.0, conf_b: float = 0.0) -> str:
    a, b = a.strip(), b.strip()
    if not a:
        return b
    if not b:
        return a
    return a if score_ocr_text(a, conf_a) >= score_ocr_text(b, conf_b) else b


def ocr_line_region_fused(bgr: np.ndarray, ocr_engine: str) -> str:
    ocr_engine = (ocr_engine or DEFAULT_OCR_ENGINE).lower()
    if ocr_engine == "tesseract":
        return ocr_line_region(bgr)
    if ocr_engine == "easyocr":
        text, _ = ocr_line_region_easyocr(bgr)
        return text

    # both: pick best-scored output between engines
    t_text = ocr_line_region(bgr)
    e_text, e_conf = ocr_line_region_easyocr(bgr)
    t_score = score_ocr_text(t_text, conf=0.0)
    e_score = score_ocr_text(e_text, conf=e_conf)
    return e_text if e_score > t_score else t_text


def estimate_text_split_x(crop: np.ndarray) -> int:
    """
    Estimate x-coordinate separating killer/victim text.
    Uses low-density valley of white-text columns near center.
    """
    h, w = crop.shape[:2]
    if w < 20:
        return w // 2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([179, 75, 255]))
    white_mask = cv2.morphologyEx(
        white_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
    )

    col_density = (white_mask > 0).sum(axis=0).astype(np.float32)
    if col_density.max() > 0:
        col_density /= col_density.max()
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    col_density = np.convolve(col_density, kernel, mode="same")

    sx = int(w * SPLIT_SEARCH_START_FRAC)
    ex = int(w * SPLIT_SEARCH_END_FRAC)
    sx = max(1, min(sx, w - 2))
    ex = max(sx + 1, min(ex, w - 1))
    valley_idx = int(np.argmin(col_density[sx:ex])) + sx
    return valley_idx


# If one HSV contour spans two stacked killfeed lines, split into two boxes before OCR.
TALL_ROW_SPLIT_MIN_H = 44
TALL_ROW_SPLIT_RATIO = 1.65
TALL_ROW_SPLIT_GAP = 2


def split_tall_killfeed_row_boxes(
    rows: list[tuple[int, int, int, int, str]],
) -> list[tuple[int, int, int, int, str]]:
    """Split unusually tall row contours (often two kill lines merged in the mask)."""
    if not rows:
        return rows
    hs = [r[3] for r in rows]
    med = max(float(np.median(hs)), 22.0)
    lonely = len(rows) == 1
    out: list[tuple[int, int, int, int, str]] = []
    for x, y, w, h, c in rows:
        should_split = h >= TALL_ROW_SPLIT_MIN_H and (
            lonely or h >= med * TALL_ROW_SPLIT_RATIO
        )
        if should_split:
            mid = h // 2
            h1 = max(MIN_HEIGHT_ROW, mid - 1)
            h2 = max(MIN_HEIGHT_ROW, h - h1 - TALL_ROW_SPLIT_GAP)
            out.append((x, y, w, h1, c))
            out.append((x, y + h1 + TALL_ROW_SPLIT_GAP, w, h2, c))
        else:
            out.append((x, y, w, h, c))
    out.sort(key=lambda b: b[1])
    return out


def _compact_name_key(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().lower())


def prune_fragment_killfeed_rows(
    row_items: list[tuple[dict, str, str, str, str]],
) -> list[tuple[dict, str, str, str, str]]:
    """
    Drop rows that are usually OCR/mask artifacts:
    - killer matches another row's victim but victim is missing or '?' (duplicate strip).
    """
    if len(row_items) <= 1:
        return row_items
    victims_compact = [_compact_name_key(v) for _, _, v, _, _ in row_items]
    keep = [True] * len(row_items)
    for i, (_, k, v, _, _) in enumerate(row_items):
        v_st = (v or "").strip()
        if v_st not in ("", "?"):
            continue
        ki = _compact_name_key(k)
        if not ki:
            continue
        for j, vc in enumerate(victims_compact):
            if j == i or not vc:
                continue
            if ki == vc:
                keep[i] = False
                break
    return [row_items[i] for i in range(len(row_items)) if keep[i]]


def expand_row_box(
    x: int, y: int, w: int, h: int, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    ex = max(0, x - ROW_EXPAND_LEFT)
    ey = max(0, y - ROW_EXPAND_TOP)
    ex2 = min(frame_w, x + w + ROW_EXPAND_RIGHT)
    ey2 = min(frame_h, y + h + ROW_EXPAND_BOTTOM)
    return ex, ey, max(1, ex2 - ex), max(1, ey2 - ey)


def crop_killfeed_region_if_possible(img: np.ndarray) -> np.ndarray:
    """If input looks like a full screenshot, crop configured killfeed region."""
    h, w = img.shape[:2]
    x, y = REGION_KILLFEED["left"], REGION_KILLFEED["top"]
    rw, rh = REGION_KILLFEED["width"], REGION_KILLFEED["height"]
    if w >= x + rw and h >= y + rh:
        return img[y : y + rh, x : x + rw]
    return img


def build_killfeed_name_rois(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left/right strips for Tesseract / per-side EasyOCR."""
    h, w = crop.shape[:2]
    split_x = estimate_text_split_x(crop)
    gap = max(2, int(w * SPLIT_GAP_PADDING_FRAC))
    lx1, lx2 = 0, max(1, split_x - gap)
    rx1, rx2 = min(w - 1, split_x + gap), w

    if lx2 <= lx1 + 2 or rx2 <= rx1 + 2:
        lx1, lx2 = 0, max(1, int(w * ROW_LEFT_END))
        rx1, rx2 = min(w - 1, int(w * ROW_RIGHT_START)), w

    left_roi = crop[0:h, lx1:lx2]
    right_roi = crop[0:h, rx1:rx2]

    pad_y = max(1, int(h * 0.10))
    ly1 = min(h - 1, pad_y)
    ly2 = max(ly1 + 1, h - pad_y)

    lw = left_roi.shape[1]
    rw = right_roi.shape[1]
    lx_in1 = int(lw * LEFT_TEXT_START_FRAC)
    lx_in2 = max(lx_in1 + 1, int(lw * LEFT_TEXT_END_FRAC))
    rx_in1 = int(rw * RIGHT_TEXT_START_FRAC)
    rx_in2 = max(rx_in1 + 1, int(rw * RIGHT_TEXT_END_FRAC))

    return left_roi[ly1:ly2, lx_in1:lx_in2], right_roi[ly1:ly2, rx_in1:rx_in2]


def parse_killer_victim_from_row_crop(
    crop: np.ndarray,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    row_cache_key: tuple | None = None,
) -> tuple[str, str, str, str]:
    """Split killfeed row into left (killer) and right (victim) OCR."""
    h, w = crop.shape[:2]
    if w < 8 or h < 4:
        return "", "", "", ""

    if row_cache_key is not None:
        cached = _row_cache_get(row_cache_key)
        if cached is not None:
            return cached

    eng = (ocr_engine or DEFAULT_OCR_ENGINE).lower()
    out: tuple[str, str, str, str]

    if eng == "easyocr" and EASYOCR_ONE_PASS:
        out = ocr_killfeed_row_easyocr_one_pass(crop)
    elif eng == "both" and EASYOCR_ONE_PASS:
        ek, ev, _, _ = ocr_killfeed_row_easyocr_one_pass(crop)
        left_roi, right_roi = build_killfeed_name_rois(crop)
        tl = ocr_line_region(left_roi)
        tr = ocr_line_region(right_roi)
        killer = pick_better_ocr_string(ek, tl)
        victim = pick_better_ocr_string(ev, tr)
        out = (killer, victim, ek, ev)
    elif eng == "tesseract":
        left_roi, right_roi = build_killfeed_name_rois(crop)
        tl = ocr_line_region(left_roi)
        tr = ocr_line_region(right_roi)
        out = (tl, tr, tl, tr)
    else:
        left_roi, right_roi = build_killfeed_name_rois(crop)
        raw_left = ocr_line_region_fused(left_roi, ocr_engine=ocr_engine)
        raw_right = ocr_line_region_fused(right_roi, ocr_engine=ocr_engine)
        out = (raw_left, raw_right, raw_left, raw_right)

    if row_cache_key is not None:
        _row_cache_put(row_cache_key, out)
    return out


def detect_green_row_boxes(frame: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_GREEN_PIXELS:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h else 0
        if w > MIN_WIDTH_ROW and h > MIN_HEIGHT_ROW and ASPECT_RATIO_MIN < aspect < ASPECT_RATIO_MAX:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[1])
    return boxes, mask


def green_red_masks_bgr(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Morphology-cleaned green / red HSV masks for the killfeed ROI (full ``frame`` size)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    red_mask_1 = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1)
    red_mask_2 = cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return green_mask, red_mask


def load_row_bands_json(path: Path) -> list[tuple[float, float]]:
    """Load ``row_bands_frac`` from JSON: list of [y0, y1] in [0, 1] relative to ROI height (0=top)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SystemExit(f"Cannot read row bands file: {path} ({e})") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON: {path}: {e}") from e
    raw = data.get("row_bands_frac")
    if raw is None:
        raise SystemExit(f"JSON must contain 'row_bands_frac': list of [y0,y1] fractions: {path}")
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"'row_bands_frac' must be a non-empty list: {path}")
    out: list[tuple[float, float]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise SystemExit(f"row_bands_frac[{i}] must be [y0, y1]: {path}")
        y0, y1 = float(item[0]), float(item[1])
        if not (0 <= y0 < y1 <= 1.0):
            raise SystemExit(f"row_bands_frac[{i}] need 0 <= y0 < y1 <= 1: got {y0}, {y1}")
        out.append((y0, y1))
    return out


def infer_band_color_from_masks(
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    y0: int,
    y1: int,
) -> str:
    """Pick ``green`` vs ``red`` for a horizontal band from HSV highlight density."""
    g = int(cv2.countNonZero(green_mask[y0:y1, :]))
    r = int(cv2.countNonZero(red_mask[y0:y1, :]))
    if g == 0 and r == 0:
        return "red"
    tie = 1.08
    if g >= r * tie:
        return "green"
    if r >= g * tie:
        return "red"
    return "green" if g >= r else "red"


def killfeed_row_band_is_active(
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    y0: int,
    y1: int,
    frame_w: int,
    *,
    min_abs: int = 280,
    min_frac: float = 0.001,
) -> bool:
    """
    True if the horizontal strip likely contains a Valorant killfeed highlight (green/red bar).

    When fixed ``row_bands_frac`` slots sit over empty map/background, green+red HSV counts are
    tiny; skip OCR / weapon work for those bands.
    """
    g = int(cv2.countNonZero(green_mask[y0:y1, :]))
    r = int(cv2.countNonZero(red_mask[y0:y1, :]))
    area = max(1, (y1 - y0) * frame_w)
    need = max(int(min_abs), int(area * min_frac))
    return (g + r) >= need


def fixed_row_boxes_from_bands(
    frame: np.ndarray,
    bands: list[tuple[float, float]],
) -> tuple[list[tuple[int, int, int, int, str]], dict[str, np.ndarray]]:
    """
    Build full-width row boxes from fractional Y bands (same convention as weapon matcher).
    Row color comes from green vs red mask counts inside each band (no contour detection).
    """
    roi_h, roi_w = frame.shape[:2]
    green_mask, red_mask = green_red_masks_bgr(frame)
    masks = {"green": green_mask, "red": red_mask}
    rows: list[tuple[int, int, int, int, str]] = []
    for y0f, y1f in bands:
        y0 = min(roi_h - 2, max(0, int(roi_h * y0f)))
        y1 = min(roi_h, max(y0 + 2, int(round(roi_h * y1f))))
        color = infer_band_color_from_masks(green_mask, red_mask, y0, y1)
        rows.append((0, y0, roi_w, y1 - y0, color))
    return rows, masks


def detect_killfeed_row_boxes(
    frame: np.ndarray,
) -> tuple[list[tuple[int, int, int, int, str]], dict[str, np.ndarray]]:
    """
    Detect killfeed rows with color tags:
    - green rows (typically own-team-related side)
    - red rows (typically enemy-team-related side)
    """
    green_mask, red_mask = green_red_masks_bgr(frame)

    rows: list[tuple[int, int, int, int, str]] = []
    for color_name, mask in (("green", green_mask), ("red", red_mask)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_GREEN_PIXELS:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h) if h else 0
            if w > MIN_WIDTH_ROW and h > MIN_HEIGHT_ROW and ASPECT_RATIO_MIN < aspect < ASPECT_RATIO_MAX:
                rows.append((x, y, w, h, color_name))

    rows.sort(key=lambda b: b[1])
    return rows, {"green": green_mask, "red": red_mask}


def is_duplicate_pair(
    killer: str,
    victim: str,
    recent: list[tuple[str, str, float]],
    now: float,
    window: float,
) -> bool:
    k, v = killer.lower(), victim.lower()
    for rk, rv, t in recent:
        if now - t > window:
            continue
        if rk == k and rv == v:
            return True
    return False


def prune_recent(recent: list[tuple[str, str, float]], now: float, window: float) -> None:
    keep = [x for x in recent if now - x[2] <= window * 2]
    recent.clear()
    recent.extend(keep)

def append_timing_log(record: dict) -> None:
    """Append a JSON record (JSONL) with OCR/detection timing stats."""
    with open(TIMINGS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_fullscreen_capture(sct: mss.mss, out_dir: Path, now: float) -> str | None:
    """Save full monitor screenshot and return path, or None on failure."""
    monitor = sct.monitors[1]  # full primary monitor
    shot = np.ascontiguousarray(np.array(sct.grab(monitor))[:, :, :3])
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    millis = int((now % 1) * 1000)
    out_path = out_dir / f"killfeed_{stamp}_{millis:03d}.png"
    ok = cv2.imwrite(str(out_path), shot)
    return str(out_path) if ok else None


def process_frame(
    frame: np.ndarray,
    now: float,
    recent_pairs: list[tuple[str, str, float]] | None,
    draw: np.ndarray | None = None,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    show_debug_windows: bool = True,
    row_bands_frac: list[tuple[float, float]] | None = None,
) -> tuple[
    list[KillfeedEvent],
    list[tuple[int, int, int, int, str]],
    dict,
    dict[str, np.ndarray],
]:
    """
    Parse all killfeed rows in frame.
    If recent_pairs is not None, skip duplicate (killer, victim) within DUPLICATE_PAIR_COOLDOWN.

    If ``row_bands_frac`` is set (from ``config/killfeed_row_bands.json``), row geometry follows
    those Y fractions of ROI height instead of HSV contour detection (aligns with weapon matcher).

    Returns ``(events, boxes, timing, masks)``. Use :func:`save_killfeed_debug_images` with the
    annotated ROI copy to write PNGs when ``--debug-dir`` is set.
    """
    t_total_0 = time.perf_counter()

    t_detect_0 = time.perf_counter()
    if row_bands_frac is not None:
        boxes, masks = fixed_row_boxes_from_bands(frame, row_bands_frac)
        detect_src = "fixed_bands"
    else:
        boxes, masks = detect_killfeed_row_boxes(frame)
        boxes = split_tall_killfeed_row_boxes(boxes)
        detect_src = "hsv_contours"
    detect_ms = (time.perf_counter() - t_detect_0) * 1000.0

    events: list[KillfeedEvent] = []
    ocr_row_cache_hits = 0

    frame_h, frame_w = frame.shape[:2]
    if draw is not None:
        for bx, by, bw, bh, _ in boxes:
            cv2.rectangle(draw, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)

    row_work: list[dict] = []
    for (x, y, w, h, row_color) in boxes:
        ex, ey, ew, eh = expand_row_box(x, y, w, h, frame_w, frame_h)
        crop = frame[ey : ey + eh, ex : ex + ew]
        row_key = frame_safe_row_cache_key(
            crop, ex >> 2, ey >> 2, ew >> 2, eh >> 2, row_color, (ocr_engine or DEFAULT_OCR_ENGINE).lower()
        )
        row_work.append(
            {
                "row_key": row_key,
                "crop": crop,
                "row_color": row_color,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "ex": ex,
                "ey": ey,
                "ew": ew,
                "eh": eh,
            }
        )

    t_parse_0 = time.perf_counter()
    parsed_by_key: dict[tuple, tuple[str, str, str, str]] = {}
    eng = (ocr_engine or DEFAULT_OCR_ENGINE).lower()

    if eng == "easyocr" and EASYOCR_ONE_PASS and EASYOCR_STACK_ROWS and len(row_work) >= 2:
        need: list[dict] = []
        for rw in row_work:
            key = rw["row_key"]
            hit = _row_cache_get(key)
            if hit is not None:
                ocr_row_cache_hits += 1
                parsed_by_key[key] = hit
            else:
                need.append(rw)
        if len(need) > 1:
            bands = [_easyocr_row_band(rw["crop"]) for rw in need]
            outs = easyocr_parse_stacked_bands(bands)
            while len(outs) < len(need):
                outs.append(("", "", "", ""))
            for rw, out in zip(need, outs[: len(need)]):
                _row_cache_put(rw["row_key"], out)
                parsed_by_key[rw["row_key"]] = out
        elif len(need) == 1:
            rw = need[0]
            out = parse_killer_victim_from_row_crop(
                rw["crop"], ocr_engine="easyocr", row_cache_key=rw["row_key"]
            )
            parsed_by_key[rw["row_key"]] = out
    else:
        for rw in row_work:
            key = rw["row_key"]
            if key in _row_ocr_cache:
                ocr_row_cache_hits += 1
            killer, victim, raw_l, raw_r = parse_killer_victim_from_row_crop(
                rw["crop"], ocr_engine=ocr_engine, row_cache_key=key
            )
            parsed_by_key[key] = (killer, victim, raw_l, raw_r)

    parse_ms_total = (time.perf_counter() - t_parse_0) * 1000.0
    rows_parsed = len(row_work)

    row_items: list[tuple[dict, str, str, str, str]] = []
    for rw in row_work:
        killer, victim, raw_l, raw_r = parsed_by_key[rw["row_key"]]
        if killer or victim:
            row_items.append((rw, killer, victim, raw_l, raw_r))
    row_items.sort(key=lambda t: t[0]["y"])
    row_items = prune_fragment_killfeed_rows(row_items)

    for rw, killer, victim, raw_l, raw_r in row_items:
        x, y, w, h = rw["x"], rw["y"], rw["w"], rw["h"]
        ex, ey, ew, eh = rw["ex"], rw["ey"], rw["ew"], rw["eh"]
        row_color = rw["row_color"]

        dup = False
        if recent_pairs is not None:
            prune_recent(recent_pairs, now, DUPLICATE_PAIR_COOLDOWN)
            if killer and victim and is_duplicate_pair(killer, victim, recent_pairs, now, DUPLICATE_PAIR_COOLDOWN):
                dup = True
            elif killer and victim:
                recent_pairs.append((killer.lower(), victim.lower(), now))

        if draw is not None:
            color = (0, 255, 0) if row_color == "green" else (0, 0, 255)
            cv2.rectangle(draw, (x, y), (x + w, y + h), color, 1)
            cv2.rectangle(draw, (ex, ey), (ex + ew, ey + eh), color, 2)
            label = f"{killer[:12]} -> {victim[:12]}" if killer or victim else "?"
            if dup:
                label = f"(seen) {label}"
            cv2.putText(
                draw,
                f"{row_color} {label}",
                (ex, max(ey - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 200, 255) if dup else (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if dup:
            continue

        events.append(
            KillfeedEvent(
                killer=killer or "?",
                victim=victim or "?",
                row_color=row_color,
                probable_enemy_kill=row_color == "red",
                raw_left=raw_l,
                raw_right=raw_r,
                t=now,
            )
        )

    timing = {
        "t_total_ms": (time.perf_counter() - t_total_0) * 1000.0,
        "t_detect_ms": detect_ms,
        "row_detect_mode": detect_src,
        "t_parse_ms_total": parse_ms_total,
        "rows_detected": len(boxes),
        "rows_green": sum(1 for b in boxes if b[4] == "green"),
        "rows_red": sum(1 for b in boxes if b[4] == "red"),
        "rows_parsed": rows_parsed,
        "events_emitted": len(events),
        "ocr_engine": ocr_engine,
        "ocr_row_cache_hits": ocr_row_cache_hits,
        "ts": time.time(),
    }

    if draw is not None and show_debug_windows:
        try:
            cv2.imshow("Killfeed mask green", masks["green"])
            cv2.imshow("Killfeed mask red", masks["red"])
        except cv2.error:
            pass

    return events, boxes, timing, masks


def save_killfeed_debug_images(
    out_dir: Path,
    base_name: str,
    *,
    annotated_bgr: np.ndarray | None,
    masks: dict[str, np.ndarray],
    footer_line: str | None = None,
    mode_tag: str | None = None,
) -> list[Path]:
    """
    Write annotated ROI (boxes + OCR labels), per-channel HSV masks, and a side-by-side mask preview.
    Filenames: ``{base_name}_annotated.png``, ``_mask_green.png``, ``_mask_red.png``, ``_masks_sidebyside.png``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)[:120]

    if annotated_bgr is not None:
        vis = annotated_bgr.copy()
        if mode_tag:
            cv2.putText(
                vis,
                mode_tag[:80],
                (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if footer_line:
            draw_status_footer(vis, footer_line)
        p = out_dir / f"{safe}_annotated.png"
        cv2.imwrite(str(p), vis)
        written.append(p)

    g = masks.get("green")
    r = masks.get("red")
    if g is not None:
        p = out_dir / f"{safe}_mask_green.png"
        cv2.imwrite(str(p), g)
        written.append(p)
    if r is not None:
        p = out_dir / f"{safe}_mask_red.png"
        cv2.imwrite(str(p), r)
        written.append(p)
    if g is not None and r is not None and g.shape == r.shape:
        gh = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        rh = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        cv2.putText(gh, "green HSV", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(rh, "red HSV", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        combo = np.hstack([gh, rh])
        p = out_dir / f"{safe}_masks_sidebyside.png"
        cv2.imwrite(str(p), combo)
        written.append(p)

    return written


def load_bgr(path: str | Path) -> np.ndarray:
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return img


def save_outputs(events_log: list[KillfeedEvent]) -> None:
    last = events_log[-MAX_STORED_EVENTS:]
    payload = [asdict(e) for e in last]
    with open(EVENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(OVERLAY_TXT, "w", encoding="utf-8") as f:
        if not events_log:
            f.write("Killfeed: (no events yet)")
        else:
            e = events_log[-1]
            f.write(f"Last: {e.killer} -> {e.victim}  ({e.row_color})")


def run_static_images(
    paths: list[Path],
    show: bool,
    ocr_engine: str,
    row_bands_frac: list[tuple[float, float]] | None = None,
    debug_dir: Path | None = None,
) -> None:
    all_events: list[KillfeedEvent] = []
    warm_easyocr_for_session(ocr_engine)

    for p in paths:
        full = load_bgr(p)
        frame = crop_killfeed_region_if_possible(full)
        now = time.time()
        need_draw = show or debug_dir is not None
        preview = frame.copy() if need_draw else None
        events, _, timing, masks = process_frame(
            frame,
            now,
            recent_pairs=None,
            draw=preview,
            ocr_engine=ocr_engine,
            row_bands_frac=row_bands_frac,
        )
        all_events.extend(events)
        for e in events:
            side = "enemy" if e.probable_enemy_kill else "ally"
            print(f"[{p.name}] [{side}] {e.killer} -> {e.victim}")

        append_timing_log(
            {
                "mode": "static",
                "image": p.name,
                "events_inc": len(events),
                "ocr_compute": describe_ocr_compute_backend(ocr_engine),
                **timing,
            }
        )

        if debug_dir is not None and preview is not None:
            footer = describe_ocr_compute_backend(ocr_engine)
            tag = f"{timing.get('row_detect_mode', '?')} rows={timing.get('rows_detected', 0)}"
            out_paths = save_killfeed_debug_images(
                debug_dir,
                p.stem,
                annotated_bgr=preview,
                masks=masks,
                footer_line=footer,
                mode_tag=tag,
            )
            if out_paths:
                print(f"Debug PNG -> {out_paths[0].parent} ({len(out_paths)} file(s), {p.name})")

        if show and preview is not None:
            draw_status_footer(preview, describe_ocr_compute_backend(ocr_engine))
            try:
                cv2.imshow(f"static: {p.name}", preview)
                cv2.waitKey(0)
                cv2.destroyWindow(f"static: {p.name}")
            except cv2.error as exc:
                print(f"OpenCV GUI unavailable, skipping preview ({exc})")

    save_outputs(all_events)
    print(f"\nWrote {OVERLAY_TXT} and {EVENTS_JSON} ({len(all_events)} row(s) total).")


def run_live(
    ocr_engine: str,
    use_gui: bool,
    save_fullscreen: bool,
    row_bands_frac: list[tuple[float, float]] | None = None,
    debug_dir: Path | None = None,
) -> None:
    recent_pairs: list[tuple[str, str, float]] = []
    events_log: list[KillfeedEvent] = []
    last_screenshot_time = 0.0
    frame_idx = 0
    screenshot_dir = Path(SCREENSHOT_DIR)
    if save_fullscreen:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    gui = use_gui and opencv_highgui_available()
    if use_gui and not gui:
        print(
            "OpenCV has no GUI (common after easyocr installs opencv-python-headless). "
            "Running headless. Fix: pip uninstall opencv-python-headless && pip install opencv-python\n"
            "Or pass --no-gui explicitly."
        )

    with mss.mss() as sct:
        if gui:
            print("Killfeed parser running (full rows: killer -> victim). Press Q to stop.")
        else:
            print("Killfeed parser running headless (no preview). Stop with Ctrl+C.")
        print(f"Fullscreen screenshots: {'on' if save_fullscreen else 'off'} -> {SCREENSHOT_DIR}")
        if debug_dir is not None:
            print(f"Killfeed debug PNGs (on OCR events): {debug_dir.resolve()}")
        time.sleep(1)
        warm_easyocr_for_session(ocr_engine)

        while True:
            frame_idx += 1
            shot = np.ascontiguousarray(np.array(sct.grab(REGION_KILLFEED))[:, :, :3])
            display = shot.copy() if (gui or debug_dir is not None) else None
            now = time.time()

            events, _, timing, masks = process_frame(
                shot,
                now,
                recent_pairs=recent_pairs,
                draw=display,
                ocr_engine=ocr_engine,
                show_debug_windows=gui,
                row_bands_frac=row_bands_frac,
            )

            for e in events:
                events_log.append(e)
                side = "enemy" if e.probable_enemy_kill else "ally"
                print(f"[{side}] {e.killer} -> {e.victim}")

            if (
                save_fullscreen
                and events
                and now - last_screenshot_time >= SCREENSHOT_COOLDOWN
            ):
                saved = save_fullscreen_capture(sct, screenshot_dir, now)
                if saved:
                    print(f"Screenshot saved: {saved}")
                last_screenshot_time = now

            if debug_dir is not None and display is not None and events:
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{frame_idx:06d}"
                save_killfeed_debug_images(
                    debug_dir,
                    f"live_{stamp}",
                    annotated_bgr=display,
                    masks=masks,
                    footer_line=describe_ocr_compute_backend(ocr_engine),
                    mode_tag=f"{timing.get('row_detect_mode', '?')} ev={len(events)}",
                )

            save_outputs(events_log)

            if gui and display is not None:
                draw_status_footer(display, describe_ocr_compute_backend(ocr_engine))
                try:
                    cv2.imshow("Killfeed region", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    gui = False
                    print("OpenCV GUI failed mid-run; continuing headless.")

            time.sleep(REFRESH_DELAY)

            if frame_idx % LIVE_LOG_EVERY_FRAMES == 0:
                append_timing_log(
                    {
                        "mode": "live",
                        "frame_idx": frame_idx,
                        "events_logged_total": len(events_log),
                        "ocr_compute": describe_ocr_compute_backend(ocr_engine),
                        **timing,
                    }
                )

    if gui:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def collect_image_paths(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valorant killfeed: killer → victim via OCR.")
    parser.add_argument("--image", type=str, default=None, help="Single screenshot to parse (no live capture).")
    parser.add_argument("--folder", type=str, default=None, help="Folder of screenshots to parse.")
    parser.add_argument("--no-show", action="store_true", help="Static mode: do not open preview windows.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Live mode: no OpenCV windows (use with opencv-python-headless or SSH).",
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        choices=["tesseract", "easyocr", "both"],
        default=DEFAULT_OCR_ENGINE,
        help="OCR engine: tesseract, easyocr, or both (ensemble).",
    )
    parser.add_argument(
        "--fullscreen-screenshots",
        action=argparse.BooleanOptionalAction,
        default=SAVE_FULLSCREEN_ON_KILLFEED,
        help="Live: save full-monitor PNG when killfeed events appear (default from SAVE_FULLSCREEN_ON_KILLFEED).",
    )
    parser.add_argument(
        "--easyocr-canvas-size",
        type=int,
        default=None,
        metavar="N",
        help="EasyOCR detection canvas (default: script constant; lower = faster, may hurt accuracy).",
    )
    parser.add_argument(
        "--easyocr-no-stack-rows",
        action="store_true",
        help="EasyOCR: one readtext per row instead of one stacked pass (debug / compare speed).",
    )
    parser.add_argument(
        "--easyocr-detect-network",
        type=str,
        choices=("craft", "dbnet18"),
        default=None,
        help='EasyOCR detector: craft (default, works with pip PyTorch+CUDA) or dbnet18 (faster if deformable-conv builds; needs MSVC+CUDA_HOME on Windows).',
    )
    parser.add_argument(
        "--killfeed-rect",
        type=str,
        default=None,
        metavar="TOP,LEFT,WIDTH,HEIGHT",
        help="Override REGION_KILLFEED for this run (full-monitor coordinates), e.g. 80,1300,600,180",
    )
    parser.add_argument(
        "--row-bands-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Fixed horizontal row strips as fractions of ROI height (key row_bands_frac). "
        "Matches weapon script config, e.g. config/killfeed_row_bands.json — skips HSV row contours.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save debug PNGs: annotated killfeed ROI (gray=all detected rows, green/red=OCR boxes), "
        "plus green/red HSV masks. Static: one set per input image. Live: when at least one event is emitted.",
    )
    args = parser.parse_args()

    global REGION_KILLFEED, EASYOCR_CANVAS_SIZE, EASYOCR_STACK_ROWS, _easyocr_detect_network
    if args.killfeed_rect:
        parts = [int(x.strip()) for x in args.killfeed_rect.split(",")]
        if len(parts) != 4 or any(p < 0 for p in parts):
            raise SystemExit("--killfeed-rect requires four non-negative integers: TOP,LEFT,WIDTH,HEIGHT")
        REGION_KILLFEED = {
            "top": parts[0],
            "left": parts[1],
            "width": parts[2],
            "height": parts[3],
        }
        print(f"Killfeed region override: {REGION_KILLFEED}")
    if args.easyocr_canvas_size is not None:
        EASYOCR_CANVAS_SIZE = max(64, min(4096, int(args.easyocr_canvas_size)))
    if args.easyocr_no_stack_rows:
        EASYOCR_STACK_ROWS = False
    if args.easyocr_detect_network is not None:
        _easyocr_detect_network = args.easyocr_detect_network
        _easyocr_invalidate_reader()

    row_bands_frac: list[tuple[float, float]] | None = None
    if args.row_bands_json is not None:
        row_bands_frac = load_row_bands_json(args.row_bands_json.resolve())
        print(f"Row layout: {len(row_bands_frac)} fixed band(s) from {args.row_bands_json}")

    debug_dir = args.debug_dir.resolve() if args.debug_dir is not None else None

    print(f"OCR / compute: {describe_ocr_compute_backend(args.ocr_engine)}")
    if args.ocr_engine in ("easyocr", "both"):
        if easyocr is None:
            print("Tip: install EasyOCR for this engine: pip install easyocr")
        elif not _EASYOCR_USE_GPU:
            print(
                "Tip: EasyOCR is on CPU. For GPU speed install PyTorch **with CUDA** for your GPU, "
                "then restart Python: https://pytorch.org/get-started/locally/"
            )

    if args.image:
        run_static_images(
            [Path(args.image)],
            show=not args.no_show,
            ocr_engine=args.ocr_engine,
            row_bands_frac=row_bands_frac,
            debug_dir=debug_dir,
        )
        return
    if args.folder:
        folder = Path(args.folder)
        paths = collect_image_paths(folder)
        if not paths:
            raise SystemExit(f"No images found in {folder}")
        run_static_images(
            paths,
            show=not args.no_show,
            ocr_engine=args.ocr_engine,
            row_bands_frac=row_bands_frac,
            debug_dir=debug_dir,
        )
        return

    run_live(
        ocr_engine=args.ocr_engine,
        use_gui=not args.no_gui,
        save_fullscreen=args.fullscreen_screenshots,
        row_bands_frac=row_bands_frac,
        debug_dir=debug_dir,
    )


if __name__ == "__main__":
    main()
