from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

#INPUT_FOLDER = "Video_Camera_Processing/throws/throw0"
INPUT_FOLDER = "Video_Camera_Processing/Intrinsics"
OUTPUT_FOLDER = None
WIDTH = 3840
HEIGHT = 2160
CHANNELS = 0
DTYPE = "uint8"
RGB_ORDER = False
INFER_SIZE_IF_MISMATCH = True
BAYER_PATTERN = "RG"
USE_JSON_SIDECAR = True


def infer_wh_pixels(px, cfg_w, cfg_h):
    if cfg_w * cfg_h == px:
        return cfg_w, cfg_h
    ref_ar = (cfg_w / cfg_h) if cfg_h else (16.0 / 9.0)
    best = None
    best_key = None
    a = 64
    while a * a <= px:
        if px % a == 0:
            b = px // a
            for w, h in ((a, b), (b, a)):
                if min(w, h) < 64 or max(w, h) > 8192:
                    continue
                ar = w / h
                key = (abs(w - cfg_w) + abs(h - cfg_h), abs(ar - ref_ar))
                if best_key is None or key < best_key:
                    best_key = key
                    best = (w, h)
        a += 1
    if best is None:
        raise ValueError(f"Could not infer width/height from {px} pixels; check DTYPE or fixed WIDTH/HEIGHT.")
    return best


def infer_channels_and_wh(num_bytes, dtype, cfg_w, cfg_h, fixed_channels):
    itemsize = dtype.itemsize
    if fixed_channels in (1, 3, 4):
        ch = fixed_channels
        bpp = ch * itemsize
        if num_bytes % bpp != 0:
            raise ValueError(f"File size {num_bytes} not divisible by {bpp} ({ch} ch, {dtype})")
        px = num_bytes // bpp
        if cfg_w * cfg_h == px:
            return ch, cfg_w, cfg_h
        w, h = infer_wh_pixels(px, cfg_w, cfg_h)
        return ch, w, h

    best = None
    best_key = None
    for ch in (1, 3, 4):
        bpp = ch * itemsize
        if num_bytes % bpp != 0:
            continue
        px = num_bytes // bpp
        if cfg_w * cfg_h == px:
            key = (-10_000, ch)
            cand = (ch, cfg_w, cfg_h)
        else:
            w, h = infer_wh_pixels(px, cfg_w, cfg_h)
            dist = abs(w - cfg_w) + abs(h - cfg_h)
            if w == cfg_w:
                dist -= 2000
            if h == cfg_h:
                dist -= 2000
            key = (dist, ch)
            cand = (ch, w, h)
        if best_key is None or key < best_key:
            best_key = key
            best = cand
    if best is None:
        raise ValueError(f"Cannot interpret {num_bytes} bytes as 1/3/4-channel {dtype}; try DTYPE or CHANNELS=1/3/4.")
    return best


def array_from_raw_bytes(data, width, height, channels, dtype, rgb_order):
    expected = width * height * channels * dtype.itemsize
    if len(data) != expected:
        raise ValueError(f"File size {len(data)} bytes != expected {expected} ({width}x{height}x{channels}, {dtype})")
    if channels == 1:
        arr = np.frombuffer(data, dtype=dtype).reshape((height, width))
    else:
        arr = np.frombuffer(data, dtype=dtype).reshape((height, width, channels))
    if channels == 3 and rgb_order:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def sidecar_bayer_hint(meta):
    pf = meta.get("pixel_format") or meta.get("pixelFormat")
    if pf is None:
        return None
    u = str(pf).upper().replace(" ", "").replace("-", "_")
    if "MONO" in u and "BAYER" not in u:
        return ""
    uflat = u.replace("_", "")
    if "BAYERRG" in uflat or "RGGB" in uflat:
        return "RG"
    if "BAYERGR" in uflat or "GRBG" in uflat:
        return "GR"
    if "BAYERGB" in uflat or "GBRG" in uflat:
        return "GB"
    if "BAYERBG" in uflat or "BGGR" in uflat:
        return "BG"
    return None


def layout_from_sidecar_json(raw_path, num_bytes, dtype):
    if not USE_JSON_SIDECAR:
        return None
    base, _ = os.path.splitext(raw_path)
    jpath = base + ".json"
    if not os.path.isfile(jpath):
        return None
    with open(jpath, encoding="utf-8") as f:
        meta = json.load(f)
    w = int(meta["width"])
    h = int(meta["height"])
    if w < 1 or h < 1:
        raise ValueError(f"Invalid width/height in {jpath}")

    pb = meta.get("payload_bytes")
    if pb is not None and int(pb) != num_bytes:
        raise ValueError(f"{jpath}: payload_bytes {pb} != raw file size {num_bytes}")

    itemsize = dtype.itemsize
    denom = w * h * itemsize
    if num_bytes % denom != 0:
        raise ValueError(f"{jpath}: file size {num_bytes} incompatible with {w}x{h} and {dtype}")
    ch = num_bytes // denom
    if ch not in (1, 3, 4):
        raise ValueError(f"{jpath}: implies {ch} channels for {dtype}; expected 1, 3, or 4")
    return ch, w, h, sidecar_bayer_hint(meta)


def apply_bayer_if_configured(arr, pattern):
    if not pattern or arr.ndim != 2:
        return arr
    if arr.dtype != np.uint8:
        print(f"Demosaic skipped: expected uint8 Bayer; got {arr.dtype}", file=sys.stderr)
        return arr
    name = f"COLOR_Bayer{pattern}2BGR"
    code = getattr(cv2, name, None)
    if code is None:
        print(f"Unknown BAYER_PATTERN {pattern!r} (expected cv2.{name})", file=sys.stderr)
        return arr
    return cv2.cvtColor(arr, code)


def effective_bayer_pattern(sidecar_hint, global_pattern):
    if sidecar_hint == "":
        return None
    if sidecar_hint is not None:
        return sidecar_hint
    return global_pattern


def raw_path_to_array(path, cfg_w, cfg_h, channels, dtype, rgb_order, infer_if_mismatch, bayer_pattern):
    with open(path, "rb") as f:
        data = f.read()

    sidecar = layout_from_sidecar_json(path, len(data), dtype)
    if sidecar is not None:
        sch, sw, sh, bayer_hint = sidecar
        if channels in (1, 3, 4) and channels != sch:
            raise ValueError(
                f"CHANNELS={channels} but {path} sidecar implies {sch} channel(s)"
            )
        ch, w, h = sch, sw, sh
        arr = array_from_raw_bytes(data, w, h, ch, dtype, rgb_order)
        pat = effective_bayer_pattern(bayer_hint, bayer_pattern)
        if ch == 1:
            arr = apply_bayer_if_configured(arr, pat)
        return arr

    if channels == 0:
        if not infer_if_mismatch:
            raise ValueError("CHANNELS=0 requires INFER_SIZE_IF_MISMATCH=True")
        ch, w, h = infer_channels_and_wh(len(data), dtype, cfg_w, cfg_h, 0)
    else:
        bpp = channels * dtype.itemsize
        need = cfg_w * cfg_h * bpp
        if len(data) == need:
            ch, w, h = channels, cfg_w, cfg_h
        elif infer_if_mismatch:
            ch = channels
            if len(data) % bpp != 0:
                raise ValueError(f"File size {len(data)} not divisible by {bpp}")
            px = len(data) // bpp
            if cfg_w * cfg_h == px:
                w, h = cfg_w, cfg_h
            else:
                w, h = infer_wh_pixels(px, cfg_w, cfg_h)
        else:
            raise ValueError(f"File size {len(data)} bytes != expected {need} " f"({cfg_w}x{cfg_h}x{channels}, {dtype})")

    arr = array_from_raw_bytes(data, w, h, ch, dtype, rgb_order)
    if ch == 1:
        arr = apply_bayer_if_configured(arr, bayer_pattern)
    return arr


def iter_raw_paths(root):
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".raw"):
                yield os.path.join(dirpath, name)


def output_path_for_raw(raw_path, input_root):
    if OUTPUT_FOLDER is None:
        base, _ = os.path.splitext(raw_path)
        return base + ".png"
    out_root = os.path.abspath(OUTPUT_FOLDER)
    rel = os.path.relpath(raw_path, input_root)
    rel_png = os.path.splitext(rel)[0] + ".png"
    return os.path.join(out_root, rel_png)


def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Not a directory: {INPUT_FOLDER}", file=sys.stderr)
        return 1

    dtype = np.dtype(DTYPE)
    input_root = os.path.abspath(INPUT_FOLDER)
    raw_paths = list(iter_raw_paths(input_root))
    if not raw_paths:
        print(f"No .raw files under {input_root}", file=sys.stderr)
        return 1

    errors = 0
    for raw_path in raw_paths:
        out_path = output_path_for_raw(raw_path, input_root)
        try:
            img = raw_path_to_array(
                raw_path,
                WIDTH,
                HEIGHT,
                CHANNELS,
                dtype,
                rgb_order=RGB_ORDER,
                infer_if_mismatch=INFER_SIZE_IF_MISMATCH,
                bayer_pattern=BAYER_PATTERN,
            )
        except (OSError, ValueError) as e:
            print(f"{raw_path}: {e}", file=sys.stderr)
            errors += 1
            continue

        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if not cv2.imwrite(out_path, img):
            print(f"Failed to write {out_path}", file=sys.stderr)
            errors += 1
            continue

        print(f"Wrote {out_path}")

    if errors:
        print(f"Done with {errors} error(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
