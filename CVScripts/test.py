"""
raw_pipeline.py

Full raw image processing pipeline using only NumPy.
Simulates a real camera RAW pipeline:

  1.  Generate synthetic RAW Bayer mosaic
  2.  Black-level subtraction
  3.  Lens shading / vignette correction
  4.  Demosaicing  (bilinear interpolation per Bayer channel)
  5.  White balance (grey-world estimation)
  6.  Color matrix  (camera RGB → linear sRGB, XYZ transform)
  7.  Highlight clipping
  8.  Gamma correction  (sRGB piecewise transfer function)
  9.  Brightness / exposure correction
  10. Contrast (sigmoid tone curve)
  11. Sharpening (unsharp mask via convolution)
  12. Convert to grayscale (luminosity weights)
  13. Save both images as PNG (pure-numpy, no PIL)
"""

import time
import struct
import zlib
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
WIDTH        = 2048
HEIGHT       = 2048
BIT_DEPTH    = 12          # simulated sensor bit depth
BLACK_LEVEL  = 64          # typical camera black point
WHITE_LEVEL  = 4095        # 2^12 - 1
BAYER_PATTERN = "RGGB"     # most common pattern
OUTPUT_RGB   = "output_rgb.png"
OUTPUT_GRAY  = "output_grayscale.png"
# ─────────────────────────────────────────────────────────────────────────────


def timeit(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f" raw {elapsed}")
    print(f"  {label:<50} {elapsed * 1000:>8.2f} ms")
    return result


# ── Pure-numpy PNG writer ─────────────────────────────────────────────────────
def write_png(path, img_uint8):
    """
    Write a uint8 numpy array to a PNG file with zero external dependencies.
    Supports H×W (grayscale) and H×W×3 (RGB) arrays.
    """
    if img_uint8.ndim == 2:
        h, w = img_uint8.shape
        color_type = 0   # grayscale
        channels   = 1
    else:
        h, w, channels = img_uint8.shape
        color_type = 2   # RGB

    def chunk(name, data):
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)

    # Build raw image data with filter byte 0 (None) per row
    raw_rows = []
    for row in range(h):
        if channels == 1:
            raw_rows.append(b"\x00" + img_uint8[row].tobytes())
        else:
            raw_rows.append(b"\x00" + img_uint8[row].tobytes())
    raw = b"".join(raw_rows)
    compressed = zlib.compress(raw, level=6)
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(sig + ihdr + idat + iend)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_generate_raw_bayer():
    """
    Simulate a 12-bit RAW Bayer mosaic.
    Add realistic noise: shot noise (Poisson) + read noise (Gaussian),
    plus a slight radial signal falloff to give shading something to correct.
    """
    rng = np.random.default_rng(42)

    # Base scene: smooth gradients + some structure
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    xx, yy = np.meshgrid(x, y)
    scene = (0.4 + 0.4 * np.sin(xx * 6) * np.cos(yy * 4) +
             0.2 * (xx + yy) / 2)                         # [0,1]

    # Radial vignetting (darkens corners)
    cx, cy = 0.5, 0.5
    r2 = (xx - cx)**2 + (yy - cy)**2
    vignette = np.exp(-1.8 * r2)
    scene = scene * vignette

    signal = (scene * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL).astype(np.float32)

    # Per-channel color tints across the Bayer grid
    raw = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    tints = {"R": 1.0, "G": 0.85, "B": 0.60}   # simulate color temperature

    for row_off, col_off, ch in [(0, 0, "R"), (0, 1, "G"),
                                  (1, 0, "G"), (1, 1, "B")]:
        sl_r = slice(row_off, None, 2)
        sl_c = slice(col_off, None, 2)
        sub   = signal[sl_r, sl_c] * tints[ch]
        # Shot noise (Poisson approximated as Gaussian for large counts)
        shot  = rng.normal(0, np.sqrt(np.maximum(sub, 1)), sub.shape)
        read  = rng.normal(0, 2.5, sub.shape)              # read noise ~2.5 DN
        raw[sl_r, sl_c] = np.clip(sub + shot + read, 0, WHITE_LEVEL)

    return raw.astype(np.uint16)


def step2_black_level_subtract(raw):
    """Subtract sensor black level and renormalise to [0, 1]."""
    out = raw.astype(np.float32)
    out = np.clip(out - BLACK_LEVEL, 0, None)
    out = out / (WHITE_LEVEL - BLACK_LEVEL)
    return out                                   # [0, 1] float


def step3_lens_shading_correction(raw_norm):
    """
    Correct vignetting by fitting and inverting a radial gain surface.
    We estimate the gain map from the image itself (flat-field style).
    """
    h, w = raw_norm.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    r2 = xx**2 + yy**2

    # Fit a smooth radial model to the mean local brightness
    # Simple closed-form: gain ∝ 1 / (1 - k*r²), k estimated from corners vs centre
    centre_val = raw_norm[h//2-50:h//2+50, w//2-50:w//2+50].mean()
    corner_val = np.mean([
        raw_norm[:50,  :50 ].mean(), raw_norm[:50,  -50:].mean(),
        raw_norm[-50:, :50 ].mean(), raw_norm[-50:, -50:].mean()
    ])
    r2_corner  = 2.0                              # corners: r2 = 1² + 1² = 2
    if corner_val > 0:
        ratio = centre_val / corner_val
        k = (ratio - 1) / r2_corner              # solve ratio = 1 + k*r2_corner
    else:
        k = 0.3

    gain_map = 1.0 + k * r2
    return np.clip(raw_norm * gain_map, 0, 1.0)


def step4_demosaic(raw_norm):
    """
    Bilinear demosaicing for RGGB Bayer pattern.
    Each colour channel is extracted at its native positions, then
    upsampled to full resolution by bilinear interpolation using
    a convolution kernel applied via stride tricks.
    """
    h, w = raw_norm.shape

    # Convolution helper (2-D, same padding)
    def convolve2d(img, kernel):
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
        out = np.zeros_like(img)
        for i in range(kh):
            for j in range(kw):
                out += kernel[i, j] * padded[i:i+h, j:j+w]
        return out

    R = np.zeros((h, w), dtype=np.float32)
    G = np.zeros((h, w), dtype=np.float32)
    B = np.zeros((h, w), dtype=np.float32)

    # Place known samples
    R[0::2, 0::2] = raw_norm[0::2, 0::2]   # R at R positions
    G[0::2, 1::2] = raw_norm[0::2, 1::2]   # G at Gr positions
    G[1::2, 0::2] = raw_norm[1::2, 0::2]   # G at Gb positions
    B[1::2, 1::2] = raw_norm[1::2, 1::2]   # B at B positions

    # Bilinear interpolation kernels
    k_cross = np.array([[0, 1, 0],
                         [1, 4, 1],
                         [0, 1, 0]], dtype=np.float32) / 4.0

    k_checker = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 4.0

    k_full = np.array([[1, 1, 1],
                        [1, 4, 1],
                        [1, 1, 1]], dtype=np.float32) / 4.0

    # Interpolate R: known at every other row+col → checker pattern fill
    R_interp = convolve2d(R, k_checker)
    # Known positions already correct; blend with interpolated
    mask_R = np.zeros((h, w), dtype=np.float32)
    mask_R[0::2, 0::2] = 1.0
    R_out = np.where(mask_R > 0, R, R_interp)

    # Interpolate B
    B_interp = convolve2d(B, k_checker)
    mask_B = np.zeros((h, w), dtype=np.float32)
    mask_B[1::2, 1::2] = 1.0
    B_out = np.where(mask_B > 0, B, B_interp)

    # Interpolate G: known at half the pixels in a cross pattern
    G_interp = convolve2d(G, k_cross)
    mask_G = np.zeros((h, w), dtype=np.float32)
    mask_G[0::2, 1::2] = 1.0
    mask_G[1::2, 0::2] = 1.0
    G_out = np.where(mask_G > 0, G, G_interp)

    rgb = np.stack([R_out, G_out, B_out], axis=-1)
    return np.clip(rgb, 0, 1)


def step5_white_balance(rgb):
    """
    Grey-world white balance: assume the scene average should be neutral grey.
    Scale each channel so its mean equals the overall mean.
    """
    mean_r, mean_g, mean_b = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
    overall = (mean_r + mean_g + mean_b) / 3.0
    gains = np.array([overall / (mean_r + 1e-9),
                      overall / (mean_g + 1e-9),
                      overall / (mean_b + 1e-9)], dtype=np.float32)
    return np.clip(rgb * gains[np.newaxis, np.newaxis, :], 0, 1)


def step6_color_matrix(rgb):
    """
    Apply a 3×3 color correction matrix to go from camera-native linear RGB
    to linear sRGB (D65 illuminant).  Matrix derived from a typical CMOS sensor.
    This corrects color crosstalk between channels.
    """
    # Typical camera → sRGB CCM (row = output channel, col = input channel)
    CCM = np.array([
        [ 1.9749, -0.6684, -0.3065],
        [-0.2024,  1.5410, -0.3386],
        [ 0.0050, -0.5439,  1.5389],
    ], dtype=np.float32)

    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)           # (N, 3)
    corrected = flat @ CCM.T           # (N, 3)
    return np.clip(corrected.reshape(h, w, 3), 0, 1)


def step7_highlight_clip(rgb):
    """
    Soft-clip highlights to prevent blown-out whites from hard clipping artefacts.
    Uses a smooth shoulder above 0.95.
    """
    threshold = 0.95

    def soft_clip(x):
        # Below threshold: linear. Above: smooth asymptote to 1.0
        above = x > threshold
        y = x.copy()
        t = (x[above] - threshold) / (1.0 - threshold)   # [0, inf)
        y[above] = threshold + (1.0 - threshold) * (1 - np.exp(-t * 3))
        return y

    return soft_clip(rgb)


def step8_gamma_correction(linear_rgb):
    """
    Apply the official sRGB piecewise gamma transfer function.
    Converts linear light values to display-encoded (perceptual) values.
      sRGB(x) = 12.92 * x               if x <= 0.0031308
                1.055 * x^(1/2.4) - 0.055  otherwise
    """
    out = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(np.maximum(linear_rgb, 0), 1.0 / 2.4) - 0.055
    )
    return np.clip(out, 0, 1)


def step9_brightness_exposure(rgb, ev_stops=0.3):
    """
    Exposure compensation in linear-light space (before gamma).
    One stop = multiply by 2. Here we go back to linear, adjust, re-gamma.
    """
    # Invert sRGB gamma
    def srgb_to_linear(s):
        return np.where(s <= 0.04045, s / 12.92,
                        np.power((s + 0.055) / 1.055, 2.4))

    linear = srgb_to_linear(rgb)
    linear = np.clip(linear * (2.0 ** ev_stops), 0, 1)
    # Re-apply gamma
    return step8_gamma_correction(linear)


def step10_sigmoid_tone_curve(rgb, contrast=1.6, midpoint=0.5):
    """
    S-curve (sigmoid) tone mapping for contrast enhancement.
    Lifts shadows slightly, compresses highlights — the classic film look.
    f(x) = 1 / (1 + exp(-contrast * (x - midpoint)))
    Normalised so f(0)=0 and f(1)=1.
    """
    f = lambda x: 1.0 / (1.0 + np.exp(-contrast * (x - midpoint)))
    lo = f(0.0)
    hi = f(1.0)
    return np.clip((f(rgb) - lo) / (hi - lo), 0, 1)


def step11_sharpen(rgb, strength=0.4):
    """
    Unsharp masking: sharpen = original + strength * (original - blurred).
    Gaussian blur computed with a separable kernel (fast pure-numpy).
    """
    def gaussian_kernel_1d(sigma=1.2, size=5):
        x = np.arange(size) - size // 2
        k = np.exp(-x**2 / (2 * sigma**2))
        return (k / k.sum()).astype(np.float32)

    k1d = gaussian_kernel_1d()
    blurred = np.zeros_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        # Horizontal pass
        padded = np.pad(ch, ((0, 0), (2, 2)), mode="reflect")
        tmp = sum(k1d[i] * padded[:, i:i+WIDTH] for i in range(5))
        # Vertical pass
        padded = np.pad(tmp, ((2, 2), (0, 0)), mode="reflect")
        blurred[..., c] = sum(k1d[i] * padded[i:i+HEIGHT, :] for i in range(5))

    sharpened = rgb + strength * (rgb - blurred)
    return np.clip(sharpened, 0, 1)


def step12_to_grayscale(rgb):
    """
    Luminosity-weighted grayscale using the ITU-R BT.709 coefficients,
    which match human perceptual brightness sensitivity.
    Y = 0.2126 R + 0.7152 G + 0.0722 B
    """
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return (rgb @ weights)                        # (H, W) float [0,1]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print(f"  RAW image pipeline  |  {WIDTH}×{HEIGHT} px  |  {BIT_DEPTH}-bit Bayer input")
    print(f"{'='*65}\n")
    print('NOW')
    now_ms = int(time.time_ns() / 1000000)
    print(now_ms)

    raw   = timeit("1.  Generate synthetic RAW Bayer mosaic",     step1_generate_raw_bayer)
    norm  = timeit("2.  Black-level subtraction + normalise",     lambda: step2_black_level_subtract(raw))
    shad  = timeit("3.  Lens shading / vignette correction",      lambda: step3_lens_shading_correction(norm))
    rgb   = timeit("4.  Demosaicing (bilinear, RGGB)",            lambda: step4_demosaic(shad))
    wb    = timeit("5.  White balance (grey-world)",              lambda: step5_white_balance(rgb))
    ccm   = timeit("6.  Color matrix (camera → linear sRGB)",     lambda: step6_color_matrix(wb))
    hc    = timeit("7.  Highlight soft-clipping",                 lambda: step7_highlight_clip(ccm))
    gam   = timeit("8.  Gamma correction (sRGB piecewise TF)",    lambda: step8_gamma_correction(hc))
    bright= timeit("9.  Brightness / exposure (+0.3 EV)",         lambda: step9_brightness_exposure(gam))
    tone  = timeit("10. Sigmoid tone curve (contrast)",           lambda: step10_sigmoid_tone_curve(bright))
    sharp = timeit("11. Sharpening (unsharp mask, Gaussian)",     lambda: step11_sharpen(tone))
    gray  = timeit("12. Grayscale (BT.709 luminosity)",          lambda: step12_to_grayscale(sharp))

    rgb_u8  = timeit("13. Quantise RGB  → uint8",  lambda: (sharp * 255).clip(0, 255).astype(np.uint8))
    gray_u8 = timeit("14. Quantise Gray → uint8",  lambda: (gray  * 255).clip(0, 255).astype(np.uint8))

    timeit(f"15. Write RGB PNG  → {OUTPUT_RGB}",  lambda: write_png(OUTPUT_RGB,  rgb_u8))
    timeit(f"16. Write Gray PNG → {OUTPUT_GRAY}", lambda: write_png(OUTPUT_GRAY, gray_u8))

    print('NOW')
    now_ms = int(time.time_ns() / 1000000)
    print(now_ms)

    print(f"\n{'='*65}")
    print(f"  Done.  Outputs: {OUTPUT_RGB}, {OUTPUT_GRAY}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()