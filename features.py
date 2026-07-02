"""
features.py
Classical computer-vision feature extraction for "real photo vs. photo-of-a-screen"
classification. No deep learning, no GPU. Every feature below targets a specific
physical artifact of the screen -> camera recapture pipeline:

  - fft_peak_ratio / fft_radial_std : moire / aliasing from photographing a pixel grid
  - colorfulness                    : screens have compressed gamut -> lower colorfulness
  - sat_mean / sat_std              : saturation compression
  - blue_bias                       : LED backlight color cast
  - clip_frac                       : blown-out glare highlights typical of glossy screens
  - lap_var                         : sharpness ceiling (double blur: screen px + camera optics)
  - line_rect_score                 : strong parallel/perpendicular edges from a bezel
  - dwt_*                           : wavelet-domain natural-image-statistics (see below)

Wavelet-domain natural image statistics (the "uncommon" feature block):
Natural photos obey fairly consistent statistical regularities -- e.g. their
wavelet subband coefficients are heavy-tailed but *smoothly* so (this is the
basis of blind image-quality assessment and steganalysis). A screen recapture
is a re-sampling of an already-quantized, already-periodic pixel grid, which
disturbs those regularities: it injects small sharp modes into the subband
coefficient distributions at specific scales. We capture this with the
variance, skewness, and excess kurtosis of the level-1 horizontal/vertical/
diagonal DWT subbands, plus a cross-scale energy ratio between level-1 and
level-2 diagonal subbands (real photos redistribute energy across scales
smoothly; a recapture concentrates it at the scale matching the display's
pixel pitch). This is a much less common signal for this kind of task than
FFT/moire or ELA and is more robust to compression/resizing than raw FFT
peak-hunting.

Returns a fixed-length numpy feature vector per image.
"""

import cv2
import numpy as np
import pywt

FEATURE_NAMES = [
    "fft_peak_ratio",
    "fft_radial_std",
    "colorfulness",
    "sat_mean",
    "sat_std",
    "blue_bias",
    "clip_frac",
    "lap_var",
    "line_rect_score",
    "dwt_h_var",
    "dwt_v_var",
    "dwt_d_var",
    "dwt_h_skew",
    "dwt_v_skew",
    "dwt_d_skew",
    "dwt_h_kurt",
    "dwt_v_kurt",
    "dwt_d_kurt",
    "dwt_scale_energy_ratio",
    "noise_variance",
    "brightness_center_ratio" ,
    "noise_variance",
    "center_std_ratio",      
    "zoom_sharpness"  
]


def _resize_max_side(img, max_side=512):
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _fft_features(gray):
    """Detect moire: real photos have smooth ~1/f frequency falloff; screen recaptures
    show sharp off-axis energy peaks from the display's pixel grid beating against the
    camera sensor grid."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    # mask out the DC/low-frequency disk (natural image energy concentrates here)
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()

    low_mask = r_norm < 0.08          # DC + very low freq, ignore
    mid_mask = (r_norm >= 0.08) & (r_norm < 0.45)   # where moire peaks typically sit
    high_mask = r_norm >= 0.45

    mid_energy = mag[mid_mask]
    total_energy = mag[~low_mask].sum() + 1e-6

    # peak-iness: a real photo's mid-band energy is diffuse; a moire pattern
    # concentrates energy into a few sharp bins, so the top-1% bins carry a
    # disproportionate share of the mid-band energy.
    if mid_energy.size > 0:
        sorted_mid = np.sort(mid_energy)[::-1]
        top_k = max(1, int(0.01 * sorted_mid.size))
        fft_peak_ratio = sorted_mid[:top_k].sum() / (mid_energy.sum() + 1e-6)
    else:
        fft_peak_ratio = 0.0

    # radial energy profile std: natural images fall off smoothly (low std between
    # neighboring radial bins); moire introduces ring-like irregular jumps.
    nbins = 40
    bin_idx = np.clip((r_norm * nbins).astype(int), 0, nbins - 1)
    radial_profile = np.zeros(nbins)
    for i in range(nbins):
        vals = mag[bin_idx == i]
        radial_profile[i] = vals.mean() if vals.size else 0.0
    radial_diff = np.diff(radial_profile)
    fft_radial_std = float(np.std(radial_diff))

    return float(fft_peak_ratio), fft_radial_std


def _colorfulness(img_bgr):
    """Hasler-Susstrunk colorfulness metric. Screens under-reproduce gamut -> lower score."""
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    return float(std_root + 0.3 * mean_root)


def _saturation_stats(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    return float(s.mean()), float(s.std())


def _blue_bias(img_bgr):
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    return float(b.mean() - r.mean())


def _clip_frac(img_bgr):
    """Fraction of near-blown-out pixels -- glossy screen glare tends to clip hard."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray > 250))


def _sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _line_rect_score(gray, work_size=200):
    """Looks for a small set of strong, near axis-aligned parallel/perpendicular lines
    -- evidence of a phone/monitor bezel edge in frame.

    IMPORTANT: always resamples to a fixed small working size (not "resize only if
    larger") before Canny/Hough. cv2.HoughLinesP has been observed to segfault on
    edge maps with strong exact periodicity -- precisely the kind of regular grid
    pattern a screen recapture can produce. Resampling to a fixed size breaks any
    pixel-exact periodicity from the source image and keeps this feature both safe
    and fast; full resolution isn't needed to detect a bezel edge anyway.
    """
    small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 60, 150)

    # Second defensive layer: skip on pathologically dense edge maps (uninformative
    # for bezel detection regardless, and the highest-risk case for Hough).
    edge_density = np.count_nonzero(edges) / edges.size
    if edge_density > 0.25:
        return 0.0

    try:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                 minLineLength=work_size // 4, maxLineGap=10)
    except cv2.error:
        return 0.0
    if lines is None:
        return 0.0
    lines = np.asarray(lines).reshape(-1, 4)

    angles = []
    for l in lines:
        x1, y1, x2, y2 = l
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        angles.append(ang)
    angles = np.array(angles)

    # bucket into 0/90-degree-ish bins (axis aligned bezel edges)
    near_horiz = np.sum((angles < 5) | (angles > 175))
    near_vert = np.sum((angles > 85) & (angles < 95))
    score = (near_horiz + near_vert) / float(len(angles))
    return float(score)


def _moments(x):
    """Mean-centered variance, skewness, and excess (Fisher) kurtosis of a flat array."""
    x = x.astype(np.float64).ravel()
    mean = x.mean()
    diffs = x - mean
    var = float(np.mean(diffs ** 2))
    std = np.sqrt(var) + 1e-8
    skew = float(np.mean(diffs ** 3) / (std ** 3))
    kurt = float(np.mean(diffs ** 4) / (std ** 4) - 3.0)
    return var, skew, kurt


def _dwt_features(gray, work_size=256, wavelet="db4"):
    """Wavelet-domain natural-image-statistics. See module docstring for the
    physical motivation. Resamples to a fixed size first for two reasons: speed,
    and so feature scale doesn't depend on the input photo's native resolution."""
    small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)

    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(small, wavelet, level=2)

    h_var, h_skew, h_kurt = _moments(cH1)
    v_var, v_skew, v_kurt = _moments(cV1)
    d_var, d_skew, d_kurt = _moments(cD1)

    # cross-scale energy ratio: how much diagonal-detail energy sits at the finest
    # scale vs. one octave up. A real photo's energy falls off smoothly across
    # scales; a recapture's moire/pixel-grid energy concentrates at the scale
    # matching the display's pixel pitch, skewing this ratio.
    energy1 = float(np.mean(cD1 ** 2))
    energy2 = float(np.mean(cD2 ** 2)) + 1e-8
    scale_energy_ratio = energy1 / energy2

    return (h_var, v_var, d_var, h_skew, v_skew, d_skew, h_kurt, v_kurt, d_kurt,
            scale_energy_ratio)


def extract_features(img_bgr):
    img_bgr = _resize_max_side(img_bgr, 512)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fft_peak_ratio, fft_radial_std = _fft_features(gray)
    colorfulness = _colorfulness(img_bgr)
    sat_mean, sat_std = _saturation_stats(img_bgr)
    blue_bias = _blue_bias(img_bgr)
    clip_frac = _clip_frac(img_bgr)
    lap_var = _sharpness(gray)
    line_rect_score = _line_rect_score(gray)
    (dwt_h_var, dwt_v_var, dwt_d_var,
     dwt_h_skew, dwt_v_skew, dwt_d_skew,
     dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
     dwt_scale_energy_ratio) = _dwt_features(gray)

    return np.array([
        fft_peak_ratio,
        fft_radial_std,
        colorfulness,
        sat_mean,
        sat_std,
        blue_bias,
        clip_frac,
        lap_var,
        line_rect_score,
        dwt_h_var,
        dwt_v_var,
        dwt_d_var,
        dwt_h_skew,
        dwt_v_skew,
        dwt_d_skew,
        dwt_h_kurt,
        dwt_v_kurt,
        dwt_d_kurt,
        dwt_scale_energy_ratio,
    ], dtype=np.float32)
