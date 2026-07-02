#!/usr/bin/env python3
"""
Live Demo Dashboard for Spot the Fake Photo.
Run with: python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import base64
import cv2
import numpy as np
import joblib
import pywt
from flask import Flask, render_template, request, jsonify
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# ---------- Load the trained model ----------
MODEL_FILE = "model.pkl"
if not os.path.isfile(MODEL_FILE):
    print(f"Error: {MODEL_FILE} not found. Run 'python predict.py train' first.")
    exit(1)

scaler, selector, clf, raw_threshold = joblib.load(MODEL_FILE)

# ---------- Feature extraction (EXACTLY matches predict.py) ----------
FEATURE_NAMES = [
    "fft_peak_ratio", "fft_radial_std", "colorfulness", "sat_mean", "sat_std",
    "blue_bias", "clip_frac", "lap_var", "line_rect_score",
    "dwt_h_var", "dwt_v_var", "dwt_d_var",
    "dwt_h_skew", "dwt_v_skew", "dwt_d_skew",
    "dwt_h_kurt", "dwt_v_kurt", "dwt_d_kurt",
    "dwt_energy_compaction",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "lbp_variance", "clahe_entropy_delta", "color_fft_ratio", "dct_ac_ratio",
    "local_var_hetero", "edge_orient_entropy",
    "noise_variance",
    "brightness_center_ratio"
]

def _resize_max_side(img, max_side=512):
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def _brightness_center_ratio(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    half_h, half_w = int(h * 0.35), int(w * 0.35)
    center = gray[cy-half_h:cy+half_h, cx-half_w:cx+half_w]
    mean_all = np.mean(gray)
    mean_center = np.mean(center)
    return float(mean_center / (mean_all + 1e-8))

def _fft_features(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    mid_mask = (r_norm >= 0.08) & (r_norm < 0.45)
    mid_energy = mag[mid_mask]
    if mid_energy.size > 0:
        sorted_mid = np.sort(mid_energy)[::-1]
        top_k = max(1, int(0.01 * sorted_mid.size))
        fft_peak_ratio = float(sorted_mid[:top_k].sum() / (mid_energy.sum() + 1e-6))
    else:
        fft_peak_ratio = 0.0
    nbins = 40
    bin_idx = np.clip((r_norm * nbins).astype(int), 0, nbins - 1)
    radial_profile = np.zeros(nbins)
    for i in range(nbins):
        vals = mag[bin_idx == i]
        radial_profile[i] = vals.mean() if vals.size else 0.0
    fft_radial_std = float(np.std(np.diff(radial_profile)))
    return fft_peak_ratio, fft_radial_std

def _colorfulness(img_bgr):
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg, yb = r - g, 0.5 * (r + g) - b
    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return float(std_root + 0.3 * mean_root)

def _saturation_stats(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    return float(s.mean()), float(s.std())

def _blue_bias(img_bgr):
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    return float(b.mean() - r.mean())

def _clip_frac(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray > 250))

def _sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _line_rect_score(gray, work_size=200):
    small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 60, 150)
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
        angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
    angles = np.array(angles)
    near_horiz = np.sum((angles < 5) | (angles > 175))
    near_vert = np.sum((angles > 85) & (angles < 95))
    return float((near_horiz + near_vert) / len(angles))

def _moments(x):
    x = x.astype(np.float64).ravel()
    mean = x.mean()
    std = x.std() + 1e-8
    diffs = x - mean
    var = float(np.mean(diffs ** 2))
    skew = float(np.mean(diffs ** 3) / (std ** 3))
    kurt = float(np.mean(diffs ** 4) / (std ** 4) - 3.0)
    return var, skew, kurt

def _dwt_features(gray, work_size=256, wavelet="db4"):
    small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(small, wavelet, level=2)
    h_var, h_skew, h_kurt = _moments(cH1)
    v_var, v_skew, v_kurt = _moments(cV1)
    d_var, d_skew, d_kurt = _moments(cD1)
    approx_energy = float(np.sum(cA2 ** 2))
    detail_energy = float(sum(np.sum(b ** 2) for b in (cH1, cV1, cD1, cH2, cV2, cD2)))
    energy_compaction = detail_energy / (approx_energy + detail_energy + 1e-8)
    return (h_var, v_var, d_var, h_skew, v_skew, d_skew,
            h_kurt, v_kurt, d_kurt, energy_compaction)

def _glcm_features(gray, work_size=128):
    small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
    glcm = graycomatrix(small, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        symmetric=True, normed=True)
    props = ['contrast', 'homogeneity', 'energy', 'correlation']
    vals = []
    for p in props:
        v = graycoprops(glcm, p)
        vals.append(float(np.mean(v)))
    return vals

def _dct_blockiness(gray, block_size=8):
    h, w = gray.shape
    h -= h % block_size
    w -= w % block_size
    gray = gray[:h, :w].astype(np.float32)
    dc_energy = 0.0
    ac_energy = 0.0
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            mean = np.mean(block)
            dc_energy += mean**2
            ac_energy += np.sum((block - mean)**2)
    return ac_energy / (dc_energy + ac_energy + 1e-8)

def _lbp_texture(gray, work_size=128):
    small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
    lbp = local_binary_pattern(small, P=8, R=1, method='uniform')
    return float(np.var(lbp))

def _image_entropy(gray):
    hist = np.histogram(gray, bins=64, range=(0, 256))[0]
    hist = hist / (hist.sum() + 1e-8)
    return -np.sum(hist * np.log2(hist + 1e-8))

def _clahe_entropy_delta(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return float(_image_entropy(equalized) - _image_entropy(gray))

def _color_fft_ratio(img_bgr):
    y, cr, cb = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb))
    f_cr = np.fft.fft2(cr.astype(np.float32))
    f_cb = np.fft.fft2(cb.astype(np.float32))
    mag_cr = np.abs(f_cr)
    mag_cb = np.abs(f_cb)
    h_energy = np.sum(mag_cr[:, :mag_cr.shape[1]//2]**2) + np.sum(mag_cb[:, :mag_cb.shape[1]//2]**2)
    v_energy = np.sum(mag_cr[:mag_cr.shape[0]//2, :]**2) + np.sum(mag_cb[:mag_cb.shape[0]//2, :]**2)
    return float(h_energy / (v_energy + 1e-8))

def _local_variance_heterogeneity(gray, patch_size=16):
    h, w = gray.shape
    variances = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            variances.append(np.var(patch))
    return float(np.std(variances))

def _edge_orientation_entropy(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(sobelx, sobely)
    ang = (ang * 180 / np.pi).astype(np.uint8)
    hist = np.histogram(ang, bins=36, range=(0, 180))[0]
    hist = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    return float(entropy)

def _noise_variance(gray, work_size=256):
    small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)
    cA, (cH, cV, cD) = pywt.dwt2(small, 'db4')
    mad = np.median(np.abs(cD))
    if mad < 1e-8:
        return 0.0
    return float((mad / 0.6745) ** 2)

def extract_features(img_bgr):
    # ----- Centre crop to 60% (discard background) -----
    h, w = img_bgr.shape[:2]
    crop_h, crop_w = int(h * 0.6), int(w * 0.6)
    y = (h - crop_h) // 2
    x = (w - crop_w) // 2
    img_bgr = img_bgr[y:y+crop_h, x:x+crop_w]

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
     dwt_energy_compaction) = _dwt_features(gray)
    glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation = _glcm_features(gray)
    lbp_var = _lbp_texture(gray)
    clahe_delta = _clahe_entropy_delta(gray)
    color_fft_ratio = _color_fft_ratio(img_bgr)
    dct_ac_ratio = _dct_blockiness(gray)
    local_var_hetero = _local_variance_heterogeneity(gray)
    edge_orient_entropy = _edge_orientation_entropy(gray)
    noise_var = _noise_variance(gray)
    brightness_center_ratio = _brightness_center_ratio(img_bgr)

    return np.array([
        fft_peak_ratio, fft_radial_std, colorfulness, sat_mean, sat_std,
        blue_bias, clip_frac, lap_var, line_rect_score,
        dwt_h_var, dwt_v_var, dwt_d_var,
        dwt_h_skew, dwt_v_skew, dwt_d_skew,
        dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
        dwt_energy_compaction,
        glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation,
        lbp_var, clahe_delta, color_fft_ratio, dct_ac_ratio,
        local_var_hetero, edge_orient_entropy,
        noise_var,
        brightness_center_ratio
    ], dtype=np.float32)


# ---------- Prediction endpoint (with rule‑based boost) ----------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('image')
        if file is None:
            data = request.get_json()
            if data and 'image' in data:
                img_data = data['image']
                if img_data.startswith('data:image'):
                    img_data = img_data.split(',')[1]
                img_bytes = base64.b64decode(img_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                return jsonify({'error': 'No image provided'}), 400
        else:
            img_bytes = file.read()
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Extract features
        feat = extract_features(img_bgr)
        X_scaled = scaler.transform(feat.reshape(1, -1))
        X_selected = selector.transform(X_scaled)
        raw_prob = float(clf.predict_proba(X_selected)[0, 1])

        # Logit‑shift calibration
        if raw_threshold is not None and 0.0 < raw_threshold < 1.0:
            logit_thresh = np.log(raw_threshold / (1.0 - raw_threshold))
            raw_clipped = np.clip(raw_prob, 1e-7, 1.0 - 1e-7)
            logit_raw = np.log(raw_clipped / (1.0 - raw_clipped))
            adjusted_logit = logit_raw - logit_thresh
            calibrated_prob = float(1.0 / (1.0 + np.exp(-adjusted_logit)))
        else:
            calibrated_prob = raw_prob

        # ----- RULE‑BASED BOOST: catch dim, blurry, bright‑centered screens -----
        # Get the individual features for this image
        brightness_center_ratio = feat[FEATURE_NAMES.index('brightness_center_ratio')]
        lap_var = feat[FEATURE_NAMES.index('lap_var')]
        if brightness_center_ratio > 1.05 and lap_var < 150:
            calibrated_prob = max(calibrated_prob, 0.65)

        return jsonify({
            'score': calibrated_prob,
            'raw_score': raw_prob,
            'threshold': raw_threshold,
            'prediction': 'screen' if calibrated_prob >= 0.5 else 'real',
            'confidence': calibrated_prob if calibrated_prob >= 0.5 else 1 - calibrated_prob
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host=host, port=port, debug=debug, use_reloader=False)
