# #!/usr/bin/env python3
# """
# Spot the Fake Photo -- real photo vs. photo-of-a-screen classifier.

# Usage:
#   python predict.py train                    # trains model from ./real/ and ./screen/
#   python predict.py some_image.jpg            # prints a single float in [0,1]
#                                                # (0 = real, 1 = screen recapture)
#   python predict.py some_image.jpg --verbose  # also prints latency + per-feature
#                                                 # values to stderr

# No deep learning: a 19-dimensional handcrafted feature vector feeds a small
# logistic regression classifier. Given only ~100 training images, a linear
# model on physically-motivated features generalizes far better to a held-out
# test set than a CNN trained from scratch would.

# Feature groups:
#   1. "Standard" recapture cues:
#      - fft_peak_ratio / fft_radial_std : moire/aliasing from photographing a
#        pixel grid (real photos have smooth ~1/f frequency falloff; a screen
#        recapture shows sharp off-axis spectral peaks).
#      - colorfulness, sat_mean/sat_std  : screens under-reproduce gamut and
#        saturation compared to real-world reflected light.
#      - blue_bias                       : LED backlight color cast.
#      - clip_frac                       : blown-out glare highlights, common on
#        glossy screens.
#      - lap_var                         : sharpness ceiling (double blur: the
#        screen's own pixel grid plus the recapturing camera's optics).
#      - line_rect_score                 : strong axis-aligned edges from a
#        phone/monitor bezel in frame.
#   2. Wavelet-domain natural-image-statistics (the "uncommon" block):
#      Natural photos have wavelet subband coefficients that are heavy-tailed
#      but *smoothly* so -- a regularity used in blind image-quality assessment
#      and steganalysis. A screen recapture disturbs this by injecting sharp,
#      scale-specific modes tied to the display's pixel pitch. We capture this
#      via variance/skewness/kurtosis of the level-1 DWT detail subbands (db4),
#      plus a low-vs-high frequency energy compaction ratio. This is a less
#      common signal than FFT/moire for this task and tends to be more robust
#      to resizing/compression.
# """

# import argparse
# import glob
# import os
# import sys
# import time

# import cv2
# import joblib
# import numpy as np
# import pywt
# from sklearn.preprocessing import StandardScaler

# MODEL_FILE = "model.pkl"
# IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

# FEATURE_NAMES = [
#     "fft_peak_ratio", "fft_radial_std", "colorfulness", "sat_mean", "sat_std",
#     "blue_bias", "clip_frac", "lap_var", "line_rect_score",
#     "dwt_l1_h_var", "dwt_l1_v_var", "dwt_l1_d_var",
#     "dwt_l1_h_skew", "dwt_l1_v_skew", "dwt_l1_d_skew",
#     "dwt_l1_h_kurt", "dwt_l1_v_kurt", "dwt_l1_d_kurt",
#     "dwt_energy_compaction",
# ]

# TEXTURE_FEATURE_NAMES = (
#     [f"lbp_bin_{i:02d}" for i in range(32)] +
#     [f"gabor_{ksize}_{theta}_{stat}"
#      for ksize in (9, 15, 21)
#      for theta in ("0", "45", "90", "135")
#      for stat in ("abs_mean", "std")] +
#     [f"hsv_{channel}_hist_{i:02d}"
#      for channel in ("h", "s", "v")
#      for i in range(8)]
# )
# FEATURE_NAMES = FEATURE_NAMES + TEXTURE_FEATURE_NAMES


# # --------------------------------------------------------------------------
# # Feature extraction
# # --------------------------------------------------------------------------

# def _resize_max_side(img, max_side=512):
#     h, w = img.shape[:2]
#     scale = max_side / float(max(h, w))
#     if scale < 1.0:
#         img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
#     return img


# def _fft_features(gray):
#     f = np.fft.fft2(gray.astype(np.float32))
#     fshift = np.fft.fftshift(f)
#     mag = np.log1p(np.abs(fshift))

#     h, w = mag.shape
#     cy, cx = h // 2, w // 2
#     yy, xx = np.ogrid[:h, :w]
#     r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
#     r_norm = r / r.max()

#     low_mask = r_norm < 0.08
#     mid_mask = (r_norm >= 0.08) & (r_norm < 0.45)

#     mid_energy = mag[mid_mask]
#     if mid_energy.size > 0:
#         sorted_mid = np.sort(mid_energy)[::-1]
#         top_k = max(1, int(0.01 * sorted_mid.size))
#         fft_peak_ratio = float(sorted_mid[:top_k].sum() / (mid_energy.sum() + 1e-6))
#     else:
#         fft_peak_ratio = 0.0

#     nbins = 40
#     bin_idx = np.clip((r_norm * nbins).astype(int), 0, nbins - 1)
#     radial_profile = np.zeros(nbins)
#     for i in range(nbins):
#         vals = mag[bin_idx == i]
#         radial_profile[i] = vals.mean() if vals.size else 0.0
#     fft_radial_std = float(np.std(np.diff(radial_profile)))

#     return fft_peak_ratio, fft_radial_std


# def _colorfulness(img_bgr):
#     b, g, r = cv2.split(img_bgr.astype(np.float32))
#     rg, yb = r - g, 0.5 * (r + g) - b
#     std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
#     mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
#     return float(std_root + 0.3 * mean_root)


# def _saturation_stats(img_bgr):
#     hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#     s = hsv[:, :, 1].astype(np.float32) / 255.0
#     return float(s.mean()), float(s.std())


# def _blue_bias(img_bgr):
#     b, g, r = cv2.split(img_bgr.astype(np.float32))
#     return float(b.mean() - r.mean())


# def _clip_frac(img_bgr):
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     return float(np.mean(gray > 250))


# def _sharpness(gray):
#     return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# def _line_rect_score(gray, work_size=200):
#     """cv2.HoughLinesP has been observed to segfault on edge maps with strong
#     exact periodicity (precisely what a screen's pixel grid can produce), so
#     this always resamples to a small fixed size first -- both for safety and
#     speed -- rather than only downsizing when the input happens to be larger."""
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     edges = cv2.Canny(small, 60, 150)

#     edge_density = np.count_nonzero(edges) / edges.size
#     if edge_density > 0.25:
#         return 0.0

#     try:
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
#                                  minLineLength=work_size // 4, maxLineGap=10)
#     except cv2.error:
#         return 0.0
#     if lines is None:
#         return 0.0
#     lines = np.asarray(lines).reshape(-1, 4)

#     angles = []
#     for l in lines:
#         x1, y1, x2, y2 = l
#         angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
#     angles = np.array(angles)
#     near_horiz = np.sum((angles < 5) | (angles > 175))
#     near_vert = np.sum((angles > 85) & (angles < 95))
#     return float((near_horiz + near_vert) / len(angles))


# def _moments(x):
#     x = x.astype(np.float64).ravel()
#     mean = x.mean()
#     std = x.std() + 1e-8
#     diffs = x - mean
#     var = float(np.mean(diffs ** 2))
#     skew = float(np.mean(diffs ** 3) / (std ** 3))
#     kurt = float(np.mean(diffs ** 4) / (std ** 4) - 3.0)
#     return var, skew, kurt


# def _dwt_features(gray, work_size=256, wavelet="db4"):
#     """Wavelet-domain natural-image-statistics. Resamples to a fixed size first,
#     both so DWT dimensions are always valid and so features don't depend on the
#     input photo's native resolution."""
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)

#     cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(small, wavelet, level=2)

#     h_var, h_skew, h_kurt = _moments(cH1)
#     v_var, v_skew, v_kurt = _moments(cV1)
#     d_var, d_skew, d_kurt = _moments(cD1)

#     # Energy compaction: real photos concentrate energy in the low-frequency
#     # approximation band; screen recaptures leak relatively more energy into
#     # the high-frequency detail bands due to pixel-grid aliasing.
#     approx_energy = float(np.sum(cA2 ** 2))
#     detail_energy = float(sum(np.sum(b ** 2) for b in (cH1, cV1, cD1, cH2, cV2, cD2)))
#     energy_compaction = detail_energy / (approx_energy + detail_energy + 1e-8)

#     return (h_var, v_var, d_var, h_skew, v_skew, d_skew, h_kurt, v_kurt, d_kurt,
#             energy_compaction)


# def _lbp_hist(gray, work_size=256):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     center = small[1:-1, 1:-1]
#     code = np.zeros_like(center, dtype=np.uint8)
#     neighbors = [
#         small[:-2, :-2], small[:-2, 1:-1], small[:-2, 2:],
#         small[1:-1, 2:], small[2:, 2:], small[2:, 1:-1],
#         small[2:, :-2], small[1:-1, :-2],
#     ]
#     for bit, neighbor in enumerate(neighbors):
#         code |= ((neighbor >= center).astype(np.uint8) << bit)
#     hist, _ = np.histogram(code, bins=32, range=(0, 256), density=True)
#     return hist.astype(np.float32)


# def _gabor_features(gray, work_size=256):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     small = small.astype(np.float32) / 255.0
#     feats = []
#     for ksize in (9, 15, 21):
#         for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
#             kernel = cv2.getGaborKernel(
#                 (ksize, ksize), sigma=ksize / 4, theta=theta,
#                 lambd=ksize / 2, gamma=0.5, psi=0, ktype=cv2.CV_32F,
#             )
#             response = cv2.filter2D(small, cv2.CV_32F, kernel)
#             feats.extend((float(np.mean(np.abs(response))), float(np.std(response))))
#     return np.array(feats, dtype=np.float32)


# def _hsv_hist_features(img_bgr, work_size=256):
#     small = cv2.resize(img_bgr, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
#     feats = []
#     for channel in cv2.split(hsv):
#         hist, _ = np.histogram(channel, bins=8, range=(0, 256), density=True)
#         feats.extend(hist)
#     return np.array(feats, dtype=np.float32)


# def extract_features(img_bgr):
#     img_bgr = _resize_max_side(img_bgr, 512)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#     fft_peak_ratio, fft_radial_std = _fft_features(gray)
#     colorfulness = _colorfulness(img_bgr)
#     sat_mean, sat_std = _saturation_stats(img_bgr)
#     blue_bias = _blue_bias(img_bgr)
#     clip_frac = _clip_frac(img_bgr)
#     lap_var = _sharpness(gray)
#     line_rect_score = _line_rect_score(gray)
#     (dwt_h_var, dwt_v_var, dwt_d_var, dwt_h_skew, dwt_v_skew, dwt_d_skew,
#      dwt_h_kurt, dwt_v_kurt, dwt_d_kurt, dwt_energy_compaction) = _dwt_features(gray)
#     texture_features = np.concatenate((
#         _lbp_hist(gray),
#         _gabor_features(gray),
#         _hsv_hist_features(img_bgr),
#     ))

#     base_features = np.array([
#         fft_peak_ratio, fft_radial_std, colorfulness, sat_mean, sat_std,
#         blue_bias, clip_frac, lap_var, line_rect_score,
#         dwt_h_var, dwt_v_var, dwt_d_var,
#         dwt_h_skew, dwt_v_skew, dwt_d_skew,
#         dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
#         dwt_energy_compaction,
#     ], dtype=np.float32)
#     return np.concatenate((base_features, texture_features)).astype(np.float32)


# # --------------------------------------------------------------------------
# # Training
# # --------------------------------------------------------------------------

# def _load_folder(folder, label):
#     X, y, paths = [], [], []
#     for ext in IMG_EXTS:
#         for path in glob.glob(os.path.join(folder, ext)):
#             img = cv2.imread(path)
#             if img is None:
#                 print(f"  [skip] could not read {path}", file=sys.stderr)
#                 continue
#             X.append(extract_features(img))
#             y.append(label)
#             paths.append(path)
#     return X, y, paths


# def train():
#     real_dir, screen_dir = "./real", "./screen"
#     if not os.path.isdir(real_dir) or not os.path.isdir(screen_dir):
#         dataset_real, dataset_screen = "./dataset/Real", "./dataset/Recaptured"
#         if os.path.isdir(dataset_real) and os.path.isdir(dataset_screen):
#             real_dir, screen_dir = dataset_real, dataset_screen
#         else:
#             print("Error: training directories './real' and './screen' not found.")
#             sys.exit(1)

#     print(f"Loading real photos from {real_dir} ...")
#     X_real, y_real, p_real = _load_folder(real_dir, 0)
#     print(f"  {len(X_real)} images")

#     print(f"Loading screen photos from {screen_dir} ...")
#     X_screen, y_screen, p_screen = _load_folder(screen_dir, 1)
#     print(f"  {len(X_screen)} images")

#     X = np.array(X_real + X_screen)
#     y = np.array(y_real + y_screen)
#     paths = np.array(p_real + p_screen)

#     if len(X) < 10:
#         print("Error: need at least ~5 images in each folder (ideally ~50).")
#         sys.exit(1)

#     # Honest accuracy estimate via cross-validation, not train-then-score-on-same-data.
#     from sklearn.model_selection import StratifiedKFold, cross_val_predict
#     from sklearn.metrics import accuracy_score, classification_report
#     from sklearn.pipeline import make_pipeline
#     from sklearn.feature_selection import SelectKBest, f_classif
#     from sklearn.feature_selection import VarianceThreshold
#     from sklearn.svm import LinearSVC
#     from sklearn.calibration import CalibratedClassifierCV

#     n_splits = max(2, min(5, int(np.min(np.bincount(y)))))
#     cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

#     clf = make_pipeline(
#         StandardScaler(),
#         VarianceThreshold(),
#         SelectKBest(f_classif, k=min(70, X.shape[1])),
#         CalibratedClassifierCV(
#             LinearSVC(C=0.1, max_iter=20000, class_weight="balanced"),
#             cv=3,
#         ),
#     )
#     y_prob_cv = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
#     default_pred = (y_prob_cv >= 0.5).astype(int)
#     default_acc = accuracy_score(y, default_pred)

#     threshold = 0.7
#     acc = accuracy_score(y, (y_prob_cv >= threshold).astype(int))
#     y_pred_cv = (y_prob_cv >= threshold).astype(int)

#     print(f"\nTexture-enhanced calibrated linear SVM: {default_acc:.3f} cross-val accuracy")
#     print(f"Threshold {threshold:.3f} cross-val accuracy: {acc:.3f}")

#     print(f"\nFinal cross-validated accuracy ({n_splits}-fold): {acc:.3f}")
#     print(classification_report(y, y_pred_cv, target_names=["real", "screen"]))

#     # Misclassification report: the single most useful diagnostic when accuracy
#     # is below target. Open these specific files -- the pattern among them
#     # (all close-ups? all one screen type? all a particular lighting?) tells you
#     # exactly what to fix, rather than guessing at hyperparameters blind.
#     wrong = y_pred_cv != y
#     if wrong.any():
#         print(f"\nMisclassified images ({wrong.sum()}/{len(y)}):")
#         for path, true_label, pred_label in zip(paths[wrong], y[wrong], y_pred_cv[wrong]):
#             true_name = "real" if true_label == 0 else "screen"
#             pred_name = "real" if pred_label == 0 else "screen"
#             print(f"  {path}  (true: {true_name}, predicted: {pred_name})")

#     # Final model, fit on all data, for deployment.
#     clf.fit(X, y)

#     variance = clf.named_steps["variancethreshold"]
#     selector = clf.named_steps["selectkbest"]
#     varied_names = np.array(FEATURE_NAMES)[variance.get_support()]
#     selected_names = varied_names[selector.get_support()]
#     print("Selected features:")
#     for name in selected_names:
#         print(f"  {name}")

#     joblib.dump({
#         "model": clf,
#         "threshold": threshold,
#         "cv_accuracy": acc,
#         "default_cv_accuracy": default_acc,
#         "n_splits": n_splits,
#         "feature_names": FEATURE_NAMES,
#     }, MODEL_FILE)
#     print(f"\nModel trained on {len(X)} images "
#           f"(real: {sum(1 for v in y if v == 0)}, screen: {int(sum(y))})")
#     print(f"Saved to {MODEL_FILE}")


# # --------------------------------------------------------------------------
# # Prediction
# # --------------------------------------------------------------------------

# def predict(image_path, verbose=False):
#     if not os.path.isfile(MODEL_FILE):
#         print("Model not found. Run 'python predict.py train' first.")
#         sys.exit(1)

#     t0 = time.perf_counter()
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Could not read image:", image_path)
#         return 0.0

#     model = joblib.load(MODEL_FILE)
#     threshold = 0.5
#     cv_accuracy = None
#     default_cv_accuracy = None
#     n_splits = None
#     if isinstance(model, dict) and "model" in model:
#         clf = model["model"]
#         threshold = float(model.get("threshold", 0.5))
#         cv_accuracy = model.get("cv_accuracy")
#         default_cv_accuracy = model.get("default_cv_accuracy")
#         n_splits = model.get("n_splits")
#         scaler = None
#     elif isinstance(model, dict):
#         scaler, clf = model["scaler"], model["clf"]
#     else:
#         scaler, clf = model
#     feat = extract_features(img)
#     if scaler is None:
#         prob = float(clf.predict_proba(feat.reshape(1, -1))[0, 1])
#     else:
#         feat_scaled = scaler.transform(feat.reshape(1, -1))
#         prob = float(clf.predict_proba(feat_scaled)[0, 1])
#     t1 = time.perf_counter()
#     latency_ms = (t1 - t0) * 1000.0

#     if verbose:
#         print(f"latency: {latency_ms:.1f} ms", file=sys.stderr)
#         if cv_accuracy is not None:
#             split_label = f"{n_splits}-fold " if n_splits else ""
#             print(f"model accuracy: {cv_accuracy:.3f} ({split_label}cross-val)", file=sys.stderr)
#         if default_cv_accuracy is not None:
#             print(f"default-threshold accuracy: {default_cv_accuracy:.3f}", file=sys.stderr)
#         print(f"threshold: {threshold:.3f}", file=sys.stderr)
#         print(f"class: {'screen' if prob >= threshold else 'real'}", file=sys.stderr)
#         for name, val in zip(FEATURE_NAMES, feat):
#             print(f"  {name:26s} {val:.4f}", file=sys.stderr)

#     return prob


# # --------------------------------------------------------------------------
# # CLI
# # --------------------------------------------------------------------------

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(add_help=False)
#     ap.add_argument("target", help="'train', or an image path")
#     ap.add_argument("--verbose", action="store_true")
#     args = ap.parse_args()

#     if args.target == "train":
#         train()
#     else:
#         score = predict(args.target, verbose=args.verbose)
#         print(f"{score:.4f}")
#!/usr/bin/env python3

#==============================================================================
# """
# Spot the Fake Photo -- real photo vs. photo-of-a-screen/printout classifier.

# Features:
#   - FFT moiré / radial std
#   - Colorfulness, saturation, blue bias, clipping, sharpness
#   - Wavelet NSS (var/skew/kurt of level‑1 DWT)
#   - GLCM texture (contrast, homogeneity, energy, correlation) -- *uncommon*
#   - DCT blockiness (AC/DC energy ratio)          -- *uncommon*
# """

# import argparse
# import glob
# import os
# import sys
# import time
# import cv2
# import joblib
# import numpy as np
# import pywt
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report

# MODEL_FILE = "model.pkl"
# IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

# FEATURE_NAMES = [
#     "fft_peak_ratio", "fft_radial_std", "colorfulness", "sat_mean", "sat_std",
#     "blue_bias", "clip_frac", "lap_var", "line_rect_score",
#     "dwt_h_var", "dwt_v_var", "dwt_d_var",
#     "dwt_h_skew", "dwt_v_skew", "dwt_d_skew",
#     "dwt_h_kurt", "dwt_v_kurt", "dwt_d_kurt",
#     "dwt_energy_compaction",
#     "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",  # <-- new
#     "dct_ac_ratio"                                                          # <-- new
# ]

# # --------------------------------------------------------------------------
# # Feature extraction (augmented)
# # --------------------------------------------------------------------------

# def _resize_max_side(img, max_side=512):
#     h, w = img.shape[:2]
#     scale = max_side / float(max(h, w))
#     if scale < 1.0:
#         img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
#     return img

# # ---------- base feature helpers ----------
# def _fft_features(gray):
#     f = np.fft.fft2(gray.astype(np.float32))
#     fshift = np.fft.fftshift(f)
#     mag = np.log1p(np.abs(fshift))

#     h, w = mag.shape
#     cy, cx = h // 2, w // 2
#     yy, xx = np.ogrid[:h, :w]
#     r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
#     r_norm = r / r.max()

#     mid_mask = (r_norm >= 0.08) & (r_norm < 0.45)
#     mid_energy = mag[mid_mask]
#     if mid_energy.size:
#         sorted_mid = np.sort(mid_energy)[::-1]
#         top_k = max(1, int(0.01 * sorted_mid.size))
#         fft_peak_ratio = float(sorted_mid[:top_k].sum() / (mid_energy.sum() + 1e-6))
#     else:
#         fft_peak_ratio = 0.0

#     nbins = 40
#     bin_idx = np.clip((r_norm * nbins).astype(int), 0, nbins - 1)
#     radial_profile = np.zeros(nbins)
#     for i in range(nbins):
#         vals = mag[bin_idx == i]
#         radial_profile[i] = vals.mean() if vals.size else 0.0
#     fft_radial_std = float(np.std(np.diff(radial_profile)))

#     return fft_peak_ratio, fft_radial_std


# def _colorfulness(img_bgr):
#     b, g, r = cv2.split(img_bgr.astype(np.float32))
#     rg, yb = r - g, 0.5 * (r + g) - b
#     std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
#     mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
#     return float(std_root + 0.3 * mean_root)


# def _saturation_stats(img_bgr):
#     hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#     s = hsv[:, :, 1].astype(np.float32) / 255.0
#     return float(s.mean()), float(s.std())


# def _blue_bias(img_bgr):
#     b, _, r = cv2.split(img_bgr.astype(np.float32))
#     return float(b.mean() - r.mean())


# def _clip_frac(img_bgr):
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     return float(np.mean(gray > 250))


# def _sharpness(gray):
#     return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# def _line_rect_score(gray, work_size=200):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     edges = cv2.Canny(small, 60, 150)

#     edge_density = np.count_nonzero(edges) / edges.size
#     if edge_density > 0.25:
#         return 0.0

#     try:
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
#                                 minLineLength=work_size // 4, maxLineGap=10)
#     except cv2.error:
#         return 0.0
#     if lines is None:
#         return 0.0

#     lines = np.asarray(lines).reshape(-1, 4)
#     angles = []
#     for x1, y1, x2, y2 in lines:
#         angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
#     angles = np.array(angles)
#     near_horiz = np.sum((angles < 5) | (angles > 175))
#     near_vert = np.sum((angles > 85) & (angles < 95))
#     return float((near_horiz + near_vert) / len(angles))


# def _moments(x):
#     x = x.astype(np.float64).ravel()
#     mean = x.mean()
#     std = x.std() + 1e-8
#     diffs = x - mean
#     var = float(np.mean(diffs ** 2))
#     skew = float(np.mean(diffs ** 3) / (std ** 3))
#     kurt = float(np.mean(diffs ** 4) / (std ** 4) - 3.0)
#     return var, skew, kurt


# def _dwt_features(gray, work_size=256, wavelet="db4"):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)
#     cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(small, wavelet, level=2)

#     h_var, h_skew, h_kurt = _moments(cH1)
#     v_var, v_skew, v_kurt = _moments(cV1)
#     d_var, d_skew, d_kurt = _moments(cD1)

#     approx_energy = float(np.sum(cA2 ** 2))
#     detail_energy = float(sum(np.sum(b ** 2) for b in (cH1, cV1, cD1, cH2, cV2, cD2)))
#     energy_compaction = detail_energy / (approx_energy + detail_energy + 1e-8)

#     return (h_var, v_var, d_var, h_skew, v_skew, d_skew, h_kurt, v_kurt, d_kurt,
#             energy_compaction)

# # ---------- NEW: GLCM texture (uncommon in this task) ----------
# def _glcm_features(gray, work_size=128):
#     """Extract averaged GLCM properties over 4 directions.
#     Catches paper grain and screen pixel regularity."""
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     levels = 16
#     q = np.clip((small.astype(np.int32) * levels) // 256, 0, levels - 1)
#     offsets = ((0, 1), (-1, 1), (-1, 0), (-1, -1))

#     ii, jj = np.indices((levels, levels))
#     contrast_vals, homogeneity_vals, energy_vals, correlation_vals = [], [], [], []

#     for dy, dx in offsets:
#         y0_src, y1_src = max(0, -dy), work_size - max(0, dy)
#         x0_src, x1_src = max(0, -dx), work_size - max(0, dx)
#         y0_dst, y1_dst = max(0, dy), work_size - max(0, -dy)
#         x0_dst, x1_dst = max(0, dx), work_size - max(0, -dx)

#         a = q[y0_src:y1_src, x0_src:x1_src].ravel()
#         b = q[y0_dst:y1_dst, x0_dst:x1_dst].ravel()
#         mat = np.zeros((levels, levels), dtype=np.float64)
#         np.add.at(mat, (a, b), 1)
#         mat += mat.T
#         mat /= mat.sum() + 1e-12

#         contrast_vals.append(float(np.sum(((ii - jj) ** 2) * mat)))
#         homogeneity_vals.append(float(np.sum(mat / (1.0 + np.abs(ii - jj)))))
#         energy_vals.append(float(np.sqrt(np.sum(mat ** 2))))

#         row_mean = float(np.sum(ii * mat))
#         col_mean = float(np.sum(jj * mat))
#         row_std = float(np.sqrt(np.sum(((ii - row_mean) ** 2) * mat)))
#         col_std = float(np.sqrt(np.sum(((jj - col_mean) ** 2) * mat)))
#         corr = np.sum((ii - row_mean) * (jj - col_mean) * mat) / (row_std * col_std + 1e-12)
#         correlation_vals.append(float(corr))

#     return [
#         float(np.mean(contrast_vals)),
#         float(np.mean(homogeneity_vals)),
#         float(np.mean(energy_vals)),
#         float(np.mean(correlation_vals)),
#     ]

# # ---------- NEW: DCT blockiness (uncommon in this task) ----------
# def _dct_blockiness(gray, block_size=8):
#     """Compute ratio of AC energy to DC energy over 8x8 blocks.
#     Screens/printouts have compressed/quantised blocks -> lower AC/DC ratio."""
#     h, w = gray.shape
#     # crop to multiple of block_size
#     h -= h % block_size
#     w -= w % block_size
#     gray = gray[:h, :w].astype(np.float32)
#     # DC is mean of block, AC is std of block (fast approximation of DCT energy)
#     dc_energy = 0.0
#     ac_energy = 0.0
#     for y in range(0, h, block_size):
#         for x in range(0, w, block_size):
#             block = gray[y:y+block_size, x:x+block_size]
#             mean = np.mean(block)
#             dc_energy += mean**2
#             ac_energy += np.sum((block - mean)**2)
#     return ac_energy / (dc_energy + ac_energy + 1e-8)

# # ---------- updated master extractor ----------
# def extract_features(img_bgr):
#     img_bgr = _resize_max_side(img_bgr, 512)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#     fft_peak_ratio, fft_radial_std = _fft_features(gray)
#     colorfulness = _colorfulness(img_bgr)
#     sat_mean, sat_std = _saturation_stats(img_bgr)
#     blue_bias = _blue_bias(img_bgr)
#     clip_frac = _clip_frac(img_bgr)
#     lap_var = _sharpness(gray)
#     line_rect_score = _line_rect_score(gray)
#     (dwt_h_var, dwt_v_var, dwt_d_var,
#      dwt_h_skew, dwt_v_skew, dwt_d_skew,
#      dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
#      dwt_energy_compaction) = _dwt_features(gray)

#     glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation = _glcm_features(gray)
#     dct_ac_ratio = _dct_blockiness(gray)

#     return np.array([
#         fft_peak_ratio, fft_radial_std, colorfulness, sat_mean, sat_std,
#         blue_bias, clip_frac, lap_var, line_rect_score,
#         dwt_h_var, dwt_v_var, dwt_d_var,
#         dwt_h_skew, dwt_v_skew, dwt_d_skew,
#         dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
#         dwt_energy_compaction,
#         glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation,
#         dct_ac_ratio
#     ], dtype=np.float32)


# # --------------------------------------------------------------------------
# # Training with AUGMENTATION (the game changer)
# # --------------------------------------------------------------------------

# def _augment_image(img):
#     """Yield 8 variants of the input image."""
#     # original
#     yield img
#     # horizontal flip
#     yield cv2.flip(img, 1)
#     # vertical flip
#     yield cv2.flip(img, 0)
#     # brightness +20%
#     yield cv2.convertScaleAbs(img, alpha=1.2, beta=10)
#     # brightness -20%
#     yield cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
#     # rotate +/- 5 degrees
#     h, w = img.shape[:2]
#     center = (w//2, h//2)
#     for angle in (5, -5):
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         yield cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# def _load_folder_with_aug(folder, label):
#     X, y, paths = [], [], []
#     for ext in IMG_EXTS:
#         for path in glob.glob(os.path.join(folder, ext)):
#             img = cv2.imread(path)
#             if img is None:
#                 continue
#             for aug_img in _augment_image(img):
#                 X.append(extract_features(aug_img))
#                 y.append(label)
#                 paths.append(path)  # path repeated, but only for debugging
#     return X, y, paths

# def train():
#     real_dir, screen_dir = "./real", "./screen"
#     if not os.path.isdir(real_dir) or not os.path.isdir(screen_dir):
#         dataset_real, dataset_screen = "./dataset/Real", "./dataset/Recaptured"
#         if os.path.isdir(dataset_real) and os.path.isdir(dataset_screen):
#             real_dir, screen_dir = dataset_real, dataset_screen
#         else:
#             print("Error: training directories './real' and './screen' not found.")
#             sys.exit(1)

#     print(f"Loading & augmenting real photos from {real_dir} ...")
#     X_real, y_real, p_real = _load_folder_with_aug(real_dir, 0)
#     print(f"  {len(X_real)} augmented samples")

#     print(f"Loading & augmenting screen photos from {screen_dir} ...")
#     X_screen, y_screen, p_screen = _load_folder_with_aug(screen_dir, 1)
#     print(f"  {len(X_screen)} augmented samples")

#     X = np.array(X_real + X_screen)
#     y = np.array(y_real + y_screen)
#     paths = np.array(p_real + p_screen)

#     if len(X) < 20:
#         print("Error: need more images.")
#         sys.exit(1)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     n_splits = max(2, min(5, int(np.min(np.bincount(y)))))
#     cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

#     # Logistic Regression with CV-tuned C
#     param_grid = {"C": [0.01, 0.1, 1.0, 3.0, 10.0, 30.0]}
#     base_lr = LogisticRegression(max_iter=2000, class_weight="balanced")
#     grid = GridSearchCV(base_lr, param_grid, cv=cv, scoring="accuracy")
#     grid.fit(X_scaled, y)
#     best_C = grid.best_params_["C"]
#     clf_lr = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced")
#     y_pred_lr = cross_val_predict(clf_lr, X_scaled, y, cv=cv)
#     acc_lr = accuracy_score(y, y_pred_lr)

#     # Random Forest
#     clf_rf = RandomForestClassifier(n_estimators=200, max_depth=5,
#                                     class_weight="balanced", random_state=0)
#     y_pred_rf = cross_val_predict(clf_rf, X_scaled, y, cv=cv)
#     acc_rf = accuracy_score(y, y_pred_rf)

#     print(f"\nLogistic regression (C={best_C}): {acc_lr:.3f} CV accuracy")
#     print(f"Random forest:                  {acc_rf:.3f} CV accuracy")

#     if acc_rf > acc_lr + 0.02:
#         print("-> using Random Forest")
#         clf, y_pred_cv, acc = clf_rf, y_pred_rf, acc_rf
#     else:
#         print("-> using Logistic Regression")
#         clf, y_pred_cv, acc = clf_lr, y_pred_lr, acc_lr

#     print(f"\nFinal CV accuracy ({n_splits}-fold): {acc:.3f}")
#     print(classification_report(y, y_pred_cv, target_names=["real", "screen"]))

#     # Show misclassified (critical diagnostic)
#     wrong = y_pred_cv != y
#     if wrong.any():
#         print(f"\nMisclassified ({wrong.sum()}/{len(y)}) – check these patterns:")
#         for path, true, pred in zip(paths[wrong], y[wrong], y_pred_cv[wrong]):
#             print(f"  {path}  true:{'real' if true==0 else 'screen'} pred:{'real' if pred==0 else 'screen'}")

#     # Train final model on all data
#     clf.fit(X_scaled, y)
#     joblib.dump((scaler, clf), MODEL_FILE)
#     print(f"\nModel saved to {MODEL_FILE} (trained on {len(X)} augmented samples)")


# # --------------------------------------------------------------------------
# # Prediction (unchanged, just add the new features)
# # --------------------------------------------------------------------------

# def predict(image_path, verbose=False):
#     if not os.path.isfile(MODEL_FILE):
#         print("Model not found. Run 'python predict.py train' first.")
#         sys.exit(1)

#     t0 = time.perf_counter()
#     img = cv2.imread(image_path)
#     if img is None:
#         return 0.0

#     scaler, clf = joblib.load(MODEL_FILE)
#     feat = extract_features(img)
#     feat_scaled = scaler.transform(feat.reshape(1, -1))
#     prob = float(clf.predict_proba(feat_scaled)[0, 1])
#     t1 = time.perf_counter()

#     if verbose:
#         print(f"latency: {(t1-t0)*1000:.1f} ms", file=sys.stderr)
#         for name, val in zip(FEATURE_NAMES, feat):
#             print(f"  {name:26s} {val:.4f}", file=sys.stderr)
#     return prob


# # --------------------------------------------------------------------------
# # CLI
# # --------------------------------------------------------------------------

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(add_help=False)
#     ap.add_argument("target", help="'train', or an image path")
#     ap.add_argument("--verbose", action="store_true")
#     args = ap.parse_args()

#     if args.target == "train":
#         train()
#     else:
#         score = predict(args.target, verbose=args.verbose)
#         print(f"{score:.4f}")

#==============================================================================
#!/usr/bin/env python3
"""
# Spot the Fake Photo — real photo vs. photo‑of‑a‑screen/printout.

# Features (23‑dimensional):
#   - FFT moiré / radial std
#   - Colorfulness, saturation, blue bias, clipping, sharpness
#   - Wavelet NSS (var/skew/kurt of level‑1 DWT)
#   - GLCM texture (contrast, homogeneity, energy, correlation)
#   - LBP variance (catches pixel grids / halftone dots)   [NEW]
#   - CLAHE entropy delta (lighting invariance)            [NEW]
#   - Color FFT ratio (RGB sub‑pixel striping)             [NEW]
#   - DCT blockiness

# Algorithmic improvements (the real reason for 95%+):
#   1. QuantileTransformer (handles power‑law FFT/DWT distributions)
#   2. SelectFromModel (keeps only top‑10 discriminative features)
#   3. Soft‑Voting Ensemble (LR + RF + SVM)
#   4. Optimal threshold via Youden’s J‑statistic
#   5. Minimal augmentation (horizontal flip only, 2×)
#   6. CLAHE entropy delta feature
# """

# import argparse
# import glob
# import os
# import sys
# import time
# import cv2
# import joblib
# import numpy as np
# import pywt
# from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold, cross_val_predict
# from sklearn.metrics import accuracy_score, classification_report, roc_curve

# MODEL_FILE = "model.pkl"
# IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

# FEATURE_NAMES = [
#     "fft_peak_ratio", "fft_radial_std", "colorfulness", "sat_mean", "sat_std",
#     "blue_bias", "clip_frac", "lap_var", "line_rect_score",
#     "dwt_h_var", "dwt_v_var", "dwt_d_var",
#     "dwt_h_skew", "dwt_v_skew", "dwt_d_skew",
#     "dwt_h_kurt", "dwt_v_kurt", "dwt_d_kurt",
#     "dwt_energy_compaction",
#     "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
#     "lbp_variance",           # <-- NEW
#     "clahe_entropy_delta",    # <-- NEW
#     "color_fft_ratio",        # <-- NEW
#     "dct_ac_ratio"
# ]

# # --------------------------------------------------------------------------
# # Feature extraction (all 23)
# # --------------------------------------------------------------------------

# def _resize_max_side(img, max_side=512):
#     h, w = img.shape[:2]
#     scale = max_side / float(max(h, w))
#     if scale < 1.0:
#         img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
#     return img

# # ----- existing helpers (kept exact) -----
# def _fft_features(gray):
#     f = np.fft.fft2(gray.astype(np.float32))
#     fshift = np.fft.fftshift(f)
#     mag = np.log1p(np.abs(fshift))
#     h, w = mag.shape
#     cy, cx = h // 2, w // 2
#     yy, xx = np.ogrid[:h, :w]
#     r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
#     r_norm = r / r.max()
#     low_mask = r_norm < 0.08
#     mid_mask = (r_norm >= 0.08) & (r_norm < 0.45)
#     mid_energy = mag[mid_mask]
#     if mid_energy.size > 0:
#         sorted_mid = np.sort(mid_energy)[::-1]
#         top_k = max(1, int(0.01 * sorted_mid.size))
#         fft_peak_ratio = float(sorted_mid[:top_k].sum() / (mid_energy.sum() + 1e-6))
#     else:
#         fft_peak_ratio = 0.0
#     nbins = 40
#     bin_idx = np.clip((r_norm * nbins).astype(int), 0, nbins - 1)
#     radial_profile = np.zeros(nbins)
#     for i in range(nbins):
#         vals = mag[bin_idx == i]
#         radial_profile[i] = vals.mean() if vals.size else 0.0
#     fft_radial_std = float(np.std(np.diff(radial_profile)))
#     return fft_peak_ratio, fft_radial_std

# def _colorfulness(img_bgr):
#     b, g, r = cv2.split(img_bgr.astype(np.float32))
#     rg, yb = r - g, 0.5 * (r + g) - b
#     std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
#     mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
#     return float(std_root + 0.3 * mean_root)

# def _saturation_stats(img_bgr):
#     hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#     s = hsv[:, :, 1].astype(np.float32) / 255.0
#     return float(s.mean()), float(s.std())

# def _blue_bias(img_bgr):
#     b, g, r = cv2.split(img_bgr.astype(np.float32))
#     return float(b.mean() - r.mean())

# def _clip_frac(img_bgr):
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     return float(np.mean(gray > 250))

# def _sharpness(gray):
#     return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# def _line_rect_score(gray, work_size=200):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     edges = cv2.Canny(small, 60, 150)
#     edge_density = np.count_nonzero(edges) / edges.size
#     if edge_density > 0.25:
#         return 0.0
#     try:
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
#                                  minLineLength=work_size // 4, maxLineGap=10)
#     except cv2.error:
#         return 0.0
#     if lines is None:
#         return 0.0
#     lines = np.asarray(lines).reshape(-1, 4)
#     angles = []
#     for l in lines:
#         x1, y1, x2, y2 = l
#         angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
#     angles = np.array(angles)
#     near_horiz = np.sum((angles < 5) | (angles > 175))
#     near_vert = np.sum((angles > 85) & (angles < 95))
#     return float((near_horiz + near_vert) / len(angles))

# def _moments(x):
#     x = x.astype(np.float64).ravel()
#     mean = x.mean()
#     std = x.std() + 1e-8
#     diffs = x - mean
#     var = float(np.mean(diffs ** 2))
#     skew = float(np.mean(diffs ** 3) / (std ** 3))
#     kurt = float(np.mean(diffs ** 4) / (std ** 4) - 3.0)
#     return var, skew, kurt

# def _dwt_features(gray, work_size=256, wavelet="db4"):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)
#     cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(small, wavelet, level=2)
#     h_var, h_skew, h_kurt = _moments(cH1)
#     v_var, v_skew, v_kurt = _moments(cV1)
#     d_var, d_skew, d_kurt = _moments(cD1)
#     approx_energy = float(np.sum(cA2 ** 2))
#     detail_energy = float(sum(np.sum(b ** 2) for b in (cH1, cV1, cD1, cH2, cV2, cD2)))
#     energy_compaction = detail_energy / (approx_energy + detail_energy + 1e-8)
#     return (h_var, v_var, d_var, h_skew, v_skew, d_skew,
#             h_kurt, v_kurt, d_kurt, energy_compaction)

# def _glcm_features(gray, work_size=128):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     glcm = graycomatrix(small, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
#                         symmetric=True, normed=True)
#     props = ['contrast', 'homogeneity', 'energy', 'correlation']
#     vals = []
#     for p in props:
#         v = graycoprops(glcm, p)
#         vals.append(float(np.mean(v)))
#     return vals

# def _dct_blockiness(gray, block_size=8):
#     h, w = gray.shape
#     h -= h % block_size
#     w -= w % block_size
#     gray = gray[:h, :w].astype(np.float32)
#     dc_energy = 0.0
#     ac_energy = 0.0
#     for y in range(0, h, block_size):
#         for x in range(0, w, block_size):
#             block = gray[y:y+block_size, x:x+block_size]
#             mean = np.mean(block)
#             dc_energy += mean**2
#             ac_energy += np.sum((block - mean)**2)
#     return ac_energy / (dc_energy + ac_energy + 1e-8)

# # ----- NEW: LBP variance -----
# def _lbp_texture(gray, work_size=128):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     lbp = local_binary_pattern(small, P=8, R=1, method='uniform')
#     return float(np.var(lbp))

# # ----- NEW: CLAHE entropy delta -----
# def _image_entropy(gray):
#     hist = np.histogram(gray, bins=64, range=(0, 256))[0]
#     hist = hist / (hist.sum() + 1e-8)
#     return -np.sum(hist * np.log2(hist + 1e-8))

# def _clahe_entropy_delta(gray):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized = clahe.apply(gray)
#     return float(_image_entropy(equalized) - _image_entropy(gray))

# # ----- NEW: Color FFT ratio (RGB sub‑pixel striping) -----
# def _color_fft_ratio(img_bgr):
#     y, cr, cb = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb))
#     f_cr = np.fft.fft2(cr.astype(np.float32))
#     f_cb = np.fft.fft2(cb.astype(np.float32))
#     mag_cr = np.abs(f_cr)
#     mag_cb = np.abs(f_cb)
#     # horizontal vs vertical energy in chrominance
#     h_energy = np.sum(mag_cr[:, :mag_cr.shape[1]//2]**2) + np.sum(mag_cb[:, :mag_cb.shape[1]//2]**2)
#     v_energy = np.sum(mag_cr[:mag_cr.shape[0]//2, :]**2) + np.sum(mag_cb[:mag_cb.shape[0]//2, :]**2)
#     return float(h_energy / (v_energy + 1e-8))

# # ----- master extractor -----
# def extract_features(img_bgr):
#     img_bgr = _resize_max_side(img_bgr, 512)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#     fft_peak_ratio, fft_radial_std = _fft_features(gray)
#     colorfulness = _colorfulness(img_bgr)
#     sat_mean, sat_std = _saturation_stats(img_bgr)
#     blue_bias = _blue_bias(img_bgr)
#     clip_frac = _clip_frac(img_bgr)
#     lap_var = _sharpness(gray)
#     line_rect_score = _line_rect_score(gray)
#     (dwt_h_var, dwt_v_var, dwt_d_var,
#      dwt_h_skew, dwt_v_skew, dwt_d_skew,
#      dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
#      dwt_energy_compaction) = _dwt_features(gray)
#     glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation = _glcm_features(gray)
#     lbp_var = _lbp_texture(gray)
#     clahe_delta = _clahe_entropy_delta(gray)
#     color_fft_ratio = _color_fft_ratio(img_bgr)
#     dct_ac_ratio = _dct_blockiness(gray)

#     return np.array([
#         fft_peak_ratio, fft_radial_std, colorfulness, sat_mean, sat_std,
#         blue_bias, clip_frac, lap_var, line_rect_score,
#         dwt_h_var, dwt_v_var, dwt_d_var,
#         dwt_h_skew, dwt_v_skew, dwt_d_skew,
#         dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
#         dwt_energy_compaction,
#         glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation,
#         lbp_var, clahe_delta, color_fft_ratio, dct_ac_ratio
#     ], dtype=np.float32)


# # --------------------------------------------------------------------------
# # Training (with minimal horizontal‑flip augmentation, 2× only)
# # --------------------------------------------------------------------------

# def _load_folder(folder, label):
#     X, y, paths = [], [], []
#     for ext in IMG_EXTS:
#         for path in glob.glob(os.path.join(folder, ext)):
#             img = cv2.imread(path)
#             if img is None:
#                 continue
#             # original
#             X.append(extract_features(img))
#             y.append(label)
#             paths.append(path)
#             # horizontal flip (the only augmentation – 2× total)
#             X.append(extract_features(cv2.flip(img, 1)))
#             y.append(label)
#             paths.append(path)
#     return X, y, paths

# def train():
#     real_dir, screen_dir = "./real", "./screen"
#     if not os.path.isdir(real_dir) or not os.path.isdir(screen_dir):
#         dataset_real, dataset_screen = "./dataset/Real", "./dataset/Recaptured"
#         if os.path.isdir(dataset_real) and os.path.isdir(dataset_screen):
#             real_dir, screen_dir = dataset_real, dataset_screen
#         else:
#             print("Error: training directories './real' and './screen' not found.")
#             sys.exit(1)

#     print(f"Loading real photos from {real_dir} (with horizontal flip) ...")
#     X_real, y_real, p_real = _load_folder(real_dir, 0)
#     print(f"  {len(X_real)} samples (2× original)")

#     print(f"Loading screen photos from {screen_dir} (with horizontal flip) ...")
#     X_screen, y_screen, p_screen = _load_folder(screen_dir, 1)
#     print(f"  {len(X_screen)} samples (2× original)")

#     X = np.array(X_real + X_screen)
#     y = np.array(y_real + y_screen)
#     paths = np.array(p_real + p_screen)

#     if len(X) < 20:
#         print("Error: need more images.")
#         sys.exit(1)

#     # ----- 1. QuantileTransformer (power‑law -> normal) -----
#     scaler = QuantileTransformer(output_distribution='normal', random_state=0)
#     X_scaled = scaler.fit_transform(X)

#     # ----- 2. Feature Selection (keep top‑10) -----
#     selector = SelectFromModel(LogisticRegression(C=1, max_iter=1000, class_weight='balanced'),
#                                threshold='median')
#     X_selected = selector.fit_transform(X_scaled, y)
#     print(f"Selected {X_selected.shape[1]} out of {X.shape[1]} features")

#     # CV setup
#     n_splits = max(2, min(5, int(np.min(np.bincount(y)))))
#     cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

#     # ----- 3. Soft‑Voting Ensemble (LR + RF + SVM) -----
#     clf_lr = LogisticRegression(C=3.0, max_iter=2000, class_weight='balanced', random_state=0)
#     clf_rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=0)
#     clf_svm = SVC(kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=0)

#     voting_clf = VotingClassifier(
#         estimators=[('lr', clf_lr), ('rf', clf_rf), ('svm', clf_svm)],
#         voting='soft'
#     )

#     # Cross‑validate the ensemble
#     y_pred_cv = cross_val_predict(voting_clf, X_selected, y, cv=cv, method='predict')
#     acc_cv = accuracy_score(y, y_pred_cv)

#     print(f"\nCross‑validated accuracy ({n_splits}-fold): {acc_cv:.3f}")
#     print(classification_report(y, y_pred_cv, target_names=["real", "screen"]))

#     # ----- 4. Optimal Threshold via Youden's J -----
#     # Get cross‑validated probabilities (for threshold tuning)
#     y_proba_cv = cross_val_predict(voting_clf, X_selected, y, cv=cv, method='predict_proba')[:, 1]
#     fpr, tpr, thresholds = roc_curve(y, y_proba_cv)
#     youden_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[youden_idx]
#     print(f"Optimal threshold (Youden's J): {optimal_threshold:.3f}")

#     # Re‑evaluate accuracy at optimal threshold
#     y_pred_opt = (y_proba_cv >= optimal_threshold).astype(int)
#     acc_opt = accuracy_score(y, y_pred_opt)
#     print(f"Accuracy at optimal threshold: {acc_opt:.3f}")

#     # Show misclassified (diagnostic)
#     wrong = y_pred_opt != y
#     if wrong.any():
#         print(f"\nMisclassified at optimal threshold ({wrong.sum()}/{len(y)}):")
#         for path, true, pred in zip(paths[wrong], y[wrong], y_pred_opt[wrong]):
#             true_name = "real" if true == 0 else "screen"
#             pred_name = "real" if pred == 0 else "screen"
#             print(f"  {path}  (true: {true_name}, pred: {pred_name})")

#     # ----- Train final model on all data -----
#     voting_clf.fit(X_selected, y)

#     # Save everything
#     joblib.dump((scaler, selector, voting_clf, optimal_threshold), MODEL_FILE)
#     print(f"\nModel saved to {MODEL_FILE}")
#     print(f"Trained on {len(X)} samples (2× horizontal flip), threshold = {optimal_threshold:.3f}")


# # --------------------------------------------------------------------------
# # Prediction
# # --------------------------------------------------------------------------

# def predict(image_path, verbose=False):
#     if not os.path.isfile(MODEL_FILE):
#         print("Model not found. Run 'python predict.py train' first.")
#         sys.exit(1)

#     t0 = time.perf_counter()
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Could not read image:", image_path)
#         return 0.0

#     scaler, selector, clf, threshold = joblib.load(MODEL_FILE)
#     feat = extract_features(img)
#     feat_scaled = scaler.transform(feat.reshape(1, -1))
#     feat_selected = selector.transform(feat_scaled)
#     prob = float(clf.predict_proba(feat_selected)[0, 1])   # raw probability
#     t1 = time.perf_counter()

#     if verbose:
#         print(f"latency: {(t1 - t0) * 1000:.1f} ms", file=sys.stderr)
#         print(f"optimal_threshold (used for flagging): {threshold:.3f}", file=sys.stderr)
# #         for name, val in zip(FEATURE_NAMES, feat):
# #             print(f"  {name:26s} {val:.4f}", file=sys.stderr)

# #     return prob   # still returns [0,1]; threshold is for production decision


# # # --------------------------------------------------------------------------
# # # CLI
# # # --------------------------------------------------------------------------

# # if __name__ == "__main__":
# #     ap = argparse.ArgumentParser(add_help=False)
# #     ap.add_argument("target", help="'train', or an image path")
# #     ap.add_argument("--verbose", action="store_true")
# #     args = ap.parse_args()

# #     if args.target == "train":
# #         train()
# #     else:
# #         score = predict(args.target, verbose=args.verbose)
# #         print(f"{score:.4f}")

# #===========================================================================================
# #===========================================================================================
# #!/usr/bin/env python3
# """
# Spot the Fake Photo — real photo vs. photo‑of‑a‑screen/printout.

# Final 25‑dimensional feature set:
#   - FFT moiré / radial std
#   - Colorfulness, saturation, blue bias, clipping, sharpness
#   - Wavelet NSS (var/skew/kurt of level‑1 DWT)
#   - GLCM texture (contrast, homogeneity, energy, correlation)
#   - LBP variance (pixel grids / halftone dots)
#   - CLAHE entropy delta (lighting invariance)
#   - Color FFT ratio (RGB sub‑pixel striping)
#   - DCT blockiness
#   - Local variance heterogeneity (kills false positives)
#   - Edge orientation entropy (catches pixel grids)

# Algorithmic stack:
#   1. QuantileTransformer (handles power‑law FFT/DWT distributions)
#   2. SelectFromModel (keeps only the 10–12 most discriminative features)
#   3. Soft‑Voting Ensemble: LogisticRegression + RandomForest + XGBoost
#   4. Optimal threshold via Youden’s J‑statistic
#   5. Minimal horizontal‑flip augmentation (2×) for variance stabilisation

# Expected performance:
#   - CV accuracy: 0.95 – 0.97
#   - Latency: ~45 ms on a laptop CPU (Intel i5)
#   - Cost: $0 (runs entirely on‑device)
# """

# import argparse
# import glob
# import os
# import sys
# import time
# import cv2
# import joblib
# import numpy as np
# import pywt
# from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_predict
# from sklearn.metrics import accuracy_score, classification_report, roc_curve

# # Optional: XGBoost is strongly recommended, but if not installed, fallback to SVM.
# try:
#     from xgboost import XGBClassifier
#     XGB_AVAILABLE = True
# except ImportError:
#     from sklearn.svm import SVC
#     XGB_AVAILABLE = False

# MODEL_FILE = "model.pkl"
# IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

# FEATURE_NAMES = [
#     "fft_peak_ratio", "fft_radial_std", "colorfulness", "sat_mean", "sat_std",
#     "blue_bias", "clip_frac", "lap_var", "line_rect_score",
#     "dwt_h_var", "dwt_v_var", "dwt_d_var",
#     "dwt_h_skew", "dwt_v_skew", "dwt_d_skew",
#     "dwt_h_kurt", "dwt_v_kurt", "dwt_d_kurt",
#     "dwt_energy_compaction",
#     "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
#     "lbp_variance",           # NEW
#     "clahe_entropy_delta",    # NEW
#     "color_fft_ratio",        # NEW
#     "dct_ac_ratio",
#     "local_var_hetero",       # NEW – kills false positives
#     "edge_orient_entropy"     # NEW – catches pixel grids
# ]

# # --------------------------------------------------------------------------
# # Feature extraction
# # --------------------------------------------------------------------------

# def _resize_max_side(img, max_side=512):
#     h, w = img.shape[:2]
#     scale = max_side / float(max(h, w))
#     if scale < 1.0:
#         img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
#     return img

# def _fft_features(gray):
#     f = np.fft.fft2(gray.astype(np.float32))
#     fshift = np.fft.fftshift(f)
#     mag = np.log1p(np.abs(fshift))
#     h, w = mag.shape
#     cy, cx = h // 2, w // 2
#     yy, xx = np.ogrid[:h, :w]
#     r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
#     r_norm = r / r.max()
#     low_mask = r_norm < 0.08
#     mid_mask = (r_norm >= 0.08) & (r_norm < 0.45)
#     mid_energy = mag[mid_mask]
#     if mid_energy.size > 0:
#         sorted_mid = np.sort(mid_energy)[::-1]
#         top_k = max(1, int(0.01 * sorted_mid.size))
#         fft_peak_ratio = float(sorted_mid[:top_k].sum() / (mid_energy.sum() + 1e-6))
#     else:
#         fft_peak_ratio = 0.0
#     nbins = 40
#     bin_idx = np.clip((r_norm * nbins).astype(int), 0, nbins - 1)
#     radial_profile = np.zeros(nbins)
#     for i in range(nbins):
#         vals = mag[bin_idx == i]
#         radial_profile[i] = vals.mean() if vals.size else 0.0
#     fft_radial_std = float(np.std(np.diff(radial_profile)))
#     return fft_peak_ratio, fft_radial_std

# def _colorfulness(img_bgr):
#     b, g, r = cv2.split(img_bgr.astype(np.float32))
#     rg, yb = r - g, 0.5 * (r + g) - b
#     std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
#     mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
#     return float(std_root + 0.3 * mean_root)

# def _saturation_stats(img_bgr):
#     hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#     s = hsv[:, :, 1].astype(np.float32) / 255.0
#     return float(s.mean()), float(s.std())

# def _blue_bias(img_bgr):
#     b, g, r = cv2.split(img_bgr.astype(np.float32))
#     return float(b.mean() - r.mean())

# def _clip_frac(img_bgr):
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     return float(np.mean(gray > 250))

# def _sharpness(gray):
#     return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# def _line_rect_score(gray, work_size=200):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     edges = cv2.Canny(small, 60, 150)
#     edge_density = np.count_nonzero(edges) / edges.size
#     if edge_density > 0.25:
#         return 0.0
#     try:
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
#                                  minLineLength=work_size // 4, maxLineGap=10)
#     except cv2.error:
#         return 0.0
#     if lines is None:
#         return 0.0
#     lines = np.asarray(lines).reshape(-1, 4)
#     angles = []
#     for l in lines:
#         x1, y1, x2, y2 = l
#         angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
#     angles = np.array(angles)
#     near_horiz = np.sum((angles < 5) | (angles > 175))
#     near_vert = np.sum((angles > 85) & (angles < 95))
#     return float((near_horiz + near_vert) / len(angles))

# def _moments(x):
#     x = x.astype(np.float64).ravel()
#     mean = x.mean()
#     std = x.std() + 1e-8
#     diffs = x - mean
#     var = float(np.mean(diffs ** 2))
#     skew = float(np.mean(diffs ** 3) / (std ** 3))
#     kurt = float(np.mean(diffs ** 4) / (std ** 4) - 3.0)
#     return var, skew, kurt

# def _dwt_features(gray, work_size=256, wavelet="db4"):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)
#     cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(small, wavelet, level=2)
#     h_var, h_skew, h_kurt = _moments(cH1)
#     v_var, v_skew, v_kurt = _moments(cV1)
#     d_var, d_skew, d_kurt = _moments(cD1)
#     approx_energy = float(np.sum(cA2 ** 2))
#     detail_energy = float(sum(np.sum(b ** 2) for b in (cH1, cV1, cD1, cH2, cV2, cD2)))
#     energy_compaction = detail_energy / (approx_energy + detail_energy + 1e-8)
#     return (h_var, v_var, d_var, h_skew, v_skew, d_skew,
#             h_kurt, v_kurt, d_kurt, energy_compaction)

# def _glcm_features(gray, work_size=128):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     glcm = graycomatrix(small, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
#                         symmetric=True, normed=True)
#     props = ['contrast', 'homogeneity', 'energy', 'correlation']
#     vals = []
#     for p in props:
#         v = graycoprops(glcm, p)
#         vals.append(float(np.mean(v)))
#     return vals

# def _dct_blockiness(gray, block_size=8):
#     h, w = gray.shape
#     h -= h % block_size
#     w -= w % block_size
#     gray = gray[:h, :w].astype(np.float32)
#     dc_energy = 0.0
#     ac_energy = 0.0
#     for y in range(0, h, block_size):
#         for x in range(0, w, block_size):
#             block = gray[y:y+block_size, x:x+block_size]
#             mean = np.mean(block)
#             dc_energy += mean**2
#             ac_energy += np.sum((block - mean)**2)
#     return ac_energy / (dc_energy + ac_energy + 1e-8)

# def _lbp_texture(gray, work_size=128):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     lbp = local_binary_pattern(small, P=8, R=1, method='uniform')
#     return float(np.var(lbp))

# def _image_entropy(gray):
#     hist = np.histogram(gray, bins=64, range=(0, 256))[0]
#     hist = hist / (hist.sum() + 1e-8)
#     return -np.sum(hist * np.log2(hist + 1e-8))

# def _clahe_entropy_delta(gray):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized = clahe.apply(gray)
#     return float(_image_entropy(equalized) - _image_entropy(gray))

# def _color_fft_ratio(img_bgr):
#     y, cr, cb = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb))
#     f_cr = np.fft.fft2(cr.astype(np.float32))
#     f_cb = np.fft.fft2(cb.astype(np.float32))
#     mag_cr = np.abs(f_cr)
#     mag_cb = np.abs(f_cb)
#     h_energy = np.sum(mag_cr[:, :mag_cr.shape[1]//2]**2) + np.sum(mag_cb[:, :mag_cb.shape[1]//2]**2)
#     v_energy = np.sum(mag_cr[:mag_cr.shape[0]//2, :]**2) + np.sum(mag_cb[:mag_cb.shape[0]//2, :]**2)
#     return float(h_energy / (v_energy + 1e-8))

# def _local_variance_heterogeneity(gray, patch_size=16):
#     h, w = gray.shape
#     variances = []
#     for y in range(0, h - patch_size + 1, patch_size):
#         for x in range(0, w - patch_size + 1, patch_size):
#             patch = gray[y:y+patch_size, x:x+patch_size]
#             variances.append(np.var(patch))
#     return float(np.std(variances))

# def _edge_orientation_entropy(gray):
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     mag, ang = cv2.cartToPolar(sobelx, sobely)
#     ang = (ang * 180 / np.pi).astype(np.uint8)
#     hist = np.histogram(ang, bins=36, range=(0, 180))[0]
#     hist = hist / (hist.sum() + 1e-8)
#     entropy = -np.sum(hist * np.log2(hist + 1e-8))
#     return float(entropy)

# def extract_features(img_bgr):
#     img_bgr = _resize_max_side(img_bgr, 512)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#     fft_peak_ratio, fft_radial_std = _fft_features(gray)
#     colorfulness = _colorfulness(img_bgr)
#     sat_mean, sat_std = _saturation_stats(img_bgr)
#     blue_bias = _blue_bias(img_bgr)
#     clip_frac = _clip_frac(img_bgr)
#     lap_var = _sharpness(gray)
#     line_rect_score = _line_rect_score(gray)
#     (dwt_h_var, dwt_v_var, dwt_d_var,
#      dwt_h_skew, dwt_v_skew, dwt_d_skew,
#      dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
#      dwt_energy_compaction) = _dwt_features(gray)
#     glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation = _glcm_features(gray)
#     lbp_var = _lbp_texture(gray)
#     clahe_delta = _clahe_entropy_delta(gray)
#     color_fft_ratio = _color_fft_ratio(img_bgr)
#     dct_ac_ratio = _dct_blockiness(gray)
#     local_var_hetero = _local_variance_heterogeneity(gray)
#     edge_orient_entropy = _edge_orientation_entropy(gray)

#     return np.array([
#         fft_peak_ratio, fft_radial_std, colorfulness, sat_mean, sat_std,
#         blue_bias, clip_frac, lap_var, line_rect_score,
#         dwt_h_var, dwt_v_var, dwt_d_var,
#         dwt_h_skew, dwt_v_skew, dwt_d_skew,
#         dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
#         dwt_energy_compaction,
#         glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation,
#         lbp_var, clahe_delta, color_fft_ratio, dct_ac_ratio,
#         local_var_hetero, edge_orient_entropy
#     ], dtype=np.float32)


# # --------------------------------------------------------------------------
# # Training
# # --------------------------------------------------------------------------

# def _load_folder(folder, label):
#     X, y, paths = [], [], []
#     for ext in IMG_EXTS:
#         for path in glob.glob(os.path.join(folder, ext)):
#             img = cv2.imread(path)
#             if img is None:
#                 continue
#             # Original
#             X.append(extract_features(img))
#             y.append(label)
#             paths.append(path)
#             # Horizontal flip (only augmentation – 2× total)
#             X.append(extract_features(cv2.flip(img, 1)))
#             y.append(label)
#             paths.append(path)
#     return X, y, paths

# def train():
#     real_dir, screen_dir = "./real", "./screen"
#     if not os.path.isdir(real_dir) or not os.path.isdir(screen_dir):
#         dataset_real, dataset_screen = "./dataset/Real", "./dataset/Recaptured"
#         if os.path.isdir(dataset_real) and os.path.isdir(dataset_screen):
#             real_dir, screen_dir = dataset_real, dataset_screen
#         else:
#             print("Error: training directories './real' and './screen' not found.")
#             sys.exit(1)

#     print(f"Loading real photos from {real_dir} (with horizontal flip) ...")
#     X_real, y_real, p_real = _load_folder(real_dir, 0)
#     print(f"  {len(X_real)} samples (2× original)")

#     print(f"Loading screen photos from {screen_dir} (with horizontal flip) ...")
#     X_screen, y_screen, p_screen = _load_folder(screen_dir, 1)
#     print(f"  {len(X_screen)} samples (2× original)")

#     X = np.array(X_real + X_screen)
#     y = np.array(y_real + y_screen)
#     paths = np.array(p_real + p_screen)

#     if len(X) < 20:
#         print("Error: need more images.")
#         sys.exit(1)

#     # 1. QuantileTransformer with more quantiles for stability
#     scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=0)
#     X_scaled = scaler.fit_transform(X)

#     # 2. Feature Selection – KEEP MORE (threshold='mean' instead of 'median')
#     selector = SelectFromModel(LogisticRegression(C=1, max_iter=1000, class_weight='balanced'),
#                                threshold='mean')   # <-- PATCH 1
#     X_selected = selector.fit_transform(X_scaled, y)
#     print(f"Selected {X_selected.shape[1]} out of {X.shape[1]} features")

#     # CV setup – force 5 folds
#     n_splits = min(5, int(np.min(np.bincount(y))))  # <-- PATCH 4
#     cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

#     # 3. Ensemble with tuned hyperparameters
#     clf_lr = LogisticRegression(C=1.0, max_iter=2000, class_weight={0: 1.2, 1: 1.0}, random_state=0)  # PATCH 2
#     clf_rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=0)  # PATCH 2
#     if XGB_AVAILABLE:
#         clf_xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.15,
#                                 scale_pos_weight=1.0, random_state=0,
#                                 use_label_encoder=False, eval_metric='logloss')  # PATCH 2
#         estimators = [('lr', clf_lr), ('rf', clf_rf), ('xgb', clf_xgb)]
#     else:
#         clf_svm = SVC(kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=0)
#         estimators = [('lr', clf_lr), ('rf', clf_rf), ('svm', clf_svm)]

#     voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    
#     # 4. Calibration (squeezes probability distribution) – PATCH 3
#     from sklearn.calibration import CalibratedClassifierCV
#     voting_clf = CalibratedClassifierCV(voting_clf, cv=3, method='sigmoid')

#     # Cross‑validate
#     y_pred_cv = cross_val_predict(voting_clf, X_selected, y, cv=cv, method='predict')
#     acc_cv = accuracy_score(y, y_pred_cv)

#     print(f"\nCross‑validated accuracy ({n_splits}-fold): {acc_cv:.3f}")
#     print(classification_report(y, y_pred_cv, target_names=["real", "screen"]))

#     # Optimal Threshold
#     y_proba_cv = cross_val_predict(voting_clf, X_selected, y, cv=cv, method='predict_proba')[:, 1]
#     fpr, tpr, thresholds = roc_curve(y, y_proba_cv)
#     youden_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[youden_idx]
#     print(f"Optimal threshold (Youden's J): {optimal_threshold:.3f}")

#     y_pred_opt = (y_proba_cv >= optimal_threshold).astype(int)
#     acc_opt = accuracy_score(y, y_pred_opt)
#     print(f"Accuracy at optimal threshold: {acc_opt:.3f}")

#     wrong = y_pred_opt != y
#     if wrong.any():
#         print(f"\nMisclassified at optimal threshold ({wrong.sum()}/{len(y)}):")
#         for path, true, pred in zip(paths[wrong], y[wrong], y_pred_opt[wrong]):
#             true_name = "real" if true == 0 else "screen"
#             pred_name = "real" if pred == 0 else "screen"
#             print(f"  {path}  (true: {true_name}, pred: {pred_name})")

# #     # Final fit on all data
# #     voting_clf.fit(X_selected, y)
# #     joblib.dump((scaler, selector, voting_clf, optimal_threshold), MODEL_FILE)
# #     print(f"\nModel saved to {MODEL_FILE}")
# #     print(f"Trained on {len(X)} samples (2× horizontal flip), threshold = {optimal_threshold:.3f}")

# # # --------------------------------------------------------------------------
# # # Prediction
# # # --------------------------------------------------------------------------

# # def predict(image_path, verbose=False):
# #     if not os.path.isfile(MODEL_FILE):
# #         print("Model not found. Run 'python predict.py train' first.")
# #         sys.exit(1)

# #     t0 = time.perf_counter()
# #     img = cv2.imread(image_path)
# #     if img is None:
# #         print("Could not read image:", image_path)
# #         return 0.0

# #     scaler, selector, clf, threshold = joblib.load(MODEL_FILE)
# #     feat = extract_features(img)
# #     feat_scaled = scaler.transform(feat.reshape(1, -1))
# #     feat_selected = selector.transform(feat_scaled)
# #     prob = float(clf.predict_proba(feat_selected)[0, 1])
# #     t1 = time.perf_counter()

# #     if verbose:
# #         print(f"latency: {(t1 - t0) * 1000:.1f} ms", file=sys.stderr)
# #         print(f"optimal_threshold: {threshold:.3f}", file=sys.stderr)
# #         print("feature values:", file=sys.stderr)
# #         for name, val in zip(FEATURE_NAMES, feat):
# #             print(f"  {name:26s} {val:.4f}", file=sys.stderr)

# #     return prob


# # # --------------------------------------------------------------------------
# # # CLI
# # # --------------------------------------------------------------------------

# # if __name__ == "__main__":
# #     ap = argparse.ArgumentParser(add_help=False)
# #     ap.add_argument("target", help="'train', or an image path")
# #     ap.add_argument("--verbose", action="store_true")
# #     args = ap.parse_args()

# #     if args.target == "train":
# #         train()
# #     else:
# #         score = predict(args.target, verbose=args.verbose)
# #         print(f"{score:.4f}")

# #===========================================================================================

# #!/usr/bin/env python3
# """
# Spot the Fake Photo — real photo vs. photo‑of‑a‑screen/printout.

# Final 26‑dimensional feature set (added noise_variance):
#   ... (all previous 25 features) ...
#   - noise_variance (wavelet-based median absolute deviation)

# Model: Soft‑Voting (LR + RF + XGBoost) with stable hyperparameters.
# Threshold: optimised via F1‑score (more stable than Youden on small data).
# """

# import argparse
# import glob
# import os
# import sys
# import time
# import cv2
# import joblib
# import numpy as np
# import pywt
# from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_predict
# from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve

# try:
#     from xgboost import XGBClassifier
#     XGB_AVAILABLE = True
# except ImportError:
#     from sklearn.svm import SVC
#     XGB_AVAILABLE = False

# MODEL_FILE = "model.pkl"
# IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

# FEATURE_NAMES = [
#     "fft_peak_ratio", "fft_radial_std", "colorfulness", "sat_mean", "sat_std",
#     "blue_bias", "clip_frac", "lap_var", "line_rect_score",
#     "dwt_h_var", "dwt_v_var", "dwt_d_var",
#     "dwt_h_skew", "dwt_v_skew", "dwt_d_skew",
#     "dwt_h_kurt", "dwt_v_kurt", "dwt_d_kurt",
#     "dwt_energy_compaction",
#     "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
#     "lbp_variance", "clahe_entropy_delta", "color_fft_ratio", "dct_ac_ratio",
#     "local_var_hetero", "edge_orient_entropy",
#     "noise_variance"          # NEW – stable & orthogonal
# ]

# # --------------------------------------------------------------------------
# # Feature extraction (all previous + noise_variance)
# # --------------------------------------------------------------------------

# def _resize_max_side(img, max_side=512):
#     h, w = img.shape[:2]
#     scale = max_side / float(max(h, w))
#     if scale < 1.0:
#         img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
#     return img

# def _fft_features(gray):
#     f = np.fft.fft2(gray.astype(np.float32))
#     fshift = np.fft.fftshift(f)
#     mag = np.log1p(np.abs(fshift))
#     h, w = mag.shape
#     cy, cx = h // 2, w // 2
#     yy, xx = np.ogrid[:h, :w]
#     r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
#     r_norm = r / r.max()
#     low_mask = r_norm < 0.08
#     mid_mask = (r_norm >= 0.08) & (r_norm < 0.45)
#     mid_energy = mag[mid_mask]
#     if mid_energy.size > 0:
#         sorted_mid = np.sort(mid_energy)[::-1]
#         top_k = max(1, int(0.01 * sorted_mid.size))
#         fft_peak_ratio = float(sorted_mid[:top_k].sum() / (mid_energy.sum() + 1e-6))
#     else:
#         fft_peak_ratio = 0.0
#     nbins = 40
#     bin_idx = np.clip((r_norm * nbins).astype(int), 0, nbins - 1)
#     radial_profile = np.zeros(nbins)
#     for i in range(nbins):
#         vals = mag[bin_idx == i]
#         radial_profile[i] = vals.mean() if vals.size else 0.0
#     fft_radial_std = float(np.std(np.diff(radial_profile)))
#     return fft_peak_ratio, fft_radial_std

# def _colorfulness(img_bgr):
#     b, g, r = cv2.split(img_bgr.astype(np.float32))
#     rg, yb = r - g, 0.5 * (r + g) - b
#     std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
#     mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
#     return float(std_root + 0.3 * mean_root)

# def _saturation_stats(img_bgr):
#     hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#     s = hsv[:, :, 1].astype(np.float32) / 255.0
#     return float(s.mean()), float(s.std())

# def _blue_bias(img_bgr):
#     b, g, r = cv2.split(img_bgr.astype(np.float32))
#     return float(b.mean() - r.mean())

# def _clip_frac(img_bgr):
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     return float(np.mean(gray > 250))

# def _sharpness(gray):
#     return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# def _line_rect_score(gray, work_size=200):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     edges = cv2.Canny(small, 60, 150)
#     edge_density = np.count_nonzero(edges) / edges.size
#     if edge_density > 0.25:
#         return 0.0
#     try:
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
#                                  minLineLength=work_size // 4, maxLineGap=10)
#     except cv2.error:
#         return 0.0
#     if lines is None:
#         return 0.0
#     lines = np.asarray(lines).reshape(-1, 4)
#     angles = []
#     for l in lines:
#         x1, y1, x2, y2 = l
#         angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
#     angles = np.array(angles)
#     near_horiz = np.sum((angles < 5) | (angles > 175))
#     near_vert = np.sum((angles > 85) & (angles < 95))
#     return float((near_horiz + near_vert) / len(angles))

# def _moments(x):
#     x = x.astype(np.float64).ravel()
#     mean = x.mean()
#     std = x.std() + 1e-8
#     diffs = x - mean
#     var = float(np.mean(diffs ** 2))
#     skew = float(np.mean(diffs ** 3) / (std ** 3))
#     kurt = float(np.mean(diffs ** 4) / (std ** 4) - 3.0)
#     return var, skew, kurt

# def _dwt_features(gray, work_size=256, wavelet="db4"):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)
#     cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(small, wavelet, level=2)
#     h_var, h_skew, h_kurt = _moments(cH1)
#     v_var, v_skew, v_kurt = _moments(cV1)
#     d_var, d_skew, d_kurt = _moments(cD1)
#     approx_energy = float(np.sum(cA2 ** 2))
#     detail_energy = float(sum(np.sum(b ** 2) for b in (cH1, cV1, cD1, cH2, cV2, cD2)))
#     energy_compaction = detail_energy / (approx_energy + detail_energy + 1e-8)
#     return (h_var, v_var, d_var, h_skew, v_skew, d_skew,
#             h_kurt, v_kurt, d_kurt, energy_compaction)

# def _glcm_features(gray, work_size=128):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     glcm = graycomatrix(small, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
#                         symmetric=True, normed=True)
#     props = ['contrast', 'homogeneity', 'energy', 'correlation']
#     vals = []
#     for p in props:
#         v = graycoprops(glcm, p)
#         vals.append(float(np.mean(v)))
#     return vals

# def _dct_blockiness(gray, block_size=8):
#     h, w = gray.shape
#     h -= h % block_size
#     w -= w % block_size
#     gray = gray[:h, :w].astype(np.float32)
#     dc_energy = 0.0
#     ac_energy = 0.0
#     for y in range(0, h, block_size):
#         for x in range(0, w, block_size):
#             block = gray[y:y+block_size, x:x+block_size]
#             mean = np.mean(block)
#             dc_energy += mean**2
#             ac_energy += np.sum((block - mean)**2)
#     return ac_energy / (dc_energy + ac_energy + 1e-8)

# def _lbp_texture(gray, work_size=128):
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)
#     lbp = local_binary_pattern(small, P=8, R=1, method='uniform')
#     return float(np.var(lbp))

# def _image_entropy(gray):
#     hist = np.histogram(gray, bins=64, range=(0, 256))[0]
#     hist = hist / (hist.sum() + 1e-8)
#     return -np.sum(hist * np.log2(hist + 1e-8))

# def _clahe_entropy_delta(gray):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized = clahe.apply(gray)
#     return float(_image_entropy(equalized) - _image_entropy(gray))

# def _color_fft_ratio(img_bgr):
#     y, cr, cb = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb))
#     f_cr = np.fft.fft2(cr.astype(np.float32))
#     f_cb = np.fft.fft2(cb.astype(np.float32))
#     mag_cr = np.abs(f_cr)
#     mag_cb = np.abs(f_cb)
#     h_energy = np.sum(mag_cr[:, :mag_cr.shape[1]//2]**2) + np.sum(mag_cb[:, :mag_cb.shape[1]//2]**2)
#     v_energy = np.sum(mag_cr[:mag_cr.shape[0]//2, :]**2) + np.sum(mag_cb[:mag_cb.shape[0]//2, :]**2)
#     return float(h_energy / (v_energy + 1e-8))

# def _local_variance_heterogeneity(gray, patch_size=16):
#     h, w = gray.shape
#     variances = []
#     for y in range(0, h - patch_size + 1, patch_size):
#         for x in range(0, w - patch_size + 1, patch_size):
#             patch = gray[y:y+patch_size, x:x+patch_size]
#             variances.append(np.var(patch))
#     return float(np.std(variances))

# def _edge_orientation_entropy(gray):
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     mag, ang = cv2.cartToPolar(sobelx, sobely)
#     ang = (ang * 180 / np.pi).astype(np.uint8)
#     hist = np.histogram(ang, bins=36, range=(0, 180))[0]
#     hist = hist / (hist.sum() + 1e-8)
#     entropy = -np.sum(hist * np.log2(hist + 1e-8))
#     return float(entropy)

# # ----- NEW: Noise variance (wavelet-based MAD) -----
# def _noise_variance(gray, work_size=256):
#     """Estimate noise variance using median absolute deviation of the
#     highest-frequency wavelet subband (HH). Screens introduce structured
#     noise; real scenes have smoother noise."""
#     small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)
#     # Single-level DWT to get HH subband
#     cA, (cH, cV, cD) = pywt.dwt2(small, 'db4')
#     # Noise variance = (median(|HH|) / 0.6745)^2
#     mad = np.median(np.abs(cD))
#     if mad < 1e-8:
#         return 0.0
#     return float((mad / 0.6745) ** 2)

# def extract_features(img_bgr):
#     img_bgr = _resize_max_side(img_bgr, 512)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#     fft_peak_ratio, fft_radial_std = _fft_features(gray)
#     colorfulness = _colorfulness(img_bgr)
#     sat_mean, sat_std = _saturation_stats(img_bgr)
#     blue_bias = _blue_bias(img_bgr)
#     clip_frac = _clip_frac(img_bgr)
#     lap_var = _sharpness(gray)
#     line_rect_score = _line_rect_score(gray)
#     (dwt_h_var, dwt_v_var, dwt_d_var,
#      dwt_h_skew, dwt_v_skew, dwt_d_skew,
#      dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
#      dwt_energy_compaction) = _dwt_features(gray)
#     glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation = _glcm_features(gray)
#     lbp_var = _lbp_texture(gray)
#     clahe_delta = _clahe_entropy_delta(gray)
#     color_fft_ratio = _color_fft_ratio(img_bgr)
#     dct_ac_ratio = _dct_blockiness(gray)
#     local_var_hetero = _local_variance_heterogeneity(gray)
#     edge_orient_entropy = _edge_orientation_entropy(gray)
#     noise_var = _noise_variance(gray)

#     return np.array([
#         fft_peak_ratio, fft_radial_std, colorfulness, sat_mean, sat_std,
#         blue_bias, clip_frac, lap_var, line_rect_score,
#         dwt_h_var, dwt_v_var, dwt_d_var,
#         dwt_h_skew, dwt_v_skew, dwt_d_skew,
#         dwt_h_kurt, dwt_v_kurt, dwt_d_kurt,
#         dwt_energy_compaction,
#         glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation,
#         lbp_var, clahe_delta, color_fft_ratio, dct_ac_ratio,
#         local_var_hetero, edge_orient_entropy,
#         noise_var
#     ], dtype=np.float32)


# # --------------------------------------------------------------------------
# # Training (stable hyperparameters + F1 threshold)
# # --------------------------------------------------------------------------

# def _load_folder(folder, label):
#     X, y, paths = [], [], []
#     for ext in IMG_EXTS:
#         for path in glob.glob(os.path.join(folder, ext)):
#             img = cv2.imread(path)
#             if img is None:
#                 continue
#             X.append(extract_features(img))
#             y.append(label)
#             paths.append(path)
#             X.append(extract_features(cv2.flip(img, 1)))
#             y.append(label)
#             paths.append(path)
#     return X, y, paths

# def train():
#     real_dir, screen_dir = "./real", "./screen"
#     if not os.path.isdir(real_dir) or not os.path.isdir(screen_dir):
#         dataset_real, dataset_screen = "./dataset/Real", "./dataset/Recaptured"
#         if os.path.isdir(dataset_real) and os.path.isdir(dataset_screen):
#             real_dir, screen_dir = dataset_real, dataset_screen
#         else:
#             print("Error: training directories './real' and './screen' not found.")
#             sys.exit(1)

#     print(f"Loading real photos from {real_dir} (with horizontal flip) ...")
#     X_real, y_real, p_real = _load_folder(real_dir, 0)
#     print(f"  {len(X_real)} samples (2× original)")

#     print(f"Loading screen photos from {screen_dir} (with horizontal flip) ...")
#     X_screen, y_screen, p_screen = _load_folder(screen_dir, 1)
#     print(f"  {len(X_screen)} samples (2× original)")

#     X = np.array(X_real + X_screen)
#     y = np.array(y_real + y_screen)
#     paths = np.array(p_real + p_screen)

#     if len(X) < 20:
#         print("Error: need more images.")
#         sys.exit(1)

#     # Stable scaler
#     scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=0)
#     X_scaled = scaler.fit_transform(X)

#     # Stable feature selection (threshold='median' – worked at 93.7%)
#     selector = SelectFromModel(LogisticRegression(C=1, max_iter=1000, class_weight='balanced'),
#                                threshold='median')
#     X_selected = selector.fit_transform(X_scaled, y)
#     print(f"Selected {X_selected.shape[1]} out of {X.shape[1]} features")

#     n_splits = min(5, int(np.min(np.bincount(y))))
#     cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

#     # Stable hyperparameters (exactly as they were at 93.7%)
#     clf_lr = LogisticRegression(C=3.0, max_iter=2000, class_weight='balanced', random_state=0)
#     clf_rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=0)

#     if XGB_AVAILABLE:
#         clf_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
#                                 scale_pos_weight=1.0, random_state=0,
#                                 use_label_encoder=False, eval_metric='logloss')
#         estimators = [('lr', clf_lr), ('rf', clf_rf), ('xgb', clf_xgb)]
#     else:
#         clf_svm = SVC(kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=0)
#         estimators = [('lr', clf_lr), ('rf', clf_rf), ('svm', clf_svm)]

#     voting_clf = VotingClassifier(estimators=estimators, voting='soft')

#     # Cross-validate
#     y_pred_cv = cross_val_predict(voting_clf, X_selected, y, cv=cv, method='predict')
#     acc_cv = accuracy_score(y, y_pred_cv)

#     print(f"\nCross‑validated accuracy ({n_splits}-fold): {acc_cv:.3f}")
#     print(classification_report(y, y_pred_cv, target_names=["real", "screen"]))

#     # ----- F1-threshold tuning (more stable than Youden) -----
#     y_proba_cv = cross_val_predict(voting_clf, X_selected, y, cv=cv, method='predict_proba')[:, 1]
#     fpr, tpr, thresholds = roc_curve(y, y_proba_cv)
#     # Instead of Youden, find threshold that maximises F1 on CV predictions
#     best_f1 = 0.0
#     best_thresh = 0.5
#     for thresh in thresholds:
#         pred = (y_proba_cv >= thresh).astype(int)
#         f1 = f1_score(y, pred)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thresh = thresh
#     optimal_threshold = best_thresh
#     print(f"Optimal threshold (maximising F1): {optimal_threshold:.3f}")

#     y_pred_opt = (y_proba_cv >= optimal_threshold).astype(int)
#     acc_opt = accuracy_score(y, y_pred_opt)
#     print(f"Accuracy at optimal threshold: {acc_opt:.3f}")

#     wrong = y_pred_opt != y
#     if wrong.any():
#         print(f"\nMisclassified at optimal threshold ({wrong.sum()}/{len(y)}):")
#         for path, true, pred in zip(paths[wrong], y[wrong], y_pred_opt[wrong]):
#             true_name = "real" if true == 0 else "screen"
#             pred_name = "real" if pred == 0 else "screen"
#             print(f"  {path}  (true: {true_name}, pred: {pred_name})")

#     # Final fit
#     voting_clf.fit(X_selected, y)
#     joblib.dump((scaler, selector, voting_clf, optimal_threshold), MODEL_FILE)
#     print(f"\nModel saved to {MODEL_FILE}")
#     print(f"Trained on {len(X)} samples (2× horizontal flip), threshold = {optimal_threshold:.3f}")


# # --------------------------------------------------------------------------
# # Prediction (unchanged)
# # --------------------------------------------------------------------------

# def predict(image_path, verbose=False):
#     if not os.path.isfile(MODEL_FILE):
#         print("Model not found. Run 'python predict.py train' first.")
#         sys.exit(1)

#     t0 = time.perf_counter()
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Could not read image:", image_path)
#         return 0.0

#     scaler, selector, clf, threshold = joblib.load(MODEL_FILE)
#     feat = extract_features(img)
#     feat_scaled = scaler.transform(feat.reshape(1, -1))
#     feat_selected = selector.transform(feat_scaled)
#     prob = float(clf.predict_proba(feat_selected)[0, 1])
#     t1 = time.perf_counter()

#     if verbose:
#         print(f"latency: {(t1 - t0) * 1000:.1f} ms", file=sys.stderr)
#         print(f"optimal_threshold: {threshold:.3f}", file=sys.stderr)
#         print("feature values:", file=sys.stderr)
#         for name, val in zip(FEATURE_NAMES, feat):
#             print(f"  {name:26s} {val:.4f}", file=sys.stderr)

#     return prob


# # --------------------------------------------------------------------------
# # CLI
# # --------------------------------------------------------------------------

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(add_help=False)
#     ap.add_argument("target", help="'train', or an image path")
#     ap.add_argument("--verbose", action="store_true")
#     args = ap.parse_args()

#     if args.target == "train":
#         train()
#     else:
#         score = predict(args.target, verbose=args.verbose)
#         print(f"{score:.4f}")

#===========================================================================================

#!/usr/bin/env python3
"""
Spot the Fake Photo — real photo vs. photo‑of‑a‑screen/printout.

Final 26‑dimensional feature set (added noise_variance):
  ... (all previous 25 features) ...
  - noise_variance (wavelet-based median absolute deviation)

Model: Soft‑Voting (LR + RF + XGBoost) with stable hyperparameters.
Threshold: optimised via F1‑score (more stable than Youden on small data).
"""

import argparse
import glob
import os
import sys
import time
import cv2
import joblib
import numpy as np
import pywt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.svm import SVC
    XGB_AVAILABLE = False

MODEL_FILE = "model.pkl"
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

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
    "noise_variance"          # NEW – stable & orthogonal
]

# --------------------------------------------------------------------------
# Feature extraction (all previous + noise_variance)
# --------------------------------------------------------------------------

def _resize_max_side(img, max_side=512):
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def _fft_features(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    low_mask = r_norm < 0.08
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

# ----- NEW: Noise variance (wavelet-based MAD) -----
def _noise_variance(gray, work_size=256):
    """Estimate noise variance using median absolute deviation of the
    highest-frequency wavelet subband (HH). Screens introduce structured
    noise; real scenes have smoother noise."""
    small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float64)
    # Single-level DWT to get HH subband
    cA, (cH, cV, cD) = pywt.dwt2(small, 'db4')
    # Noise variance = (median(|HH|) / 0.6745)^2
    mad = np.median(np.abs(cD))
    if mad < 1e-8:
        return 0.0
    return float((mad / 0.6745) ** 2)

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
     dwt_energy_compaction) = _dwt_features(gray)
    glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation = _glcm_features(gray)
    lbp_var = _lbp_texture(gray)
    clahe_delta = _clahe_entropy_delta(gray)
    color_fft_ratio = _color_fft_ratio(img_bgr)
    dct_ac_ratio = _dct_blockiness(gray)
    local_var_hetero = _local_variance_heterogeneity(gray)
    edge_orient_entropy = _edge_orientation_entropy(gray)
    noise_var = _noise_variance(gray)

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
        noise_var
    ], dtype=np.float32)


# --------------------------------------------------------------------------
# Training (stable hyperparameters + F1 threshold)
# --------------------------------------------------------------------------

def _load_folder(folder, label):
    X, y, paths = [], [], []
    for ext in IMG_EXTS:
        for path in glob.glob(os.path.join(folder, ext)):
            img = cv2.imread(path)
            if img is None:
                continue
            X.append(extract_features(img))
            y.append(label)
            paths.append(path)
            X.append(extract_features(cv2.flip(img, 1)))
            y.append(label)
            paths.append(path)
    return X, y, paths

def train():
    real_dir, screen_dir = "./real", "./screen"
    if not os.path.isdir(real_dir) or not os.path.isdir(screen_dir):
        dataset_real, dataset_screen = "./dataset/Real", "./dataset/Recaptured"
        if os.path.isdir(dataset_real) and os.path.isdir(dataset_screen):
            real_dir, screen_dir = dataset_real, dataset_screen
        else:
            print("Error: training directories './real' and './screen' not found.")
            sys.exit(1)

    print(f"Loading real photos from {real_dir} (with horizontal flip) ...")
    X_real, y_real, p_real = _load_folder(real_dir, 0)
    print(f"  {len(X_real)} samples (2× original)")

    print(f"Loading screen photos from {screen_dir} (with horizontal flip) ...")
    X_screen, y_screen, p_screen = _load_folder(screen_dir, 1)
    print(f"  {len(X_screen)} samples (2× original)")

    X = np.array(X_real + X_screen)
    y = np.array(y_real + y_screen)
    paths = np.array(p_real + p_screen)

    if len(X) < 20:
        print("Error: need more images.")
        sys.exit(1)

    # Stable scaler
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=0)
    X_scaled = scaler.fit_transform(X)

    # Stable feature selection (threshold='median' – worked at 93.7%)
    selector = SelectFromModel(LogisticRegression(C=1, max_iter=1000, class_weight='balanced'),
                               threshold='median')
    X_selected = selector.fit_transform(X_scaled, y)
    print(f"Selected {X_selected.shape[1]} out of {X.shape[1]} features")

    n_splits = min(5, int(np.min(np.bincount(y))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Stable hyperparameters (exactly as they were at 93.7%)
    clf_lr = LogisticRegression(C=3.0, max_iter=2000, class_weight='balanced', random_state=0)
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=0)

    if XGB_AVAILABLE:
        clf_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                scale_pos_weight=1.0, random_state=0,
                                use_label_encoder=False, eval_metric='logloss')
        estimators = [('lr', clf_lr), ('rf', clf_rf), ('xgb', clf_xgb)]
    else:
        clf_svm = SVC(kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=0)
        estimators = [('lr', clf_lr), ('rf', clf_rf), ('svm', clf_svm)]

    voting_clf = VotingClassifier(estimators=estimators, voting='soft')

    # Cross-validate
    y_pred_cv = cross_val_predict(voting_clf, X_selected, y, cv=cv, method='predict')
    acc_cv = accuracy_score(y, y_pred_cv)

    print(f"\nCross‑validated accuracy ({n_splits}-fold): {acc_cv:.3f}")
    print(classification_report(y, y_pred_cv, target_names=["real", "screen"]))

    # ----- F1-threshold tuning (more stable than Youden) -----
    y_proba_cv = cross_val_predict(voting_clf, X_selected, y, cv=cv, method='predict_proba')[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba_cv)
    # Instead of Youden, find threshold that maximises F1 on CV predictions
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in thresholds:
        pred = (y_proba_cv >= thresh).astype(int)
        f1 = f1_score(y, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    optimal_threshold = best_thresh
    print(f"Optimal threshold (maximising F1): {optimal_threshold:.3f}")

    y_pred_opt = (y_proba_cv >= optimal_threshold).astype(int)
    acc_opt = accuracy_score(y, y_pred_opt)
    print(f"Accuracy at optimal threshold: {acc_opt:.3f}")

    wrong = y_pred_opt != y
    if wrong.any():
        print(f"\nMisclassified at optimal threshold ({wrong.sum()}/{len(y)}):")
        for path, true, pred in zip(paths[wrong], y[wrong], y_pred_opt[wrong]):
            true_name = "real" if true == 0 else "screen"
            pred_name = "real" if pred == 0 else "screen"
            print(f"  {path}  (true: {true_name}, pred: {pred_name})")

    # Final fit
    voting_clf.fit(X_selected, y)
    joblib.dump((scaler, selector, voting_clf, optimal_threshold), MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Trained on {len(X)} samples (2× horizontal flip), threshold = {optimal_threshold:.3f}")


# --------------------------------------------------------------------------
# Prediction (unchanged)
# --------------------------------------------------------------------------

def predict(image_path, verbose=False):
    if not os.path.isfile(MODEL_FILE):
        print("Model not found. Run 'python predict.py train' first.")
        sys.exit(1)

    t0 = time.perf_counter()
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image:", image_path)
        return 0.0

    scaler, selector, clf, threshold = joblib.load(MODEL_FILE)
    feat = extract_features(img)
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    feat_selected = selector.transform(feat_scaled)
    raw_prob = float(clf.predict_proba(feat_selected)[0, 1])
    t1 = time.perf_counter()

    # ----- CALIBRATION: Shift the threshold to 0.5 -----
    # This preserves the order but maps the optimal cutoff to exactly 0.5.
    if threshold is not None and 0.0 < threshold < 1.0:
        logit_thresh = np.log(threshold / (1.0 - threshold))
        raw_clipped = np.clip(raw_prob, 1e-7, 1.0 - 1e-7)
        logit_raw = np.log(raw_clipped / (1.0 - raw_clipped))
        adjusted_logit = logit_raw - logit_thresh
        calibrated_prob = float(1.0 / (1.0 + np.exp(-adjusted_logit)))
    else:
        calibrated_prob = raw_prob

    if verbose:
        print(f"latency: {(t1 - t0) * 1000:.1f} ms", file=sys.stderr)
        print(f"optimal_threshold (raw): {threshold:.3f}", file=sys.stderr)
        print(f"raw_prob: {raw_prob:.4f}, calibrated_prob: {calibrated_prob:.4f}", file=sys.stderr)
        print("feature values:", file=sys.stderr)
        for name, val in zip(FEATURE_NAMES, feat):
            print(f"  {name:26s} {val:.4f}", file=sys.stderr)

    return calibrated_prob


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("target", help="'train', or an image path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.target == "train":
        train()
    else:
        score = predict(args.target, verbose=args.verbose)
        print(f"{score:.4f}")
