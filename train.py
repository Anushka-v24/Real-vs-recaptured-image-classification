"""
train.py

Usage:
    python train.py --real_dir real/ --screen_dir screen/ --out model.pkl

Trains a small, interpretable classifier (logistic regression, ~9 handcrafted
features) on top of features.py. Deliberately avoids a CNN: with only ~100
labeled images, a deep model would overfit and generalize poorly to the
held-out judging set, whereas a linear model on physically-motivated features
is far more robust to that distribution shift.

Uses 5-fold cross-validation to report an honest accuracy estimate (not just
training accuracy) since that's what gets reported in the note.
"""

import argparse
import glob
import os
import pickle

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from features import extract_features, FEATURE_NAMES

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_folder(folder, label):
    X, y, paths = [], [], []
    for ext in IMG_EXTS:
        for path in glob.glob(os.path.join(folder, ext)):
            img = cv2.imread(path)
            if img is None:
                print(f"  [skip] could not read {path}")
                continue
            feat = extract_features(img)
            X.append(feat)
            y.append(label)
            paths.append(path)
    return X, y, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="real")
    ap.add_argument("--screen_dir", default="screen")
    ap.add_argument("--out", default="model.pkl")
    args = ap.parse_args()

    print(f"Loading real photos from {args.real_dir} ...")
    X_real, y_real, p_real = load_folder(args.real_dir, 0)
    print(f"  {len(X_real)} images")

    print(f"Loading screen photos from {args.screen_dir} ...")
    X_screen, y_screen, p_screen = load_folder(args.screen_dir, 1)
    print(f"  {len(X_screen)} images")

    X = np.array(X_real + X_screen, dtype=np.float32)
    y = np.array(y_real + y_screen, dtype=np.int32)

    if len(X) < 10:
        raise SystemExit("Not enough images found. Check --real_dir / --screen_dir.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    # Honest accuracy: cross-validated, not fit-then-score-on-same-data.
    n_splits = min(5, np.min(np.bincount(y)))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    y_pred_cv = cross_val_predict(clf, X_scaled, y, cv=cv)

    acc = accuracy_score(y, y_pred_cv)
    print(f"\nCross-validated accuracy ({n_splits}-fold): {acc:.3f}")
    print(classification_report(y, y_pred_cv, target_names=["real", "screen"]))

    print("Feature importances (standardized logistic regression coefficients):")
    clf.fit(X_scaled, y)
    for name, coef in sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:18s} {coef:+.3f}")

    # Final model fit on all data for deployment.
    with open(args.out, "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf, "feature_names": FEATURE_NAMES}, f)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
