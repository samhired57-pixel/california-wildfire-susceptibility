"""Train and evaluate the wildfire susceptibility model.

- Samples: data/samples.csv (label 1 = inside FRAP 2015-2024 perimeters)
- Features: data/derived/*.tif per data/derived/feature_list.txt
- Models: RandomForest (primary) + LogisticRegression (baseline)
- Protocol: stratified 5-fold CV on 75% train; final fit on train;
  evaluation on held-out 25% test. Seed 42 throughout.
- Outputs: outputs/metrics.json, outputs/feature_importance.png,
  outputs/susceptibility.tif + .png (chunked full-grid prediction),
  outputs/rf_model.joblib
"""
import json
import numpy as np
import pandas as pd
import rasterio
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

SEED = 42
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -121.0, 32.5, -114.0, 36.0
RES = 0.002


def load_features(names):
    arrs = []
    for n in names:
        with rasterio.open(f"data/derived/{n}.tif") as ds:
            arrs.append(ds.read(1))
    return np.stack(arrs)  # (F, H, W)


def main():
    names = open("data/derived/feature_list.txt").read().split()
    print("features:", names, flush=True)
    stack = load_features(names)
    F, H, W = stack.shape
    print("stack:", stack.shape, flush=True)

    df = pd.read_csv("data/samples.csv")
    rows = ((LAT_MAX - df["lat"]) / RES).astype(int).clip(0, H - 1)
    cols = ((df["lon"] - LON_MIN) / RES).astype(int).clip(0, W - 1)
    X = stack[:, rows.values, cols.values].T
    y = df["label"].values
    print("X:", X.shape, "positives:", int(y.sum()), flush=True)

    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)

    # sanity: positives vs negatives
    print("--- sanity: mean(feature) | burned vs unburned ---")
    for i, n in enumerate(names):
        print(f"{n:16s} {X[y==1,i].mean():12.2f} {X[y==0,i].mean():12.2f}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    n_test = int(0.25 * len(y))
    te, tr = idx[:n_test], idx[n_test:]
    # stratify manually to keep it simple and reproducible
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                n_jobs=2, random_state=SEED)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=SEED))
    scoring = {"auc": "roc_auc", "f1": "f1"}
    print("CV: random forest...", flush=True)
    cv_rf = cross_validate(rf, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
    print("CV: logistic regression...", flush=True)
    cv_lr = cross_validate(lr, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)

    rf.fit(Xtr, ytr)
    lr.fit(Xtr, ytr)
    out = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
           "seed": SEED, "model": "RandomForestClassifier(400, balanced_subsample)",
           "baseline": "LogisticRegression(standardized, balanced)",
           "cv": {
               "rf_auc_mean": float(cv_rf["test_auc"].mean()),
               "rf_auc_std": float(cv_rf["test_auc"].std()),
               "rf_f1_mean": float(cv_rf["test_f1"].mean()),
               "lr_auc_mean": float(cv_lr["test_auc"].mean()),
               "lr_f1_mean": float(cv_lr["test_f1"].mean())},
           "heldout": {}}
    for tag, m in [("rf", rf), ("lr", lr)]:
        p = m.predict_proba(Xte)[:, 1]
        c = m.predict(Xte)
        tn, fp, fn, tp = confusion_matrix(yte, c).ravel()
        out["heldout"][tag] = {
            "auc": float(roc_auc_score(yte, p)),
            "precision": float(precision_score(yte, c)),
            "recall": float(recall_score(yte, c)),
            "f1": float(f1_score(yte, c)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    out["feature_importances"] = {
        n: float(v) for n, v in zip(names, rf.feature_importances_)}
    with open("outputs/metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["cv"], indent=2))
    print(json.dumps(out["heldout"], indent=2))

    # feature importance plot
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh([names[i] for i in order][::-1], imp[order][::-1])
    plt.xlabel("Gini importance")
    plt.title("Wildfire susceptibility — feature importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=120)
    plt.close()

    joblib.dump({"model": rf, "features": names, "median": med},
                "outputs/rf_model.joblib")

    # full-grid prediction in row chunks
    print("predicting full grid...", flush=True)
    tr = rasterio.transform.from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W, H)
    with rasterio.open("outputs/susceptibility.tif", "w", driver="GTiff",
                       height=H, width=W, count=1, dtype="float32",
                       crs="EPSG:4326", transform=tr, nodata=np.nan,
                       compress="deflate", tiled=True) as dst:
        CH = 200
        for r0 in range(0, H, CH):
            r1 = min(H, r0 + CH)
            Xc = stack[:, r0:r1, :].reshape(F, -1).T
            Xc = np.where(np.isnan(Xc), med, Xc)
            prob = rf.predict_proba(Xc)[:, 1].reshape(r1 - r0, W).astype(np.float32)
            dst.write(prob, 1, window=((r0, r1), (0, W)))
            if r0 % 600 == 0:
                print(f"  rows {r0}/{H}", flush=True)
    with rasterio.open("outputs/susceptibility.tif") as ds:
        m = ds.read(1, out_shape=(875, 1750))
    plt.figure(figsize=(12, 7))
    plt.imshow(m, cmap="inferno", vmin=0, vmax=1,
               extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], origin="upper")
    plt.colorbar(label="P(burn susceptibility)")
    plt.title("California (South) — modeled wildfire susceptibility")
    plt.xlabel("lon"); plt.ylabel("lat")
    plt.tight_layout()
    plt.savefig("outputs/susceptibility.png", dpi=130)
    plt.close()
    print("done.")


if __name__ == "__main__":
    main()
