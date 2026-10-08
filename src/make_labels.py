"""Build labeled samples for the wildfire susceptibility model.

Positives: grid cells inside CAL FIRE FRAP fire perimeters (2015-2024).
Negatives: 50% "hard" negatives sampled 1-5 km outside perimeters (same
climate/terrain regime, did not burn) + 50% background negatives sampled
>5 km from any perimeter. Water cells excluded from negatives. This design
is documented in the README; it avoids the trivial-negative AUC inflation
trap while keeping the task realistic.

Requires: data/perimeters.geojson, data/landcover.tif
Writes: data/samples.csv, data/label_provenance.csv
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt

LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -121.0, 32.5, -114.0, 36.0
RES = 0.002
SEED = 42
N_POS_CAP = 30000
N_NEG = 30000
NEAR_MIN_M, NEAR_MAX_M = 1000, 5000
CELL_M = 200  # approx meters per cell at this latitude

rng = np.random.default_rng(SEED)


def main():
    W = int(round((LON_MAX - LON_MIN) / RES))
    H = int(round((LAT_MAX - LAT_MIN) / RES))
    tr = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W, H)

    print("rasterizing perimeters...", flush=True)
    gdf = gpd.read_file("data/perimeters.geojson")
    burned = features.rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=(H, W), transform=tr, fill=0,
        dtype="uint8", all_touched=True)
    n_burned = int(burned.sum())
    print(f"burned cells: {n_burned} ({100*n_burned/burned.size:.2f}%)", flush=True)

    print("distance transform...", flush=True)
    dist_px = distance_transform_edt(1 - burned)
    dist_m = dist_px * CELL_M

    with rasterio.open("data/landcover.tif") as ds:
        lc = ds.read(1)
    water = (lc == 80)

    # positives
    ys, xs = np.nonzero(burned == 1)
    idx = rng.choice(len(xs), size=min(N_POS_CAP, len(xs)), replace=False)
    pos = np.stack([xs[idx], ys[idx]], axis=1)

    # negatives: near ring 1-5 km
    near_ok = (dist_m >= NEAR_MIN_M) & (dist_m <= NEAR_MAX_M) & (~water)
    ys_n, xs_n = np.nonzero(near_ok)
    n_near = N_NEG // 2
    idx_n = rng.choice(len(xs_n), size=min(n_near, len(xs_n)), replace=False)
    neg_near = np.stack([xs_n[idx_n], ys_n[idx_n]], axis=1)

    # negatives: background >5 km
    bg_ok = (dist_m > NEAR_MAX_M) & (~water)
    ys_b, xs_b = np.nonzero(bg_ok)
    n_bg = N_NEG - len(neg_near)
    idx_b = rng.choice(len(xs_b), size=min(n_bg, len(xs_b)), replace=False)
    neg_bg = np.stack([xs_b[idx_b], ys_b[idx_b]], axis=1)

    neg = np.vstack([neg_near, neg_bg])
    print(f"positives: {len(pos)}, negatives: {len(neg)} "
          f"(near {len(neg_near)}, background {len(neg_bg)})", flush=True)

    pts = np.vstack([pos, neg])
    labels = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    lons = LON_MIN + (pts[:, 0] + 0.5) * RES
    lats = LAT_MAX - (pts[:, 1] + 0.5) * RES
    df = pd.DataFrame({"lon": lons, "lat": lats, "label": labels,
                       "dist_to_fire_m": (dist_px[pts[:, 1], pts[:, 0]] * CELL_M).round(0)})
    df.to_csv("data/samples.csv", index=False)
    print("wrote data/samples.csv", df.shape)

    prov = pd.DataFrame([{
        "label_type": "positive (burned)",
        "source": "CAL FIRE FRAP 'California Historical Fire Perimeters' "
                  "(egis.CALFIRE), ArcGIS FeatureServer layer 2, "
                  "query YEAR_ 2015-2024 within AOI bbox; accessed 2026-10-07. "
                  f"{len(gdf)} perimeters, rasterized at 0.002 deg (~200 m), all_touched=True.",
        "n": len(pos)}, {
        "label_type": "negative (unburned)",
        "source": "Sampled from grid cells outside all perimeters: 50% from "
                  "1-5 km ring outside perimeters ('hard' negatives), 50% from "
                  ">5 km background; water cells excluded. Seed 42.",
        "n": len(neg)}])
    prov.to_csv("data/label_provenance.csv", index=False)
    print("wrote data/label_provenance.csv")


if __name__ == "__main__":
    main()
