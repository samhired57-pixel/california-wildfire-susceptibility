"""Derive model features on the 0.002 deg (~200 m) analysis grid.

Inputs: data/dem.tif, data/landcover.tif, data/prism_tmax_summer.tif,
        data/prism_ppt_annual.tif
Outputs: data/derived/*.tif (float32) + data/derived/feature_list.txt

Features (14):
  elevation_m, slope_deg, southness (1=south-facing), tpi_1km,
  tmax_summer_c, ppt_annual_mm, dist_builtup_m,
  fuel_forest, fuel_shrub, fuel_grass, fuel_crop, fuel_built, fuel_bare,
  fuel_nonburn  (one-hot fuel groups from ESA WorldCover)

Fuel grouping: 10->forest; 20->shrub; 30->grass; 40->crop; 50->built;
60->bare; 70,80,90,95,100,0->nonburn (snow/water/wetland/mangrove/moss/nodata).
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter, distance_transform_edt

LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -121.0, 32.5, -114.0, 36.0
RES = 0.002
CELL_M = 200

FUEL = {"forest": [10], "shrub": [20], "grass": [30], "crop": [40],
        "built": [50], "bare": [60], "nonburn": [0, 70, 80, 90, 95, 100]}


def grid():
    W = int(round((LON_MAX - LON_MIN) / RES))
    H = int(round((LAT_MAX - LAT_MIN) / RES))
    return H, W, from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W, H)


def resample_to_grid(path, resampling):
    H, W, tr = grid()
    with rasterio.open(path) as src:
        arr = np.empty((H, W), dtype=np.float32)
        reproject(rasterio.band(src, 1), arr, src_transform=src.transform,
                  src_crs=src.crs, dst_transform=tr, dst_crs="EPSG:4326",
                  resampling=resampling)
    return arr


def write(name, arr):
    H, W, tr = grid()
    p = f"data/derived/{name}.tif"
    with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="float32", crs="EPSG:4326", transform=tr,
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(arr.astype(np.float32), 1)
    print("wrote", p)


def main():
    import os
    os.makedirs("data/derived", exist_ok=True)
    H, W, tr = grid()
    feats = {}

    print("DEM...", flush=True)
    elev = resample_to_grid("data/dem.tif", Resampling.bilinear)
    elev[elev < -500] = np.nan
    feats["elevation_m"] = elev

    print("slope/aspect...", flush=True)
    dy_m = RES * 111320.0
    dx_m = RES * 111320.0 * np.cos(np.radians(34.0))
    gy, gx = np.gradient(elev, dy_m, dx_m)  # gy: dE/dy(north), gx: dE/dx(east)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    feats["slope_deg"] = slope.astype(np.float32)
    aspect = np.arctan2(gx, -gy)  # clockwise from north, radians
    feats["southness"] = np.cos(aspect - np.pi).astype(np.float32)

    print("TPI...", flush=True)
    k = int(round(1000 / CELL_M))  # 1 km window
    k = k + 1 if k % 2 == 0 else k
    mean1k = uniform_filter(np.nan_to_num(elev, nan=np.nanmean(elev)),
                            size=k, mode="nearest")
    feats["tpi_1km"] = (elev - mean1k).astype(np.float32)

    print("climate...", flush=True)
    feats["tmax_summer_c"] = resample_to_grid("data/prism_tmax_summer.tif",
                                              Resampling.bilinear)
    feats["ppt_annual_mm"] = resample_to_grid("data/prism_ppt_annual.tif",
                                              Resampling.bilinear)

    print("landcover...", flush=True)
    with rasterio.open("data/landcover.tif") as ds:
        lc = ds.read(1)
    built = (lc == 50)
    d = distance_transform_edt(~built) * CELL_M
    feats["dist_builtup_m"] = np.minimum(d, 20000).astype(np.float32)
    for name, codes in FUEL.items():
        feats[f"fuel_{name}"] = np.isin(lc, codes).astype(np.float32)

    for name, arr in feats.items():
        write(name, arr)
    with open("data/derived/feature_list.txt", "w") as f:
        f.write("\n".join(feats.keys()))
    print(f"{len(feats)} features written")


if __name__ == "__main__":
    main()
