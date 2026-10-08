"""Download PRISM 30-year normals (800m) for tmax (July+August) and annual
precipitation, crop to the AOI, and write analysis-ready GeoTIFFs.

Source: PRISM Climate Group, Oregon State University — 30-year normals
1991-2020 ("30s" = 30 arc-seconds, ~800 m). New data portal:
https://data.prism.oregonstate.edu/normals/us/800m/{tmax,ppt}/monthly/
Files: prism_tmax_us_30s_202007_avg_30y.zip etc. Each zip holds a .bil +
.hdr pair covering CONUS.

Outputs:
  data/prism_tmax_summer.tif  (mean of July & August 30-yr normal tmax, deg C)
  data/prism_ppt_annual.tif   (annual 30-yr normal precipitation, mm)
"""
import os
import zipfile
import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds

BASE = "https://data.prism.oregonstate.edu/normals/us/800m"
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -121.0, 32.5, -114.0, 36.0
TMP = "data/_dl_prism"  # workspace-local: /tmp is a tiny tmpfs


def dl(name, subdir):
    os.makedirs(TMP, exist_ok=True)
    url = f"{BASE}/{subdir}/monthly/{name}"
    dest = os.path.join(TMP, name)
    if not os.path.exists(dest):
        print("downloading", name, flush=True)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        open(dest, "wb").write(r.content)
    return dest


def read_cropped(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        tif = [n for n in z.namelist() if n.endswith(".tif")][0]
        z.extract(tif, TMP)
    src = rasterio.open(os.path.join(TMP, tif))
    win = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
    arr = src.read(1, window=win).astype(np.float32)
    out_tr = src.window_transform(win)
    nodata = src.nodata
    src.close()
    arr[arr == nodata] = np.nan
    return arr, out_tr


def write(path, arr, tr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=tr, nodata=np.nan,
                       compress="deflate") as dst:
        dst.write(arr, 1)
    print("wrote", path, arr.shape)


def main():
    t7, tr = read_cropped(dl("prism_tmax_us_30s_202007_avg_30y.zip", "tmax"))
    t8, _ = read_cropped(dl("prism_tmax_us_30s_202008_avg_30y.zip", "tmax"))
    summer = np.nanmean(np.stack([t7, t8]), axis=0).astype(np.float32)
    write("data/prism_tmax_summer.tif", summer, tr)
    ppt, tr2 = read_cropped(dl("prism_ppt_us_30s_2020_avg_30y.zip", "ppt"))
    write("data/prism_ppt_annual.tif", ppt.astype(np.float32), tr2)


if __name__ == "__main__":
    main()
