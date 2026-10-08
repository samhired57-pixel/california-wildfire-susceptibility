"""Fetch ESA WorldCover 2021 (10 m) land cover for the AOI.

Downloads the four 3x3-degree tiles with `requests` (GDAL's /vsicurl/
cannot verify the sandbox egress TLS cert, so tiles are fetched to /tmp
first, then read locally), crops to the AOI, downsamples to ~200 m with
nearest-neighbor, writes data/landcover.tif.

Source: ESA WorldCover 2021 v200, open data on AWS
(registry.opendata.aws/esa-worldcover).
s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_<T>_Map.tif

Classes: 10 tree, 20 shrubland, 30 grassland, 40 cropland, 50 built-up,
60 bare/sparse vegetation, 70 snow/ice, 80 permanent water, 90 herbaceous
wetland, 95 mangroves, 100 moss/lichen.
"""
import os
import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -121.0, 32.5, -114.0, 36.0
TILES = ["N30W120", "N30W117", "N33W120", "N33W117"]
TMP = "data/_dl_worldcover"  # workspace-local: /tmp is a tiny tmpfs
OUT = "data/landcover.tif"
RES = 0.002  # ~200 m analysis grid


def dl(tile):
    os.makedirs(TMP, exist_ok=True)
    url = (f"https://esa-worldcover.s3.amazonaws.com/v200/2021/map/"
           f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
    dest = os.path.join(TMP, f"{tile}.tif")
    if not os.path.exists(dest):
        print("downloading", tile, flush=True)
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        open(dest, "wb").write(r.content)
        print("  %.1f MB" % (os.path.getsize(dest) / 1e6), flush=True)
    return dest


def main():
    W = int(round((LON_MAX - LON_MIN) / RES))
    H = int(round((LAT_MAX - LAT_MIN) / RES))
    canvas = np.zeros((H, W), dtype=np.uint8)
    filled = np.zeros((H, W), dtype=bool)
    for t in TILES:
        path = dl(t)
        with rasterio.open(path) as src:
            win = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
            win = win.round_offsets().round_shape()
            scale = RES / abs(src.transform.a)
            oh = max(1, int(round(win.height / scale)))
            ow = max(1, int(round(win.width / scale)))
            arr = src.read(1, window=win, out_shape=(oh, ow),
                           resampling=Resampling.nearest)
            lon_w = src.transform.c + win.col_off * src.transform.a
            lat_n = src.transform.f + win.row_off * src.transform.e
            c0 = int(round((lon_w - LON_MIN) / RES))
            r0 = int(round((LAT_MAX - lat_n) / RES))
            rs, re_ = max(r0, 0), min(r0 + oh, H)
            cs, ce = max(c0, 0), min(c0 + ow, W)
            if rs < re_ and cs < ce:
                canvas[rs:re_, cs:ce] = arr[rs - r0:re_ - r0, cs - c0:ce - c0]
                filled[rs:re_, cs:ce] = True
    print("coverage: %.1f%%" % (100 * filled.mean()), flush=True)
    tr = rasterio.transform.from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W, H)
    with rasterio.open(OUT, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=tr,
                       nodata=0, compress="deflate") as dst:
        dst.write(canvas, 1)
        vals, counts = np.unique(canvas[filled], return_counts=True)
        print("class histogram:", dict(zip(vals.tolist(), counts.tolist())))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
