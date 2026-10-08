"""Fetch SRTM elevation for the AOI via Mapzen Terrarium tiles (z10),
decode per the Terrarium spec, mosaic into data/dem.tif (EPSG:4326).

Terrarium encoding: elevation_m = R*256 + G + B/256 - 32768
Tiles: https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
"""
import math
import io
import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.merge import merge

Z = 10
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -121.0, 32.5, -114.0, 36.0
OUT = "data/dem.tif"


def deg2num(lon, lat, z):
    n = 2 ** z
    xt = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    yt = int((1.0 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return xt, yt


def num2deg(xt, yt, z):
    n = 2 ** z
    lon = xt / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * yt / n)))
    lat = math.degrees(lat_r)
    return lon, lat


def fetch_tile(z, x, y):
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def decode(png_bytes):
    import struct, zlib
    # minimal PNG decode via rasterio in-memory read
    with rasterio.open(io.BytesIO(png_bytes)) as ds:
        arr = ds.read()  # (3, 256, 256) uint8
    r, g, b = arr[0].astype(np.float32), arr[1].astype(np.float32), arr[2].astype(np.float32)
    return r * 256.0 + g + b / 256.0 - 32768.0


def main():
    x0, y0 = deg2num(LON_MIN, LAT_MAX, Z)   # NW corner
    x1, y1 = deg2num(LON_MAX, LAT_MIN, Z)   # SE corner
    print(f"tiles x {x0}..{x1}, y {y0}..{y1} = {(x1-x0+1)*(y1-y0+1)} tiles", flush=True)
    datasets = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            png = fetch_tile(Z, x, y)
            elev = decode(png)
            lon_w, lat_n = num2deg(x, y, Z)
            lon_e, lat_s = num2deg(x + 1, y + 1, Z)
            tr = from_bounds(lon_w, lat_s, lon_e, lat_n, 256, 256)
            mem = rasterio.io.MemoryFile()
            ds = mem.open(driver="GTiff", height=256, width=256, count=1,
                          dtype="float32", crs="EPSG:4326", transform=tr,
                          nodata=-9999)
            ds.write(elev.astype(np.float32), 1)
            datasets.append(ds)
    mosaic, out_tr = merge(datasets, nodata=-9999)
    print("mosaic shape:", mosaic.shape, flush=True)
    with rasterio.open(OUT, "w", driver="GTiff", height=mosaic.shape[1],
                       width=mosaic.shape[2], count=1, dtype="float32",
                       crs="EPSG:4326", transform=out_tr, nodata=-9999,
                       compress="deflate") as dst:
        dst.write(mosaic[0].astype(np.float32), 1)
    for ds in datasets:
        ds.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
