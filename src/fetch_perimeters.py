"""Download CAL FIRE FRAP historical fire perimeters (2015-2024) for the AOI
via the official CAL FIRE ArcGIS FeatureServer, paginated, saved as GeoJSON.

Source: "California Historical Fire Perimeters" (egis.CALFIRE)
https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/California_Historic_Fire_Perimeters/FeatureServer
Layer 2: California Fire Perimeters (1950+). Fields include YEAR_, FIRE_NAME,
GIS_ACRES, ALARM_DATE, CAUSE.
"""
import json
import time
import urllib.parse
import urllib.request

BASE = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
        "California_Historic_Fire_Perimeters/FeatureServer/2/query")
BBOX = (-121.0, 32.5, -114.0, 36.0)  # xmin, ymin, xmax, ymax (lon/lat)
WHERE = "YEAR_ >= 2015 AND YEAR_ <= 2024"
OUT = "data/perimeters.geojson"


def fetch_page(offset, limit=1000):
    params = {
        "where": WHERE,
        "geometry": ",".join(map(str, BBOX)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID,YEAR_,FIRE_NAME,GIS_ACRES,ALARM_DATE,CAUSE,AGENCY",
        "orderByFields": "OBJECTID",
        "resultOffset": offset,
        "resultRecordCount": limit,
        "outSR": "4326",
        "f": "geojson",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "wildfire-susceptibility-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    feats = []
    offset = 0
    while True:
        page = fetch_page(offset)
        batch = page.get("features", [])
        feats.extend(batch)
        print(f"offset {offset}: got {len(batch)} (total {len(feats)})", flush=True)
        if len(batch) < 1000:
            break
        offset += 1000
        time.sleep(0.5)
    fc = {"type": "FeatureCollection", "features": feats}
    with open(OUT, "w") as f:
        json.dump(fc, f)
    years = sorted({ft["properties"].get("YEAR_") for ft in feats})
    print(f"saved {len(feats)} perimeters -> {OUT}; years: {years}")


if __name__ == "__main__":
    main()
