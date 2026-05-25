# ============================================================
# HREF | R2 Meso-Ensemble Probability Product
# Organized Severe Ingredients
# Fill: CAPE > 1500 J/kg probability
# Contours: 0-6 km shear > 20.6 m/s probability
# Uploads runs.json and PNGs to runs/mesoensprob/href/organized_severe/
# ============================================================

import os
import re
import json
import zipfile
import requests
import boto3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import geopandas as gpd
from shapely.ops import unary_union
from shapely.prepared import prep

from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta, timezone
from botocore.config import Config


# ============================================================
# PATHS / ASSETS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSET_DIR = os.path.join(BASE_DIR, "assets")
COUNTY_SHP = os.path.join(ASSET_DIR, "cb_2018_us_county_500k.shp")
STATE_SHP = os.path.join(ASSET_DIR, "cb_2018_us_state_500k.shp")
LBF_CWA_SHP = os.path.join(ASSET_DIR, "c_18mr25.shp")
LOGO_PATH = os.path.join(ASSET_DIR, "NOAANWSLogos.png")

zip_path = os.path.join(ASSET_DIR, "c_18mr25.zip")
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(ASSET_DIR)

DATA_DIR = os.path.join(BASE_DIR, "href_organized_severe_subsets")

SECTION_KEY = "mesoensprob"
MODEL_KEY = "href"
PRODUCT_KEY = "organized_severe"

R2_PRODUCT_PATH = f"runs/{SECTION_KEY}/{MODEL_KEY}/{PRODUCT_KEY}"

OUTDIR_BASE = os.path.join(
    "site",
    "runs",
    SECTION_KEY,
    MODEL_KEY,
    PRODUCT_KEY
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTDIR_BASE, exist_ok=True)


# ============================================================
# R2 SETUP
# ============================================================

BUCKET = os.environ["AWS_BUCKET"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_REGION"],
)


def upload_to_r2(local_file, remote_key, content_type="image/png"):
    s3.upload_file(
        local_file,
        BUCKET,
        remote_key,
        ExtraArgs={"ContentType": content_type}
    )
    print("Uploaded to R2:", remote_key)


# ============================================================
# DOMAIN CONFIG
# ============================================================

DOMAINS = {
    "lbf": {
        "label": "LBF",
        "extent": [-103.8, -97.0, 40.0, 43.4],
        "title_size": 12,
        "subtitle_size": 11,
    },

    "regional": {
        "label": "Default",
        "extent": [-107.5, -93.0, 38.5, 44.2],
        "title_size": 13,
        "subtitle_size": 11,
    },

    "central_plains": {
        "label": "Central Plains",
        "extent": [-107.5, -91.0, 34.5, 45.2],
        "title_size": 13,
        "subtitle_size": 11,
    },
}


# ============================================================
# DYNAMIC SPC SEVERE DOMAIN
# ============================================================

SPC_DAY1_CAT_URL = (
    "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/"
    "SPC_wx_outlks/MapServer/1/query"
)

SPC_RISK_ORDER = {
    "TSTM": 1,
    "MRGL": 2,
    "SLGT": 3,
    "ENH": 4,
    "MDT": 5,
    "HIGH": 6,
}

MIN_SPC_RISK = "SLGT"
SEVERE_DOMAIN_WIDTH = 14.0
SEVERE_DOMAIN_HEIGHT = 10.0


def fetch_spc_day1_geojson():
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "returnGeometry": "true",
        "outSR": "4326",
    }

    r = requests.get(SPC_DAY1_CAT_URL, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()

    if "features" not in data or len(data["features"]) == 0:
        raise RuntimeError("SPC query returned no features.")

    return gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")


def add_spc_severe_domain():
    try:
        gdf = fetch_spc_day1_geojson().to_crs(epsg=4326)

        risk_col = None

        for col in gdf.columns:
            vals = gdf[col].astype(str).str.upper()
            if vals.isin(SPC_RISK_ORDER.keys()).any():
                risk_col = col
                break

        if risk_col is None:
            print("SPC severe domain skipped: could not find risk category column.")
            return

        gdf["risk"] = gdf[risk_col].astype(str).str.upper()
        gdf["risk_rank"] = gdf["risk"].map(SPC_RISK_ORDER)

        severe = gdf[gdf["risk_rank"] >= SPC_RISK_ORDER[MIN_SPC_RISK]].copy()

        if severe.empty:
            print("SPC severe domain skipped: no SLGT+ risk found.")
            return

        highest_rank = severe["risk_rank"].max()
        highest = severe[severe["risk_rank"] == highest_rank].copy()

        highest_proj = highest.to_crs(epsg=5070)
        highest["_area"] = highest_proj.geometry.area.values

        main_poly = highest.loc[highest["_area"].idxmax()]
        highest_label = main_poly["risk"]

        main_gdf = gpd.GeoDataFrame(
            [main_poly],
            geometry="geometry",
            crs="EPSG:4326"
        )

        centroid_proj = main_gdf.to_crs(epsg=5070).geometry.centroid

        centroid_ll = gpd.GeoSeries(
            centroid_proj,
            crs="EPSG:5070"
        ).to_crs(epsg=4326).iloc[0]

        center_lon = centroid_ll.x
        center_lat = centroid_ll.y

        extent = [
            center_lon - SEVERE_DOMAIN_WIDTH / 2,
            center_lon + SEVERE_DOMAIN_WIDTH / 2,
            center_lat - SEVERE_DOMAIN_HEIGHT / 2,
            center_lat + SEVERE_DOMAIN_HEIGHT / 2,
        ]

        DOMAINS["spc_severe"] = {
            "label": f"SPC {highest_label} Risk",
            "extent": extent,
            "title_size": 11,
            "subtitle_size": 11,
        }

        print(f"Added SPC severe domain: {highest_label}")
        print(f"SPC severe extent: {extent}")

    except Exception as e:
        print(f"SPC severe domain skipped due to error: {e}")


add_spc_severe_domain()


# ============================================================
# SETTINGS
# ============================================================

MODEL_LABEL = "HREF"

VALID_HREF_CYCLES = [0, 6, 12, 18]

START_FHR = 1
MAX_FHR = 48

CYCLE_DELAY_MINUTES = 120

PROB_LEVELS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
PROB_TICKS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

SHEAR_CONTOURS = [10, 20, 30, 40, 50, 60, 70]


# ============================================================
# BASIC HELPERS
# ============================================================

def to_lon180(lon):
    return ((np.asarray(lon) + 180) % 360) - 180


def get_lat_lon(da):
    if "latitude" in da.coords and "longitude" in da.coords:
        lat = np.asarray(da.latitude.values)
        lon = to_lon180(da.longitude.values)
    elif "lat" in da.coords and "lon" in da.coords:
        lat = np.asarray(da.lat.values)
        lon = to_lon180(da.lon.values)
    else:
        raise RuntimeError("Could not find latitude/longitude coordinates.")

    lat = np.squeeze(lat)
    lon = np.squeeze(lon)

    if lat.ndim != 2 or lon.ndim != 2:
        raise RuntimeError(f"Lat/lon not 2D. lat={lat.shape}, lon={lon.shape}")

    return lat, lon


def ensure_2d_field(da, label):
    arr = np.asarray(da.values, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise RuntimeError(
            f"{label} is not 2D after squeeze. "
            f"Shape={arr.shape}, dims={getattr(da, 'dims', None)}"
        )

    return arr


def normalize_probability(arr):
    arr = np.asarray(arr, dtype=float)

    if np.nanmax(arr) <= 1.01:
        arr = arr * 100.0

    return arr


# ============================================================
# SHAPEFILE HELPERS
# ============================================================

def add_shapefile_outline(ax, shp_path, edgecolor="k", linewidth=1.2, zorder=6):
    if not os.path.exists(shp_path):
        print("Missing shapefile:", shp_path)
        return

    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    gdf = gdf.cx[LON_MIN - 1:LON_MAX + 1, LAT_MIN - 1:LAT_MAX + 1]

    ax.add_geometries(
        gdf.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )


def get_lbf_cwa_geom(cwa_shp_path):
    if not os.path.exists(cwa_shp_path):
        print("Missing LBF CWA shapefile:", cwa_shp_path)
        return None

    reader = shpreader.Reader(cwa_shp_path)
    recs = list(reader.records())

    geoms = [
        r.geometry for r in recs
        if str(r.attributes.get("CWA", "")).upper() == "LBF"
        or str(r.attributes.get("WFO", "")).upper() == "LBF"
    ]

    if not geoms:
        geoms = [r.geometry for r in recs]

    return unary_union(geoms)


def add_counties_clipped_to_cwa(ax, counties_shp_path, cwa_geom, lw=1.0, color="black", zorder=6):
    if cwa_geom is None or not os.path.exists(counties_shp_path):
        return

    reader = shpreader.Reader(counties_shp_path)
    cwa_p = prep(cwa_geom)
    clipped = []

    for r in reader.records():
        g = r.geometry
        if cwa_p.intersects(g):
            inter = g.intersection(cwa_geom)
            if not inter.is_empty:
                clipped.append(inter)

    ax.add_geometries(
        clipped,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        zorder=zorder,
    )


# ============================================================
# HREF URL / IDX BYTE-RANGE SUBSETTING
# ============================================================

def href_grib_url(init_dt, fhr):
    ymd = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")

    fname = f"href.t{hh}z.conus.prob.f{fhr:02d}.grib2"

    return (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/"
        f"href.{ymd}/ensprod/{fname}"
    )


def url_exists(url, timeout=15):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def find_latest_available_href_cycle(max_back_hours=96):
    now = datetime.now(timezone.utc) - timedelta(minutes=CYCLE_DELAY_MINUTES)

    for back in range(max_back_hours + 1):
        dt = now - timedelta(hours=back)

        if dt.hour not in VALID_HREF_CYCLES:
            continue

        dt = dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)

        test_url = href_grib_url(dt, 1) + ".idx"

        if url_exists(test_url):
            print(f"Latest {MODEL_LABEL} cycle found: {dt:%Y%m%d} {dt:%HZ}")
            print("Matched IDX:", test_url)
            return dt

    raise RuntimeError(f"Could not find recent {MODEL_LABEL} cycle.")


def read_idx(idx_url):
    r = requests.get(idx_url, timeout=30)
    r.raise_for_status()
    return r.text.strip().splitlines()


def parse_idx_lines(lines):
    parsed = []

    for i, line in enumerate(lines):
        parts = line.split(":")

        if len(parts) < 5:
            continue

        try:
            msg_num = int(parts[0])
            start_byte = int(parts[1])
        except Exception:
            continue

        parsed.append({
            "i": i,
            "line": line,
            "msg_num": msg_num,
            "start": start_byte,
        })

    for j in range(len(parsed)):
        if j < len(parsed) - 1:
            parsed[j]["end"] = parsed[j + 1]["start"] - 1
        else:
            parsed[j]["end"] = None

    return parsed


def find_idx_match(parsed, all_terms, label):
    all_terms_lower = [t.lower() for t in all_terms]
    matches = []

    for item in parsed:
        line_lower = item["line"].lower()
        if all(term in line_lower for term in all_terms_lower):
            matches.append(item)

    if not matches:
        sample = "\n".join([p["line"] for p in parsed[:200]])
        raise RuntimeError(
            f"Could not find {label} in IDX using terms {all_terms}.\n"
            f"First 200 IDX lines:\n{sample}"
        )

    match = matches[0]

    print(f"Matched {label}:")
    print(match["line"])

    return match


def download_byte_range(grib_url, start, end, outpath):
    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        print("Using cached subset:", outpath)
        return outpath

    headers = {}

    if end is None:
        headers["Range"] = f"bytes={start}-"
    else:
        headers["Range"] = f"bytes={start}-{end}"

    print("Downloading byte range:", headers["Range"])

    r = requests.get(grib_url, headers=headers, stream=True, timeout=120)
    r.raise_for_status()

    with open(outpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return outpath


def open_subset_grib(path, label):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    )

    if len(ds.data_vars) == 0:
        raise RuntimeError(f"No variables found in subset for {label}")

    var = list(ds.data_vars)[0]
    da = ds[var]

    print(f"Opened {label}: var={var}, dims={da.dims}, shape={da.shape}")

    return da


def href_idx_field(init_dt, fhr, term_sets, label):
    grib_url = href_grib_url(init_dt, fhr)
    idx_url = grib_url + ".idx"

    lines = read_idx(idx_url)
    parsed = parse_idx_lines(lines)

    last_error = None

    for terms in term_sets:
        try:
            match = find_idx_match(parsed, terms, label)

            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")

            outname = (
                f"href_prob_{init_dt:%Y%m%d_%H}z_f{fhr:02d}_"
                f"{safe_label}_{match['msg_num']}.grib2"
            )

            outpath = os.path.join(DATA_DIR, outname)

            download_byte_range(
                grib_url,
                match["start"],
                match["end"],
                outpath
            )

            return open_subset_grib(outpath, label)

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not open {label}. Last error: {last_error}")


# ============================================================
# SPATIAL HELPERS
# ============================================================

def subset_2d(lat, lon, *fields):
    mask = (
        np.isfinite(lat) &
        np.isfinite(lon) &
        (lon >= LON_MIN) &
        (lon <= LON_MAX) &
        (lat >= LAT_MIN) &
        (lat <= LAT_MAX)
    )

    if not np.any(mask):
        raise RuntimeError("No grid points found inside selected domain.")

    iy, ix = np.where(mask)

    iy0 = max(iy.min() - 2, 0)
    iy1 = min(iy.max() + 3, lat.shape[0])

    ix0 = max(ix.min() - 2, 0)
    ix1 = min(ix.max() + 3, lon.shape[1])

    return (
        lat[iy0:iy1, ix0:ix1],
        lon[iy0:iy1, ix0:ix1],
        [f[iy0:iy1, ix0:ix1] for f in fields]
    )


# ============================================================
# R2 runs.json
# ============================================================

def upload_runs_json(init_dt, cycle_str, max_fhr):
    old_runs = []

    try:
        obj = s3.get_object(
            Bucket=BUCKET,
            Key=f"{R2_PRODUCT_PATH}/runs.json"
        )

        old_data = json.loads(
            obj["Body"].read().decode("utf-8")
        )

        old_runs = old_data.get("runs", [])

    except Exception:
        old_runs = []

    new_run = {
        "id": cycle_str,
        "label": init_dt.strftime("%Y-%m-%d %Hz"),
        "max_fhr": max_fhr,
    }

    combined = [new_run]

    for r in old_runs:
        if isinstance(r, str):
            rid = r

            combined.append({
                "id": rid,
                "label": rid.replace("_", " "),
                "max_fhr": max_fhr,
            })

        elif r.get("id") != cycle_str:
            combined.append(r)

    runs_json = {
        "runs": combined[:4]
    }

    with open("runs.json", "w") as f:
        json.dump(runs_json, f, indent=2)

    upload_to_r2(
        "runs.json",
        f"{R2_PRODUCT_PATH}/runs.json",
        content_type="application/json"
    )

    print("Uploaded runs.json with last 4 HREF runs.")


# ============================================================
# FIND HREF CYCLE
# ============================================================

init_dt = find_latest_available_href_cycle()
cycle_str = init_dt.strftime("%Y%m%d_%Hz")

OUTDIR = os.path.join(OUTDIR_BASE, cycle_str)
os.makedirs(OUTDIR, exist_ok=True)

fhrs = range(START_FHR, MAX_FHR + 1)

upload_runs_json(init_dt, cycle_str, MAX_FHR)

print(f"Using {MODEL_LABEL} init:", init_dt.strftime("%Y-%m-%d %HZ"))
print("Forecast hours:", list(fhrs))
print("Output directory:", OUTDIR)
print("Domains:", list(DOMAINS.keys()))

lbf_geom = get_lbf_cwa_geom(LBF_CWA_SHP)


# ============================================================
# LOAD HREF FIELDS
# ============================================================

def load_href_fields_once(fhr):
    print("\n" + "=" * 70)
    print(f"Loading {MODEL_LABEL} | Init {init_dt:%Y-%m-%d %HZ} | F{fhr:02d}")
    print("=" * 70)

    cape_da = href_idx_field(
        init_dt,
        fhr,
        [
            ["CAPE", "90-0 mb", "prob >1500"],
            ["CAPE", "prob >1500"],
        ],
        "CAPE >1500 probability"
    )

    lat, lon = get_lat_lon(cape_da)

    cape_prob = normalize_probability(
        ensure_2d_field(cape_da, "CAPE >1500 probability")
    )

    shear_da = href_idx_field(
        init_dt,
        fhr,
        [
            ["VWSH", "0-6000 m", "prob >20.6"],
            ["VWSH", "0-6000", "prob >20.6"],
            ["VWSH", "prob >20.6"],
        ],
        "0-6km shear >20.6 probability"
    )

    shear_prob = normalize_probability(
        ensure_2d_field(shear_da, "0-6km shear >20.6 probability")
    )

    return {
        "lat": lat,
        "lon": lon,
        "cape_prob": cape_prob,
        "shear_prob": shear_prob,
    }


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_domain_from_fields(fields, domain_key, cfg, fhr):
    global LON_MIN, LON_MAX, LAT_MIN, LAT_MAX

    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = cfg["extent"]

    domain_outdir = os.path.join(OUTDIR, domain_key)
    os.makedirs(domain_outdir, exist_ok=True)

    print(f"Plotting {domain_key.upper()} | F{fhr:02d}")

    try:
        lat = fields["lat"]
        lon = fields["lon"]
        cape_prob = fields["cape_prob"]
        shear_prob = fields["shear_prob"]

        lat_sub, lon_sub, [
            cape_sub,
            shear_sub,
        ] = subset_2d(
            lat,
            lon,
            cape_prob,
            shear_prob
        )

        cape_plot = gaussian_filter(np.nan_to_num(cape_sub, nan=0.0), sigma=0.7)
        shear_plot = gaussian_filter(np.nan_to_num(shear_sub, nan=0.0), sigma=0.7)

        cape_plot = np.where(cape_plot >= 5, cape_plot, np.nan)
        shear_plot = np.where(shear_plot >= 5, shear_plot, np.nan)

        plt.close("all")
        plt.rcParams["contour.negative_linestyle"] = "solid"

        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent(cfg["extent"], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)

        pm = ax.contourf(
            lon_sub,
            lat_sub,
            cape_plot,
            levels=PROB_LEVELS,
            cmap=plt.cm.plasma,
            extend="max",
            transform=ccrs.PlateCarree(),
            zorder=5
        )

        if np.isfinite(shear_plot).any() and np.nanmax(shear_plot) >= 10:
            cs = ax.contour(
                lon_sub,
                lat_sub,
                shear_plot,
                levels=SHEAR_CONTOURS,
                colors="black",
                linewidths=[0.6, 0.7, 0.9, 1.2, 1.4, 1.6, 1.8],
                transform=ccrs.PlateCarree(),
                zorder=8
            )

            ax.clabel(
                cs,
                fmt="%d",
                fontsize=7,
                inline=True,
                inline_spacing=3
            )

        add_shapefile_outline(ax, STATE_SHP, edgecolor="black", linewidth=1.4, zorder=13)
        add_shapefile_outline(ax, COUNTY_SHP, edgecolor="lightgray", linewidth=0.35, zorder=12)

        if lbf_geom is not None:
            add_counties_clipped_to_cwa(ax, COUNTY_SHP, lbf_geom, lw=1.0, color="black", zorder=13)

            ax.add_geometries(
                [lbf_geom],
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="black",
                linewidth=3.5,
                zorder=14
            )

            ax.add_geometries(
                [lbf_geom],
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="white",
                linewidth=1.8,
                zorder=15
            )

        valid_dt = init_dt + timedelta(hours=fhr)
        valid_title = f"F{fhr:02d} Valid: {valid_dt:%a %Y-%m-%d %HZ}"
        init_title = f"Init: {init_dt:%a %Y-%m-%d %HZ} {MODEL_LABEL}"

        main_title = (
            f"{MODEL_LABEL} | Fill: CAPE > 1500 J/kg Probability | "
            "Black Contours: 0-6 km Shear > 40 kts Probability"
        )

        ax.text(
            0.0, 1.042,
            main_title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=cfg["title_size"],
            fontweight="bold"
        )

        ax.text(
            0.0, 1.005,
            valid_title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=cfg["subtitle_size"],
            fontweight="bold"
        )

        ax.text(
            1.0, 1.005,
            init_title,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=cfg["subtitle_size"],
            fontweight="bold"
        )

        divider = make_axes_locatable(ax)

        cax = divider.append_axes(
            "bottom",
            size="3%",
            pad=0.25,
            axes_class=plt.Axes
        )

        cbar = plt.colorbar(
            pm,
            cax=cax,
            orientation="horizontal",
            ticks=PROB_TICKS,
            drawedges=True
        )

        cbar.set_label(
            "Probability of CAPE > 1500 J/kg (%)",
            fontsize=10,
            weight="bold"
        )
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(axis="x", which="both", length=0)

        if os.path.exists(LOGO_PATH):
            logo = mpimg.imread(LOGO_PATH)

            logo_ax = ax.inset_axes(
                [0.82, 0.84, 0.165, 0.155],
                transform=ax.transAxes,
                zorder=50
            )

            logo_ax.imshow(logo)
            logo_ax.axis("off")

        ax.text(
            0.902,
            0.835,
            "NWS North Platte, NE",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="black",
            zorder=51,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
        )

        ax.text(
            0.01,
            0.015,
            "Plot created by: Matthew Labenz",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            weight="bold",
            color="black",
            zorder=40,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
        )

        outname = os.path.join(domain_outdir, f"href_organized_severe_f{fhr:02d}.png")

        plt.savefig(outname, dpi=140, bbox_inches="tight")
        plt.close(fig)

        print("Saved:", outname)

        filename = os.path.basename(outname)

        remote_key = (
            f"{R2_PRODUCT_PATH}/"
            f"{cycle_str}/"
            f"{domain_key}/"
            f"{filename}"
        )

        upload_to_r2(outname, remote_key)

    except Exception as e:
        print(f"Failed {domain_key.upper()} F{fhr:02d}: {e}")


# ============================================================
# MAIN LOOP
# ============================================================

for fhr in fhrs:
    try:
        fields = load_href_fields_once(fhr)

        for domain_key, cfg in DOMAINS.items():
            plot_domain_from_fields(fields, domain_key, cfg, fhr)

    except Exception as e:
        print(f"FAILED F{fhr:02d}: {e}")

print("Done. Uploaded HREF organized severe ingredients probability to R2:", R2_PRODUCT_PATH)
