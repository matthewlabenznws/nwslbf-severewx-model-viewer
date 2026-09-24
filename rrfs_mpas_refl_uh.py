# -*- coding: utf-8 -*-

# ============================================================
# RRFS-MPAS R2 | Reflectivity + UH + Theta Cold Pools
# + 4–6 km Storm-Relative Winds using 700–500 mb proxy
#
# Incremental publishing design:
#   * Process only 00/03/06/09/12/15/18/21Z cycles.
#   * 00Z/12Z publish through F048.
#   * Other selected cycles publish through F012.
#   * Every invocation scans recent FTP cycles independently.
#   * Only forecast hours not already complete in R2 are plotted.
#   * Long 00Z/12Z cycles keep filling even after newer cycles arrive.
#   * runs.json advertises only the highest contiguous FHR actually
#     complete across every configured domain.
#
# GSL FTP:
#   host: gsdftp.fsl.noaa.gov
#   directory: /ncar/upp/det
#
# Required environment variables:
#   AWS_BUCKET
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_REGION
#   GSL_FTP_EMAIL
# ============================================================

import os
import re
import json
import zipfile
import time
import tempfile
import shutil
from ftplib import FTP
from pathlib import Path
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import requests
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.image as mpimg

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.ops import unary_union
from shapely.prepared import prep

from matplotlib.colors import ListedColormap, BoundaryNorm
from botocore.config import Config

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_array,
    codes_release,
)


# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL / FTP SETTINGS
# ============================================================

MODEL_KEY = "rrfs_mpas"
PRODUCT_KEY = "refl_uh"

FTP_HOST = "gsdftp.fsl.noaa.gov"
FTP_DIR = "/ncar/upp/det"

GSL_FTP_EMAIL = os.environ["GSL_FTP_EMAIL"]

# Only these cycles are published.
TARGET_FHRS = {
    0: 48,
    3: 12,
    6: 12,
    9: 12,
    12: 48,
    15: 12,
    18: 12,
    21: 12,
}

# Scan this far back on every invocation.  This MUST be long enough
# that a slow 00Z/12Z run remains eligible while newer runs arrive.
LOOKBACK_HOURS = 36

# Number of cycles retained in the viewer manifest.
KEEP_RUNS = 8

# If True, FHR completion is determined by requiring the PNG in every
# currently configured domain. This is the safest behavior for the viewer.
REQUIRE_ALL_DOMAINS_FOR_COMPLETE_FHR = True

# FTP retry behavior.
FTP_CONNECT_ATTEMPTS = 3
FTP_DOWNLOAD_ATTEMPTS = 3
FTP_RETRY_WAIT_SECONDS = 20

# Forecast-hour retry behavior.
MAX_FHR_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 20


# ============================================================
# R2 SETUP
# ============================================================

BUCKET = os.environ["AWS_BUCKET"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_REGION"],
    config=Config(
        retries={
            "max_attempts": 10,
            "mode": "standard",
        }
    ),
)


def upload_to_r2(local_file, remote_key, content_type="image/png"):
    s3.upload_file(
        local_file,
        BUCKET,
        remote_key,
        ExtraArgs={"ContentType": content_type},
    )
    print("Uploaded to R2:", remote_key)


def r2_object_exists(remote_key):
    try:
        s3.head_object(Bucket=BUCKET, Key=remote_key)
        return True
    except Exception:
        return False


def get_json_from_r2(remote_key, default=None):
    if default is None:
        default = {}

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=remote_key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return default


# ============================================================
# ASSETS
# ============================================================

zip_path = os.path.join(BASE_DIR, "assets", "c_18mr25.zip")
extract_path = os.path.join(BASE_DIR, "assets")

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

COUNTY_SHP = os.path.join(
    BASE_DIR,
    "assets",
    "cb_2018_us_county_500k.shp",
)

STATE_SHP = os.path.join(
    BASE_DIR,
    "assets",
    "cb_2018_us_state_500k.shp",
)

LBF_CWA_SHP = os.path.join(
    BASE_DIR,
    "assets",
    "c_18mr25.shp",
)

LOGO_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "NOAANWSLogos.png",
)


# ============================================================
# DOMAINS
# ============================================================

DOMAINS = {
    "lbf": {
        "label": "LBF",
        "extent": [-103.8, -97.0, 40.0, 43.4],
        "title_size": 14,
        "subtitle_size": 11,
        "logo_ax": [0.78, 0.70, 0.10, 0.10],
        "office_text_xy": [0.83, 0.71],
        "credit_xy": [0.13, 0.25],
        "barb_skip": 11,
    },

    "regional": {
        "label": "Default",
        "extent": [-107.5, -93.0, 38.5, 44.2],
        "title_size": 13,
        "subtitle_size": 11,
        "logo_ax": [0.78, 0.63, 0.10, 0.10],
        "office_text_xy": [0.83, 0.64],
        "credit_xy": [0.13, 0.31],
        "barb_skip": 20,
    },

    "central_plains": {
        "label": "Central Plains",
        "extent": [-107.5, -91.0, 34.5, 45.2],
        "title_size": 13,
        "subtitle_size": 11,
        "logo_ax": [0.78, 0.77, 0.10, 0.10],
        "office_text_xy": [0.83, 0.78],
        "credit_xy": [0.13, 0.175],
        "barb_skip": 24,
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

    r = requests.get(
        SPC_DAY1_CAT_URL,
        params=params,
        timeout=30,
    )
    r.raise_for_status()

    data = r.json()

    if "features" not in data or len(data["features"]) == 0:
        raise RuntimeError("SPC query returned no features.")

    return gpd.GeoDataFrame.from_features(
        data["features"],
        crs="EPSG:4326",
    )


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
            print(
                "SPC severe domain skipped: "
                "could not find risk category column."
            )
            return

        gdf["risk"] = gdf[risk_col].astype(str).str.upper()
        gdf["risk_rank"] = gdf["risk"].map(SPC_RISK_ORDER)

        severe = gdf[
            gdf["risk_rank"] >= SPC_RISK_ORDER[MIN_SPC_RISK]
        ].copy()

        if severe.empty:
            print("SPC severe domain skipped: no SLGT+ risk found.")
            return

        highest_rank = severe["risk_rank"].max()

        highest = severe[
            severe["risk_rank"] == highest_rank
        ].copy()

        highest_proj = highest.to_crs(epsg=5070)
        highest["_area"] = highest_proj.geometry.area.values

        main_poly = highest.loc[
            highest["_area"].idxmax()
        ]

        highest_label = main_poly["risk"]

        main_gdf = gpd.GeoDataFrame(
            [main_poly],
            geometry="geometry",
            crs="EPSG:4326",
        )

        centroid_proj = (
            main_gdf
            .to_crs(epsg=5070)
            .geometry
            .centroid
        )

        centroid_ll = gpd.GeoSeries(
            centroid_proj,
            crs="EPSG:5070",
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
            "title_size": 13,
            "subtitle_size": 11,
            "logo_ax": [0.78, 0.70, 0.10, 0.10],
            "office_text_xy": [0.83, 0.71],
            "credit_xy": [0.13, 0.25],
            "barb_skip": 22,
        }

        print(
            f"Added SPC severe domain: {highest_label}"
        )
        print(
            f"SPC severe extent: {extent}"
        )

    except Exception as e:
        print(
            "SPC severe domain skipped due to error:",
            e,
        )


add_spc_severe_domain()


# ============================================================
# PLOT SETTINGS
# ============================================================

PLOT_SR_WIND_BARBS = True
BARB_SKIP = 11
PLOT_CITY_LABELS = False

STATIONS = {
    "Gordon":       (-102.2038, 42.8061),
    "Ellsworth":    (-102.3172, 42.0628),
    "Oshkosh":      (-102.3465, 41.4047),
    "Ogallala":     (-101.7205, 41.1275),
    "Mullen":       (-101.0427, 42.0425),
    "Valentine":    (-100.5514, 42.8586),
    "Ainsworth":    (-99.8516, 42.5467),
    "Burwell":      (-99.1766, 41.7666),
    "North Platte": (-100.6689, 41.1220),
    "Broken Bow":   (-99.6385, 41.4365),
    "Imperial":     (-101.6243, 40.5106),
    "Curtis":       (-100.5219, 40.6344),
    "O'Neill":      (-98.6470, 42.4578),
    "Butte":        (-98.8511, 42.9130),
}


# ============================================================
# REFLECTIVITY COLOR TABLE
# ============================================================

bounds = [
    0, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30,
    32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5,
    55, 57.5, 60, 62.5, 65, 67.5, 70, 72.5,
]

colors = [
    "#ffffff",
    "#dae2f2",
    "#b4c4e5",
    "#8fa7d9",
    "#6a89cb",
    "#486cbf",
    "#2c4eb2",
    "#1e4f5e",
    "#48746d",
    "#799b7c",
    "#aac08b",
    "#fbf477",
    "#f1d461",
    "#e7b54c",
    "#dd9738",
    "#d37826",
    "#ca5917",
    "#c31d14",
    "#9a1511",
    "#710e10",
    "#9c3aae",
    "#7f27a0",
    "#601392",
    "#828282",
    "#b4b4b4",
    "#e6e6e6",
]

cmap = ListedColormap(
    colors,
    name="reflec_bins",
)

norm = BoundaryNorm(
    bounds,
    cmap.N,
    clip=True,
)

REF_LEVELS = [
    10, 15, 20, 25, 30, 35, 40,
    45, 50, 55, 60, 65, 70, 75,
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def to_lon180(lon):
    return ((np.asarray(lon) + 180) % 360) - 180


def ms_to_kt(ms):
    return np.asarray(ms) * 1.94384


def safe_get(gid, key, default=None):
    try:
        return codes_get(gid, key)
    except Exception:
        return default


def values_match(a, b, tol=0.01):
    if a is None or b is None:
        return False

    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return str(a) == str(b)


def cycle_id(init_dt):
    return init_dt.strftime("%Y%m%d_%Hz")


def cycle_label(init_dt):
    return init_dt.strftime("%Y-%m-%d %Hz")


def parse_cycle_id(rid):
    return datetime.strptime(
        rid,
        "%Y%m%d_%Hz",
    )


def target_max_fhr(init_dt):
    return TARGET_FHRS[init_dt.hour]


def output_filename(fhr):
    # Keep the same filename for every domain, just as the HRRR
    # script does. The domain is represented by the directory.
    return f"rrfs_mpas_lbf_f{fhr:03d}.png"


def remote_png_key(init_dt, domain_key, fhr):
    return (
        f"runs/cams/{MODEL_KEY}/{PRODUCT_KEY}/"
        f"{cycle_id(init_dt)}/"
        f"{domain_key}/"
        f"{output_filename(fhr)}"
    )


RUNS_JSON_KEY = (
    f"runs/cams/{MODEL_KEY}/{PRODUCT_KEY}/runs.json"
)


# ============================================================
# SHAPEFILE HELPERS
# ============================================================

def add_shapefile_outline(
    ax,
    shp_path,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    edgecolor="k",
    linewidth=1.2,
    zorder=6,
):
    if not os.path.exists(shp_path):
        print("Missing shapefile:", shp_path)
        return

    gdf = (
        gpd.read_file(shp_path)
        .to_crs(epsg=4326)
    )

    gdf = gdf.cx[
        lon_min - 1:lon_max + 1,
        lat_min - 1:lat_max + 1,
    ]

    ax.add_geometries(
        gdf.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )


def get_lbf_cwa_geom(cwa_shp_path):
    reader = shpreader.Reader(cwa_shp_path)
    recs = list(reader.records())

    geoms = [
        r.geometry
        for r in recs
        if (
            str(
                r.attributes.get("CWA", "")
            ).upper() == "LBF"
            or
            str(
                r.attributes.get("WFO", "")
            ).upper() == "LBF"
        )
    ]

    if not geoms:
        geoms = [
            r.geometry
            for r in recs
        ]

    return unary_union(geoms)


def add_counties_clipped_to_cwa(
    ax,
    counties_shp_path,
    cwa_geom,
    lw=1.0,
    color="black",
    zorder=6,
):
    reader = shpreader.Reader(
        counties_shp_path
    )

    cwa_p = prep(cwa_geom)
    clipped = []

    for r in reader.records():
        g = r.geometry

        if cwa_p.intersects(g):
            inter = g.intersection(
                cwa_geom
            )

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


def plot_city_labels(
    ax,
    cities,
    zorder=40,
    fontsize=9,
):
    for name, (lon, lat) in cities.items():
        ax.text(
            lon,
            lat,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=fontsize,
            color="black",
            ha="center",
            va="center",
            zorder=zorder,
            path_effects=[
                pe.withStroke(
                    linewidth=3,
                    foreground="white",
                )
            ],
        )


# ============================================================
# FTP INVENTORY
# ============================================================

RRFS_FILE_RE = re.compile(
    r"^(?P<date>\d{8})(?P<hour>\d{2})"
    r"\.rrfs\.t(?P=hour)z\.testbed"
    r"\.f(?P<fhr>\d{3})\.conus\.grib2$"
)


def connect_ftp():
    last_error = None

    for attempt in range(
        1,
        FTP_CONNECT_ATTEMPTS + 1,
    ):
        try:
            print(
                f"Connecting to {FTP_HOST} "
                f"(attempt {attempt}/"
                f"{FTP_CONNECT_ATTEMPTS})"
            )

            ftp = FTP(
                FTP_HOST,
                timeout=90,
            )

            ftp.login(
                "anonymous",
                GSL_FTP_EMAIL,
            )

            ftp.set_pasv(True)
            ftp.cwd(FTP_DIR)

            return ftp

        except Exception as e:
            last_error = e

            print(
                "FTP connection failed:",
                e,
            )

            if attempt < FTP_CONNECT_ATTEMPTS:
                time.sleep(
                    FTP_RETRY_WAIT_SECONDS
                )

    raise RuntimeError(
        f"Could not connect to GSL FTP: "
        f"{last_error}"
    )


def list_rrfs_inventory():
    ftp = connect_ftp()

    try:
        names = ftp.nlst()
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()

    now = datetime.now(timezone.utc).replace(
        tzinfo=None
    )

    cutoff = now - timedelta(
        hours=LOOKBACK_HOURS
    )

    inventory = {}

    for name in names:
        m = RRFS_FILE_RE.match(
            os.path.basename(name)
        )

        if not m:
            continue

        init_dt = datetime.strptime(
            m.group("date")
            + m.group("hour"),
            "%Y%m%d%H",
        )

        if init_dt < cutoff:
            continue

        if init_dt.hour not in TARGET_FHRS:
            continue

        fhr = int(
            m.group("fhr")
        )

        if fhr > target_max_fhr(
            init_dt
        ):
            continue

        inventory.setdefault(
            init_dt,
            {},
        )[fhr] = os.path.basename(
            name
        )

    return inventory


def download_ftp_file(
    remote_name,
    local_path,
):
    last_error = None

    for attempt in range(
        1,
        FTP_DOWNLOAD_ATTEMPTS + 1,
    ):
        ftp = None

        try:
            print(
                f"Downloading {remote_name} "
                f"(attempt {attempt}/"
                f"{FTP_DOWNLOAD_ATTEMPTS})"
            )

            ftp = connect_ftp()

            with open(
                local_path,
                "wb",
            ) as f:
                ftp.retrbinary(
                    f"RETR {remote_name}",
                    f.write,
                    blocksize=1024 * 1024,
                )

            try:
                ftp.quit()
            except Exception:
                ftp.close()

            size_mb = (
                os.path.getsize(local_path)
                / 1024
                / 1024
            )

            print(
                f"Downloaded {remote_name}: "
                f"{size_mb:.1f} MB"
            )

            return

        except Exception as e:
            last_error = e

            print(
                "FTP download failed:",
                e,
            )

            if ftp is not None:
                try:
                    ftp.close()
                except Exception:
                    pass

            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass

            if attempt < FTP_DOWNLOAD_ATTEMPTS:
                time.sleep(
                    FTP_RETRY_WAIT_SECONDS
                )

    raise RuntimeError(
        f"Could not download {remote_name}: "
        f"{last_error}"
    )


# ============================================================
# GRIB FIELD IDENTIFICATION
# ============================================================

# The testbed stream currently exposes several local UPP
# diagnostics as "unknown" in ecCodes. Therefore UH and 1-km
# max reflectivity are identified by raw discipline/category/
# parameter metadata rather than shortName.
#
# Established from the test file:
#   d0 c16 p198 = max 1-km reflectivity
#   d0 c7  p199 = max updraft helicity
#
# UH layer metadata:
#   2–5 km = first=5000, second=2000 m
#   0–2 km = first=2000, second=0 m
#
# 2-m theta is directly available as shortName=pt.
#
# Winds:
#   700/500 mb U/V plus USTM/VSTM are available.
#   To stay close to the existing HRRR product, this first
#   production implementation uses the mean 700/500-mb wind
#   as the 4–6-km environmental-wind proxy, then subtracts
#   the model storm-motion field.


def message_metadata(gid):
    return {
        "discipline": safe_get(
            gid,
            "discipline",
        ),
        "category": safe_get(
            gid,
            "parameterCategory",
        ),
        "parameter": safe_get(
            gid,
            "parameterNumber",
        ),
        "shortName": safe_get(
            gid,
            "shortName",
        ),
        "typeOfLevel": safe_get(
            gid,
            "typeOfLevel",
        ),
        "level": safe_get(
            gid,
            "level",
        ),
        "topLevel": safe_get(
            gid,
            "topLevel",
        ),
        "bottomLevel": safe_get(
            gid,
            "bottomLevel",
        ),
        "scaledValueOfFirstFixedSurface": safe_get(
            gid,
            "scaledValueOfFirstFixedSurface",
        ),
        "scaledValueOfSecondFixedSurface": safe_get(
            gid,
            "scaledValueOfSecondFixedSurface",
        ),
        "stepType": safe_get(
            gid,
            "stepType",
        ),
        "Nx": safe_get(
            gid,
            "Nx",
        ),
        "Ny": safe_get(
            gid,
            "Ny",
        ),
        "Ni": safe_get(
            gid,
            "Ni",
        ),
        "Nj": safe_get(
            gid,
            "Nj",
        ),
    }


def first_surface_value(meta):
    for key in (
        "scaledValueOfFirstFixedSurface",
        "topLevel",
        "level",
    ):
        value = meta.get(key)

        if value is not None:
            return value

    return None


def second_surface_value(meta):
    for key in (
        "scaledValueOfSecondFixedSurface",
        "bottomLevel",
    ):
        value = meta.get(key)

        if value is not None:
            return value

    return None


def is_refl_1km(meta):
    return (
        meta["discipline"] == 0
        and meta["category"] == 16
        and meta["parameter"] == 198
        and meta["typeOfLevel"] == "heightAboveGround"
        and values_match(
            meta["level"],
            1000,
        )
        and str(
            meta["stepType"]
        ).lower() == "max"
    )


def is_uh25(meta):
    first = first_surface_value(meta)
    second = second_surface_value(meta)

    return (
        meta["discipline"] == 0
        and meta["category"] == 7
        and meta["parameter"] == 199
        and meta["typeOfLevel"]
        == "heightAboveGroundLayer"
        and values_match(first, 5000)
        and values_match(second, 2000)
        and str(
            meta["stepType"]
        ).lower() == "max"
    )


def is_uh02(meta):
    first = first_surface_value(meta)
    second = second_surface_value(meta)

    return (
        meta["discipline"] == 0
        and meta["category"] == 7
        and meta["parameter"] == 199
        and meta["typeOfLevel"]
        == "heightAboveGroundLayer"
        and values_match(first, 2000)
        and values_match(second, 0)
        and str(
            meta["stepType"]
        ).lower() == "max"
    )


def is_theta2(meta):
    return (
        str(meta["shortName"]).lower()
        == "pt"
        and meta["typeOfLevel"]
        == "heightAboveGround"
        and values_match(
            meta["level"],
            2,
        )
    )


def is_pressure_wind(
    meta,
    short_name,
    pressure_hpa,
):
    return (
        str(meta["shortName"]).lower()
        == short_name.lower()
        and meta["typeOfLevel"]
        == "isobaricInhPa"
        and values_match(
            meta["level"],
            pressure_hpa,
        )
    )


def is_storm_motion(
    meta,
    short_name,
):
    if (
        str(meta["shortName"]).lower()
        != short_name.lower()
    ):
        return False

    # Testbed stream has USTM/VSTM on a 0–6000-m
    # heightAboveGroundLayer. shortName is distinctive enough,
    # but retain the layer check when metadata is available.
    if meta["typeOfLevel"] != "heightAboveGroundLayer":
        return False

    first = first_surface_value(meta)
    second = second_surface_value(meta)

    layer_values = {
        int(round(float(x)))
        for x in (first, second)
        if x is not None
    }

    if layer_values:
        return layer_values == {
            0,
            6000,
        }

    return True


FIELD_MATCHERS = {
    "refl": is_refl_1km,
    "uh25": is_uh25,
    "uh02": is_uh02,
    "theta": is_theta2,
    "u700": lambda m: is_pressure_wind(
        m,
        "u",
        700,
    ),
    "v700": lambda m: is_pressure_wind(
        m,
        "v",
        700,
    ),
    "u500": lambda m: is_pressure_wind(
        m,
        "u",
        500,
    ),
    "v500": lambda m: is_pressure_wind(
        m,
        "v",
        500,
    ),
    "ustm": lambda m: is_storm_motion(
        m,
        "ustm",
    ),
    "vstm": lambda m: is_storm_motion(
        m,
        "vstm",
    ),
}


def get_grid_shape(gid, values):
    nx = safe_get(gid, "Nx")
    ny = safe_get(gid, "Ny")

    if nx is None or ny is None:
        nx = safe_get(gid, "Ni")
        ny = safe_get(gid, "Nj")

    if nx is None or ny is None:
        raise RuntimeError(
            "Could not determine GRIB Nx/Ny."
        )

    nx = int(nx)
    ny = int(ny)

    if nx * ny != values.size:
        raise RuntimeError(
            f"Grid shape {ny}x{nx} does not "
            f"match {values.size} values."
        )

    return ny, nx


def read_message_arrays(gid):
    values_1d = np.asarray(
        codes_get_array(
            gid,
            "values",
        ),
        dtype=float,
    )

    lats_1d = np.asarray(
        codes_get_array(
            gid,
            "latitudes",
        ),
        dtype=float,
    )

    lons_1d = to_lon180(
        codes_get_array(
            gid,
            "longitudes",
        )
    )

    ny, nx = get_grid_shape(
        gid,
        values_1d,
    )

    return (
        values_1d.reshape(ny, nx),
        lats_1d.reshape(ny, nx),
        np.asarray(
            lons_1d,
            dtype=float,
        ).reshape(ny, nx),
    )


def load_rrfs_fields_once(grib_path):
    print("\n" + "=" * 70)
    print("Reading RRFS-MPAS GRIB once:", grib_path)
    print("=" * 70)

    found = {}

    with open(
        grib_path,
        "rb",
    ) as f:
        while True:
            gid = codes_grib_new_from_file(f)

            if gid is None:
                break

            try:
                meta = message_metadata(gid)

                for field_name, matcher in FIELD_MATCHERS.items():
                    if field_name in found:
                        continue

                    try:
                        matched = matcher(meta)
                    except Exception:
                        matched = False

                    if not matched:
                        continue

                    values, lat, lon = read_message_arrays(
                        gid
                    )

                    found[field_name] = {
                        "values": values,
                        "lat": lat,
                        "lon": lon,
                        "meta": meta,
                    }

                    print(
                        f"Found {field_name}: "
                        f"shortName={meta['shortName']} "
                        f"type={meta['typeOfLevel']} "
                        f"level={meta['level']} "
                        f"d/c/p="
                        f"{meta['discipline']}/"
                        f"{meta['category']}/"
                        f"{meta['parameter']} "
                        f"stepType={meta['stepType']}"
                    )

            finally:
                codes_release(gid)

    missing = [
        name
        for name in FIELD_MATCHERS
        if name not in found
    ]

    if missing:
        raise RuntimeError(
            "Required RRFS-MPAS fields missing: "
            + ", ".join(missing)
        )

    # All required testbed fields were observed on the same
    # UPP grid. Verify this before doing array arithmetic.
    ref_shape = found["refl"]["values"].shape
    ref_lat = found["refl"]["lat"]
    ref_lon = found["refl"]["lon"]

    for name, item in found.items():
        if item["values"].shape != ref_shape:
            raise RuntimeError(
                f"Grid mismatch: refl={ref_shape}, "
                f"{name}={item['values'].shape}"
            )

    refl = np.asarray(
        found["refl"]["values"],
        dtype=float,
    )

    refl = np.where(
        refl >= REF_LEVELS[0],
        refl,
        np.nan,
    )

    uh25 = np.asarray(
        found["uh25"]["values"],
        dtype=float,
    )

    uh02 = np.asarray(
        found["uh02"]["values"],
        dtype=float,
    )

    theta = np.asarray(
        found["theta"]["values"],
        dtype=float,
    )

    theta_bg = gaussian_filter(
        theta,
        sigma=18,
    )

    theta_prime = (
        theta
        - theta_bg
    )

    u700 = np.asarray(
        found["u700"]["values"],
        dtype=float,
    )

    v700 = np.asarray(
        found["v700"]["values"],
        dtype=float,
    )

    u500 = np.asarray(
        found["u500"]["values"],
        dtype=float,
    )

    v500 = np.asarray(
        found["v500"]["values"],
        dtype=float,
    )

    # MPAS testbed UPP stream does not provide the 600-mb
    # wind used in the HRRR product, so use the mean of
    # 700 and 500 mb for this first production version.
    u46 = np.nanmean(
        np.stack(
            [u700, u500]
        ),
        axis=0,
    )

    v46 = np.nanmean(
        np.stack(
            [v700, v500]
        ),
        axis=0,
    )

    storm_u = np.asarray(
        found["ustm"]["values"],
        dtype=float,
    )

    storm_v = np.asarray(
        found["vstm"]["values"],
        dtype=float,
    )

    sr_u46 = u46 - storm_u
    sr_v46 = v46 - storm_v

    sr46_kt = ms_to_kt(
        np.sqrt(
            sr_u46 ** 2
            + sr_v46 ** 2
        )
    )

    return {
        "lat": ref_lat,
        "lon": ref_lon,
        "refl": refl,
        "uh25": uh25,
        "uh02": uh02,
        "theta_prime": theta_prime,
        "sr46_kt": sr46_kt,
        "sr_u46": sr_u46,
        "sr_v46": sr_v46,
    }


# ============================================================
# DOMAIN SUBSETTING
# ============================================================

def subset_2d(
    lat,
    lon,
    extent,
    *fields,
):
    lon_min, lon_max, lat_min, lat_max = extent

    mask = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lon >= lon_min)
        & (lon <= lon_max)
        & (lat >= lat_min)
        & (lat <= lat_max)
    )

    if not np.any(mask):
        raise RuntimeError(
            "No grid points found inside selected domain."
        )

    iy, ix = np.where(mask)

    iy0 = max(
        iy.min() - 2,
        0,
    )

    iy1 = min(
        iy.max() + 3,
        lat.shape[0],
    )

    ix0 = max(
        ix.min() - 2,
        0,
    )

    ix1 = min(
        ix.max() + 3,
        lon.shape[1],
    )

    return (
        lat[
            iy0:iy1,
            ix0:ix1,
        ],
        lon[
            iy0:iy1,
            ix0:ix1,
        ],
        [
            f[
                iy0:iy1,
                ix0:ix1,
            ]
            for f in fields
        ],
    )


# ============================================================
# RUN / R2 STATE
# ============================================================

def required_domain_keys():
    return list(
        DOMAINS.keys()
    )


def fhr_complete_in_r2(
    init_dt,
    fhr,
):
    domain_keys = required_domain_keys()

    if not domain_keys:
        return False

    results = [
        r2_object_exists(
            remote_png_key(
                init_dt,
                domain_key,
                fhr,
            )
        )
        for domain_key in domain_keys
    ]

    if REQUIRE_ALL_DOMAINS_FOR_COMPLETE_FHR:
        return all(results)

    return any(results)


def missing_domains_for_fhr(
    init_dt,
    fhr,
):
    missing = []

    for domain_key in required_domain_keys():
        key = remote_png_key(
            init_dt,
            domain_key,
            fhr,
        )

        if not r2_object_exists(key):
            missing.append(
                domain_key
            )

    return missing


def highest_contiguous_complete_fhr(
    init_dt,
):
    target = target_max_fhr(
        init_dt
    )

    highest = -1

    for fhr in range(
        0,
        target + 1,
    ):
        if fhr_complete_in_r2(
            init_dt,
            fhr,
        ):
            highest = fhr
        else:
            break

    return highest


def build_runs_json(
    candidate_cycles,
):
    old_data = get_json_from_r2(
        RUNS_JSON_KEY,
        default={"runs": []},
    )

    old_runs = old_data.get(
        "runs",
        [],
    )

    run_map = {}

    # Preserve existing valid entries first.
    for r in old_runs:
        if isinstance(r, str):
            rid = r

            try:
                dt = parse_cycle_id(rid)
            except Exception:
                continue

            run_map[rid] = {
                "id": rid,
                "label": cycle_label(dt),
                "max_fhr": 0,
            }

        elif isinstance(r, dict):
            rid = r.get("id")

            if rid:
                run_map[rid] = dict(r)

    # Refresh every cycle considered during this invocation.
    for init_dt in candidate_cycles:
        max_complete = highest_contiguous_complete_fhr(
            init_dt
        )

        rid = cycle_id(init_dt)

        # Do not advertise a cycle until F000 exists in every
        # required domain.
        if max_complete < 0:
            run_map.pop(
                rid,
                None,
            )
            continue

        run_map[rid] = {
            "id": rid,
            "label": cycle_label(init_dt),
            "max_fhr": max_complete,
        }

    sortable = []

    for rid, entry in run_map.items():
        try:
            dt = parse_cycle_id(rid)
        except Exception:
            continue

        sortable.append(
            (
                dt,
                entry,
            )
        )

    sortable.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    runs = [
        entry
        for _, entry in sortable[
            :KEEP_RUNS
        ]
    ]

    return {
        "runs": runs
    }


def upload_runs_json(
    candidate_cycles,
):
    runs_json = build_runs_json(
        candidate_cycles
    )

    local_path = os.path.join(
        BASE_DIR,
        "runs_rrfs_mpas.json",
    )

    with open(
        local_path,
        "w",
    ) as f:
        json.dump(
            runs_json,
            f,
            indent=2,
        )

    upload_to_r2(
        local_path,
        RUNS_JSON_KEY,
        content_type="application/json",
    )

    try:
        os.remove(
            local_path
        )
    except Exception:
        pass

    print(
        "Updated RRFS-MPAS runs.json:"
    )

    print(
        json.dumps(
            runs_json,
            indent=2,
        )
    )


# ============================================================
# PLOT DOMAIN
# ============================================================

def plot_domain_from_fields(
    fields,
    domain_key,
    cfg,
    init_dt,
    fhr,
    local_cycle_dir,
    lbf_geom,
):
    extent = cfg["extent"]

    lon_min, lon_max, lat_min, lat_max = extent

    domain_outdir = os.path.join(
        local_cycle_dir,
        domain_key,
    )

    os.makedirs(
        domain_outdir,
        exist_ok=True,
    )

    print(
        f"Plotting {domain_key.upper()} "
        f"| {cycle_id(init_dt)} "
        f"| F{fhr:03d}"
    )

    lat = fields["lat"]
    lon = fields["lon"]
    refl = fields["refl"]
    uh25 = fields["uh25"]
    uh02 = fields["uh02"]
    theta_prime = fields["theta_prime"]
    sr46_kt = fields["sr46_kt"]
    sr_u46 = fields["sr_u46"]
    sr_v46 = fields["sr_v46"]

    (
        lat_sub,
        lon_sub,
        [
            refl_sub,
            uh25_sub,
            uh02_sub,
            theta_prime_sub,
            sr46_sub,
            sr_u46_sub,
            sr_v46_sub,
        ],
    ) = subset_2d(
        lat,
        lon,
        extent,
        refl,
        uh25,
        uh02,
        theta_prime,
        sr46_kt,
        sr_u46,
        sr_v46,
    )

    refl_plot = gaussian_filter(
        np.nan_to_num(
            refl_sub,
            nan=0.0,
        ),
        sigma=0.5,
    )

    refl_plot = np.where(
        refl_plot >= 5,
        refl_plot,
        np.nan,
    )

    uh25_plot = gaussian_filter(
        uh25_sub,
        sigma=0.2,
    )

    uh02_plot = gaussian_filter(
        uh02_sub,
        sigma=0.2,
    )

    uh_combined = np.where(
        (uh25_plot >= 75)
        | (uh02_plot >= 50),
        1,
        np.nan,
    )

    theta_prime_smooth = gaussian_filter(
        theta_prime_sub,
        sigma=2.5,
    )

    theta_cp_mask = np.ma.masked_where(
        theta_prime_smooth > -2.0,
        theta_prime_smooth,
    )

    plt.close("all")

    plt.rcParams["hatch.color"] = "#b7d6ff"
    plt.rcParams["hatch.linewidth"] = 0.7
    plt.rcParams["contour.negative_linestyle"] = "solid"

    fig = plt.figure(
        figsize=(14, 10)
    )

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    ax.set_extent(
        extent,
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="white",
        zorder=0,
    )

    # --------------------------------------------------------
    # THETA COLD POOLS
    # --------------------------------------------------------

    ax.contourf(
        lon_sub,
        lat_sub,
        theta_cp_mask,
        levels=[
            -30,
            -2,
        ],
        colors="none",
        hatches=["///"],
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    ax.contour(
        lon_sub,
        lat_sub,
        theta_prime_smooth,
        levels=[-2],
        colors="#b7d6ff",
        linewidths=1.2,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # --------------------------------------------------------
    # 1-KM REFLECTIVITY
    # --------------------------------------------------------

    pm = ax.contourf(
        lon_sub,
        lat_sub,
        refl_plot,
        levels=bounds,
        cmap=cmap,
        norm=norm,
        extend="neither",
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # --------------------------------------------------------
    # UPDRAFT HELICITY
    # --------------------------------------------------------

    ax.contourf(
        lon_sub,
        lat_sub,
        uh_combined,
        levels=[
            0.5,
            1.5,
        ],
        colors=["#8f8f8f"],
        alpha=0.55,
        transform=ccrs.PlateCarree(),
        zorder=8,
    )

    ax.contour(
        lon_sub,
        lat_sub,
        uh25_plot,
        levels=[75],
        colors="#4a4a4a",
        linewidths=0.9,
        transform=ccrs.PlateCarree(),
        zorder=9,
    )

    ax.contour(
        lon_sub,
        lat_sub,
        uh02_plot,
        levels=[50],
        colors="black",
        linewidths=0.9,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    # --------------------------------------------------------
    # STORM-RELATIVE WIND BARBS
    # --------------------------------------------------------

    if PLOT_SR_WIND_BARBS:
        barb_skip = cfg.get(
            "barb_skip",
            BARB_SKIP,
        )

        ax.barbs(
            lon_sub[
                ::barb_skip,
                ::barb_skip,
            ],
            lat_sub[
                ::barb_skip,
                ::barb_skip,
            ],
            ms_to_kt(
                sr_u46_sub[
                    ::barb_skip,
                    ::barb_skip,
                ]
            ),
            ms_to_kt(
                sr_v46_sub[
                    ::barb_skip,
                    ::barb_skip,
                ]
            ),
            length=5,
            linewidth=0.7,
            color="black",
            transform=ccrs.PlateCarree(),
            zorder=23,
        )

    # --------------------------------------------------------
    # STATES / COUNTIES / LBF CWA
    # --------------------------------------------------------

    add_shapefile_outline(
        ax,
        STATE_SHP,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        edgecolor="black",
        linewidth=1.4,
        zorder=13,
    )

    add_shapefile_outline(
        ax,
        COUNTY_SHP,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        edgecolor="lightgray",
        linewidth=0.35,
        zorder=12,
    )

    add_counties_clipped_to_cwa(
        ax,
        COUNTY_SHP,
        lbf_geom,
        lw=1.0,
        color="black",
        zorder=13,
    )

    ax.add_geometries(
        [lbf_geom],
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=3.5,
        zorder=14,
    )

    ax.add_geometries(
        [lbf_geom],
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="white",
        linewidth=1.8,
        zorder=15,
    )

    if PLOT_CITY_LABELS:
        plot_city_labels(
            ax,
            STATIONS,
            zorder=40,
            fontsize=9,
        )

    # --------------------------------------------------------
    # TITLES
    # --------------------------------------------------------

    valid_dt = (
        init_dt
        + timedelta(hours=fhr)
    )

    main_title = (
        "RRFS-MPAS | 1 km Refl, 2-5km UH > 75, "
        "0-2km UH > 50, θ Cold Pools, 4-6 km SR Winds"
    )

    valid_title = (
        f"F{fhr:03d} Valid: "
        f"{valid_dt:%a %Y-%m-%d %Hz}"
    )

    init_title = (
        f"Init: {init_dt:%a %Y-%m-%d %Hz} "
        f"RRFS-MPAS"
    )

    ax.text(
        0.0,
        1.042,
        main_title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=cfg["title_size"],
        fontweight="bold",
    )

    ax.text(
        0.0,
        1.005,
        valid_title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=cfg["subtitle_size"],
        fontweight="bold",
    )

    ax.text(
        1.0,
        1.005,
        init_title,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=cfg["subtitle_size"],
        fontweight="bold",
    )

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    divider = make_axes_locatable(
        ax
    )

    cax = divider.append_axes(
        "bottom",
        size="3%",
        pad=0.25,
        axes_class=plt.Axes,
    )

    cbar = plt.colorbar(
        pm,
        cax=cax,
        orientation="horizontal",
        ticks=REF_LEVELS,
        drawedges=True,
    )

    cbar.set_label(
        "1 km Reflectivity (dBZ)",
        fontsize=10,
        weight="bold",
    )

    cbar.ax.xaxis.set_label_position(
        "top"
    )

    cbar.ax.tick_params(
        axis="x",
        which="both",
        length=0,
    )

    # --------------------------------------------------------
    # LOGO / OFFICE LABEL
    # --------------------------------------------------------

    if os.path.exists(LOGO_PATH):
        logo = mpimg.imread(
            LOGO_PATH
        )

        logo_ax = ax.inset_axes(
            [
                0.82,
                0.84,
                0.165,
                0.155,
            ],
            transform=ax.transAxes,
            zorder=50,
        )

        logo_ax.imshow(
            logo
        )

        logo_ax.axis(
            "off"
        )

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
        path_effects=[
            pe.withStroke(
                linewidth=2.5,
                foreground="white",
            )
        ],
    )

    ax.text(
        0.01,
        0.015,
        "Plot created by: Matthew Labenz",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        weight="bold",
        color="black",
        zorder=40,
        path_effects=[
            pe.withStroke(
                linewidth=2.5,
                foreground="white",
            )
        ],
    )

    # --------------------------------------------------------
    # SAVE / UPLOAD
    # --------------------------------------------------------

    outname = os.path.join(
        domain_outdir,
        output_filename(fhr),
    )

    plt.savefig(
        outname,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        outname,
    )

    remote_key = remote_png_key(
        init_dt,
        domain_key,
        fhr,
    )

    upload_to_r2(
        outname,
        remote_key,
    )

    return outname


# ============================================================
# PROCESS ONE FORECAST HOUR
# ============================================================

def process_forecast_hour(
    init_dt,
    fhr,
    remote_filename,
    work_root,
    lbf_geom,
):
    missing_domains = missing_domains_for_fhr(
        init_dt,
        fhr,
    )

    if not missing_domains:
        print(
            f"{cycle_id(init_dt)} F{fhr:03d} "
            f"already complete in R2."
        )
        return True

    print(
        f"{cycle_id(init_dt)} F{fhr:03d} "
        f"missing domains: "
        f"{missing_domains}"
    )

    cycle_work = os.path.join(
        work_root,
        cycle_id(init_dt),
    )

    os.makedirs(
        cycle_work,
        exist_ok=True,
    )

    grib_path = os.path.join(
        cycle_work,
        remote_filename,
    )

    try:
        download_ftp_file(
            remote_filename,
            grib_path,
        )

        fields = load_rrfs_fields_once(
            grib_path
        )

        for domain_key in missing_domains:
            cfg = DOMAINS[
                domain_key
            ]

            plot_domain_from_fields(
                fields,
                domain_key,
                cfg,
                init_dt,
                fhr,
                cycle_work,
                lbf_geom,
            )

        # Verify every required domain exists in R2 before
        # considering this FHR complete.
        complete = fhr_complete_in_r2(
            init_dt,
            fhr,
        )

        if complete:
            print(
                f"Successfully completed "
                f"{cycle_id(init_dt)} "
                f"F{fhr:03d}"
            )
        else:
            print(
                f"{cycle_id(init_dt)} "
                f"F{fhr:03d} still incomplete "
                f"after plotting."
            )

        return complete

    finally:
        # Large MPAS GRIBs are deleted immediately after the
        # forecast hour is processed.
        if os.path.exists(
            grib_path
        ):
            try:
                os.remove(
                    grib_path
                )

                print(
                    "Deleted local GRIB:",
                    grib_path,
                )

            except Exception as e:
                print(
                    "Could not delete local GRIB:",
                    e,
                )


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 78)
    print("RRFS-MPAS INCREMENTAL R2 PUBLISHER")
    print("=" * 78)

    print(
        "Selected cycles:",
        sorted(
            TARGET_FHRS.keys()
        ),
    )

    print(
        "00Z/12Z target: F048"
    )

    print(
        "03/06/09/15/18/21Z target: F012"
    )

    # --------------------------------------------------------
    # INVENTORY FTP
    # --------------------------------------------------------

    inventory = list_rrfs_inventory()

    if not inventory:
        print(
            "No eligible RRFS-MPAS cycles "
            "found on FTP."
        )
        return

    cycles = sorted(
        inventory.keys()
    )

    print(
        "\nEligible FTP cycles:"
    )

    for init_dt in cycles:
        available = sorted(
            inventory[init_dt].keys()
        )

        print(
            f"  {cycle_id(init_dt)} | "
            f"target F{target_max_fhr(init_dt):03d} | "
            f"available "
            f"F{available[0]:03d}-"
            f"F{available[-1]:03d} | "
            f"{len(available)} files"
        )

    # --------------------------------------------------------
    # LOAD CWA GEOMETRY ONCE
    # --------------------------------------------------------

    lbf_geom = get_lbf_cwa_geom(
        LBF_CWA_SHP
    )

    # --------------------------------------------------------
    # TEMPORARY WORKSPACE
    # --------------------------------------------------------

    work_root = tempfile.mkdtemp(
        prefix="rrfs_mpas_"
    )

    successful = []
    failed = []

    try:
        # Process oldest -> newest so an unfinished long run
        # gets a chance to advance even when newer cycles exist.
        for init_dt in cycles:
            target = target_max_fhr(
                init_dt
            )

            available_map = inventory[
                init_dt
            ]

            available_fhrs = sorted(
                fhr
                for fhr in available_map
                if fhr <= target
            )

            print("\n" + "=" * 78)
            print(
                f"RUN {cycle_id(init_dt)} | "
                f"target through F{target:03d}"
            )
            print("=" * 78)

            for fhr in available_fhrs:
                if fhr_complete_in_r2(
                    init_dt,
                    fhr,
                ):
                    print(
                        f"Skipping "
                        f"{cycle_id(init_dt)} "
                        f"F{fhr:03d}: "
                        f"already complete."
                    )
                    continue

                fhr_success = False

                for attempt in range(
                    1,
                    MAX_FHR_ATTEMPTS + 1,
                ):
                    try:
                        print("\n" + "-" * 70)
                        print(
                            f"Processing "
                            f"{cycle_id(init_dt)} "
                            f"F{fhr:03d} | "
                            f"attempt {attempt}/"
                            f"{MAX_FHR_ATTEMPTS}"
                        )
                        print("-" * 70)

                        fhr_success = process_forecast_hour(
                            init_dt,
                            fhr,
                            available_map[fhr],
                            work_root,
                            lbf_geom,
                        )

                        if fhr_success:
                            successful.append(
                                (
                                    cycle_id(init_dt),
                                    fhr,
                                )
                            )

                            # Publish the new contiguous state
                            # immediately after each successful
                            # forecast hour. This allows the site
                            # to grow while a long run is still
                            # being generated.
                            upload_runs_json(
                                cycles
                            )

                            break

                    except Exception as e:
                        print(
                            f"{cycle_id(init_dt)} "
                            f"F{fhr:03d} failed on "
                            f"attempt {attempt}/"
                            f"{MAX_FHR_ATTEMPTS}: "
                            f"{e}"
                        )

                    if (
                        attempt
                        < MAX_FHR_ATTEMPTS
                    ):
                        print(
                            f"Waiting "
                            f"{RETRY_WAIT_SECONDS} "
                            f"seconds before retry..."
                        )

                        time.sleep(
                            RETRY_WAIT_SECONDS
                        )

                if not fhr_success:
                    failed.append(
                        (
                            cycle_id(init_dt),
                            fhr,
                        )
                    )

                    print(
                        f"Skipping "
                        f"{cycle_id(init_dt)} "
                        f"F{fhr:03d} after "
                        f"{MAX_FHR_ATTEMPTS} "
                        f"failed attempts."
                    )

        # Always refresh manifest once more at the end,
        # including cases where every PNG already existed.
        upload_runs_json(
            cycles
        )

    finally:
        shutil.rmtree(
            work_root,
            ignore_errors=True,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("RRFS-MPAS PROCESSING SUMMARY")
    print("=" * 78)

    print(
        "Newly completed forecast hours:"
    )

    if successful:
        for rid, fhr in successful:
            print(
                f"  {rid} F{fhr:03d}"
            )
    else:
        print("  None")

    print(
        "\nFailed forecast hours:"
    )

    if failed:
        for rid, fhr in failed:
            print(
                f"  {rid} F{fhr:03d}"
            )
    else:
        print("  None")


if __name__ == "__main__":
    main()
