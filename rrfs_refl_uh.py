# ============================================================
# RRFS | R2 CAMs Product
#
# Reflectivity + UH + Sim IR + Theta Cold Pools
# + 4–6 km Storm-Relative Winds
#
# SOURCE:
# https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/para/
#
# IMPORTANT:
# NOMADS parallel RRFS files are downloaded as full GRIB2 files.
# The needed messages are extracted directly with ecCodes.
# Large GRIB2 files are deleted after each forecast hour.
#
# Uploads:
# runs/cams/rrfs/refl_uh/
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import re
import gc
import json
import time
import zipfile
import requests
import boto3

import numpy as np
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
from scipy.interpolate import griddata

from mpl_toolkits.axes_grid1 import make_axes_locatable

from datetime import (
    datetime,
    timedelta,
    timezone
)

from matplotlib.colors import (
    ListedColormap,
    BoundaryNorm
)

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_array,
    codes_release
)


# ============================================================
# PATHS / ASSETS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


ASSET_DIR = os.path.join(
    BASE_DIR,
    "assets"
)


COUNTY_SHP = os.path.join(
    ASSET_DIR,
    "cb_2018_us_county_500k.shp"
)


STATE_SHP = os.path.join(
    ASSET_DIR,
    "cb_2018_us_state_500k.shp"
)


LBF_CWA_SHP = os.path.join(
    ASSET_DIR,
    "c_18mr25.shp"
)


LOGO_PATH = os.path.join(
    ASSET_DIR,
    "NOAANWSLogos.png"
)


zip_path = os.path.join(
    ASSET_DIR,
    "c_18mr25.zip"
)


if os.path.exists(
    zip_path
):

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            ASSET_DIR
        )


DATA_DIR = os.path.join(
    BASE_DIR,
    "rrfs_full_grib"
)


SECTION_KEY = "cams"

MODEL_KEY = "rrfs"

PRODUCT_KEY = "refl_uh"


R2_PRODUCT_PATH = (
    f"runs/"
    f"{SECTION_KEY}/"
    f"{MODEL_KEY}/"
    f"{PRODUCT_KEY}"
)


OUTDIR_BASE = os.path.join(
    "site",
    "runs",
    SECTION_KEY,
    MODEL_KEY,
    PRODUCT_KEY
)


os.makedirs(
    DATA_DIR,
    exist_ok=True
)


os.makedirs(
    OUTDIR_BASE,
    exist_ok=True
)


# ============================================================
# R2 SETUP
# ============================================================

BUCKET = os.environ[
    "AWS_BUCKET"
]


s3 = boto3.client(
    "s3",

    aws_access_key_id=
        os.environ[
            "AWS_ACCESS_KEY_ID"
        ],

    aws_secret_access_key=
        os.environ[
            "AWS_SECRET_ACCESS_KEY"
        ],

    region_name=
        os.environ[
            "AWS_REGION"
        ],
)


def upload_to_r2(
    local_file,
    remote_key,
    content_type="image/png"
):

    s3.upload_file(
        local_file,
        BUCKET,
        remote_key,

        ExtraArgs={
            "ContentType":
                content_type
        }
    )


    print(
        "Uploaded to R2:",
        remote_key
    )


# ============================================================
# DOMAIN CONFIG
# ============================================================

DOMAINS = {

    "lbf": {

        "label":
            "LBF",

        "extent":
            [
                -103.8,
                -97.0,
                40.0,
                43.4
            ],

        "title_size":
            14,

        "subtitle_size":
            11,

        "barb_skip":
            11,

    },


    "regional": {

        "label":
            "Default",

        "extent":
            [
                -107.5,
                -93.0,
                38.5,
                44.2
            ],

        "title_size":
            13,

        "subtitle_size":
            11,

        "barb_skip":
            20,

    },


    "central_plains": {

        "label":
            "Central Plains",

        "extent":
            [
                -107.5,
                -91.0,
                34.5,
                45.2
            ],

        "title_size":
            13,

        "subtitle_size":
            11,

        "barb_skip":
            24,

    },

}


# ============================================================
# DYNAMIC SPC SEVERE DOMAIN
# ============================================================

SPC_DAY1_CAT_URL = (
    "https://mapservices.weather.noaa.gov/"
    "vector/rest/services/outlooks/"
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

        "where":
            "1=1",

        "outFields":
            "*",

        "f":
            "geojson",

        "returnGeometry":
            "true",

        "outSR":
            "4326",

    }


    r = requests.get(
        SPC_DAY1_CAT_URL,
        params=params,
        timeout=30
    )


    r.raise_for_status()


    data = r.json()


    if (
        "features"
        not in data
        or
        len(
            data[
                "features"
            ]
        )
        ==
        0
    ):

        raise RuntimeError(
            "SPC query returned no features."
        )


    return gpd.GeoDataFrame.from_features(
        data[
            "features"
        ],
        crs="EPSG:4326"
    )


def add_spc_severe_domain():

    try:

        gdf = (
            fetch_spc_day1_geojson()
            .to_crs(
                epsg=4326
            )
        )


        risk_col = None


        for col in gdf.columns:

            vals = (
                gdf[
                    col
                ]
                .astype(
                    str
                )
                .str
                .upper()
            )


            if (
                vals
                .isin(
                    SPC_RISK_ORDER.keys()
                )
                .any()
            ):

                risk_col = col

                break


        if risk_col is None:

            print(
                "SPC severe domain skipped: "
                "could not find risk category column."
            )

            return


        gdf[
            "risk"
        ] = (
            gdf[
                risk_col
            ]
            .astype(str)
            .str
            .upper()
        )


        gdf[
            "risk_rank"
        ] = (
            gdf[
                "risk"
            ]
            .map(
                SPC_RISK_ORDER
            )
        )


        severe = gdf[
            gdf[
                "risk_rank"
            ]
            >=
            SPC_RISK_ORDER[
                MIN_SPC_RISK
            ]
        ].copy()


        if severe.empty:

            print(
                "SPC severe domain skipped: "
                "no SLGT+ risk found."
            )

            return


        highest_rank = (
            severe[
                "risk_rank"
            ]
            .max()
        )


        highest = severe[
            severe[
                "risk_rank"
            ]
            ==
            highest_rank
        ].copy()


        highest_proj = (
            highest
            .to_crs(
                epsg=5070
            )
        )


        highest[
            "_area"
        ] = (
            highest_proj
            .geometry
            .area
            .values
        )


        main_poly = highest.loc[
            highest[
                "_area"
            ]
            .idxmax()
        ]


        highest_label = (
            main_poly[
                "risk"
            ]
        )


        main_gdf = gpd.GeoDataFrame(
            [
                main_poly
            ],
            geometry="geometry",
            crs="EPSG:4326"
        )


        centroid_proj = (
            main_gdf
            .to_crs(
                epsg=5070
            )
            .geometry
            .centroid
        )


        centroid_ll = (
            gpd.GeoSeries(
                centroid_proj,
                crs="EPSG:5070"
            )
            .to_crs(
                epsg=4326
            )
            .iloc[0]
        )


        center_lon = centroid_ll.x
        center_lat = centroid_ll.y


        extent = [

            center_lon
            -
            SEVERE_DOMAIN_WIDTH / 2,

            center_lon
            +
            SEVERE_DOMAIN_WIDTH / 2,

            center_lat
            -
            SEVERE_DOMAIN_HEIGHT / 2,

            center_lat
            +
            SEVERE_DOMAIN_HEIGHT / 2,

        ]


        DOMAINS[
            "spc_severe"
        ] = {

            "label":
                f"SPC {highest_label} Risk",

            "extent":
                extent,

            "title_size":
                13,

            "subtitle_size":
                11,

            "barb_skip":
                22,

        }


        print(
            f"Added SPC severe domain: "
            f"{highest_label}"
        )


        print(
            f"SPC severe extent: "
            f"{extent}"
        )


    except Exception as e:

        print(
            f"SPC severe domain skipped "
            f"due to error: {e}"
        )


add_spc_severe_domain()


# ============================================================
# SETTINGS
# ============================================================

VALID_RRFS_CYCLES = list(
    range(24)
)


START_FHR = 1


CYCLE_DELAY_MINUTES = 45


LONG_CYCLE_HOURS = [
    0,
    6,
    12,
    18
]


MAX_FHR_LONG = 60

MAX_FHR_SHORT = 18


MANUAL_STORM_MOTION_FROM_DEG = 250

MANUAL_STORM_MOTION_SPEED_KT = 35


PLOT_SR_WIND_BARBS = True


# ------------------------------------------------------------
# DOWNLOAD SETTINGS
# ------------------------------------------------------------

DOWNLOAD_ATTEMPTS = 3

DOWNLOAD_RETRY_SECONDS = 15


# Remove large NOMADS GRIB files after each forecast hour.
DELETE_GRIB_AFTER_HOUR = True


# ============================================================
# RRFS NOMADS BASE
# ============================================================

RRFS_NOMADS_BASE = (
    "https://nomads.ncep.noaa.gov/"
    "pub/data/nccf/com/rrfs/para"
)


# ============================================================
# REFLECTIVITY COLOR TABLE
# ============================================================

bounds = [

    0,
    10,
    12.5,
    15,
    17.5,
    20,
    22.5,
    25,
    27.5,
    30,

    32.5,
    35,
    37.5,
    40,
    42.5,
    45,
    47.5,
    50,
    52.5,
    55,

    57.5,
    60,
    62.5,
    65,
    67.5,
    70,
    72.5

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
    "#e6e6e6"

]


cmap = ListedColormap(
    colors,
    name="reflec_bins"
)


norm = BoundaryNorm(
    bounds,
    cmap.N,
    clip=True
)


REF_LEVELS = [

    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75

]


# ============================================================
# BASIC HELPERS
# ============================================================

def to_lon180(
    lon
):

    return (
        (
            np.asarray(
                lon
            )
            +
            180
        )
        %
        360
    ) - 180


def k_to_c(
    k
):

    return (
        np.asarray(
            k
        )
        -
        273.15
    )


def kt_to_ms(
    kt
):

    return (
        kt
        *
        0.514444
    )


def ms_to_kt(
    ms
):

    return (
        ms
        *
        1.94384
    )


def wind_from_dir_speed_to_uv(
    direction_from_deg,
    speed_ms
):

    rad = np.deg2rad(
        direction_from_deg
    )


    u = (
        -speed_ms
        *
        np.sin(
            rad
        )
    )


    v = (
        -speed_ms
        *
        np.cos(
            rad
        )
    )


    return (
        u,
        v
    )


# ============================================================
# RRFS URL BUILDER
# ============================================================

def rrfs_grib_url(
    init_dt,
    fhr,
    product="2dfld"
):

    ymd = init_dt.strftime(
        "%Y%m%d"
    )


    hh = init_dt.strftime(
        "%H"
    )


    if product == "2dfld":

        fname = (
            f"rrfs.t{hh}z."
            f"2dfld.3km."
            f"f{fhr:03d}."
            f"conus.grib2"
        )


    elif product == "prslev":

        fname = (
            f"rrfs.t{hh}z."
            f"prslev.3km."
            f"f{fhr:03d}."
            f"conus.grib2"
        )


    else:

        raise ValueError(
            "product must be "
            "'2dfld' or 'prslev'"
        )


    return (
        f"{RRFS_NOMADS_BASE}/"
        f"rrfs.{ymd}/"
        f"{hh}/"
        f"{fname}"
    )


# ============================================================
# URL EXISTS
#
# Opens the request as a stream and immediately closes it.
# It does NOT download the complete GRIB.
# ============================================================

def url_exists(
    url,
    timeout=15
):

    try:

        with requests.get(
            url,
            stream=True,
            timeout=timeout
        ) as r:

            return (
                r.status_code
                ==
                200
            )


    except Exception:

        return False


# ============================================================
# FIND LATEST RRFS CYCLE
# ============================================================

def find_latest_available_rrfs_cycle(
    max_back_hours=48
):

    now = (
        datetime.now(
            timezone.utc
        )
        -
        timedelta(
            minutes=
                CYCLE_DELAY_MINUTES
        )
    )


    print("")
    print("=" * 70)
    print(
        "SEARCHING FOR LATEST "
        "RRFS PARALLEL CYCLE"
    )
    print("=" * 70)


    for back in range(
        max_back_hours + 1
    ):

        dt = (
            now
            -
            timedelta(
                hours=back
            )
        )


        if (
            dt.hour
            not in
            VALID_RRFS_CYCLES
        ):

            continue


        dt = dt.replace(
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=None
        )


        test_url = rrfs_grib_url(
            dt,
            1,
            product="2dfld"
        )


        print(
            f"Checking RRFS "
            f"{dt:%Y%m%d %HZ}..."
        )


        if url_exists(
            test_url
        ):

            print("")
            print(
                f"Latest RRFS cycle found: "
                f"{dt:%Y%m%d %HZ}"
            )


            print(
                "Matched GRIB:",
                test_url
            )


            return dt


    raise RuntimeError(
        "Could not find recent "
        "RRFS parallel cycle."
    )


# ============================================================
# FULL GRIB DOWNLOAD
# ============================================================

def download_full_grib(
    init_dt,
    fhr,
    product
):

    url = rrfs_grib_url(
        init_dt,
        fhr,
        product=product
    )


    filename = os.path.basename(
        url
    )


    outpath = os.path.join(
        DATA_DIR,
        filename
    )


    if (
        os.path.exists(
            outpath
        )
        and
        os.path.getsize(
            outpath
        )
        >
        1_000_000
    ):

        print(
            "Using cached GRIB:",
            outpath
        )


        return outpath


    last_error = None


    for attempt in range(
        1,
        DOWNLOAD_ATTEMPTS + 1
    ):

        try:

            print("")
            print(
                f"Downloading {product} "
                f"F{fhr:03d}"
            )


            print(
                url
            )


            print(
                f"Attempt "
                f"{attempt}/"
                f"{DOWNLOAD_ATTEMPTS}"
            )


            with requests.get(
                url,
                stream=True,
                timeout=180
            ) as r:

                r.raise_for_status()


                total_bytes = 0


                with open(
                    outpath,
                    "wb"
                ) as f:

                    for chunk in r.iter_content(
                        chunk_size=
                            8
                            *
                            1024
                            *
                            1024
                    ):

                        if not chunk:

                            continue


                        f.write(
                            chunk
                        )


                        total_bytes += len(
                            chunk
                        )


                        if (
                            total_bytes
                            %
                            (
                                100
                                *
                                1024
                                *
                                1024
                            )
                            <
                            len(
                                chunk
                            )
                        ):

                            print(
                                f"  "
                                f"{total_bytes / 1024 / 1024:.0f} MB"
                            )


            if (
                not os.path.exists(
                    outpath
                )
                or
                os.path.getsize(
                    outpath
                )
                <
                1_000_000
            ):

                raise RuntimeError(
                    "Downloaded GRIB "
                    "appears incomplete."
                )


            print(
                "Downloaded:",
                outpath
            )


            print(
                f"Size: "
                f"{os.path.getsize(outpath) / 1024 / 1024:.1f} MB"
            )


            return outpath


        except Exception as e:

            last_error = e


            print(
                f"Download failed: "
                f"{e}"
            )


            if os.path.exists(
                outpath
            ):

                try:

                    os.remove(
                        outpath
                    )

                except Exception:

                    pass


            if (
                attempt
                <
                DOWNLOAD_ATTEMPTS
            ):

                print(
                    f"Retrying in "
                    f"{DOWNLOAD_RETRY_SECONDS} seconds..."
                )


                time.sleep(
                    DOWNLOAD_RETRY_SECONDS
                )


    raise RuntimeError(
        f"Could not download "
        f"{product} F{fhr:03d}. "
        f"Last error: {last_error}"
    )


# ============================================================
# SAFE ECCODES GET
# ============================================================

def safe_codes_get(
    gid,
    key,
    default=""
):

    try:

        return codes_get(
            gid,
            key
        )


    except Exception:

        return default


# ============================================================
# BUILD SEARCHABLE GRIB MESSAGE DESCRIPTION
#
# This lets us retain search terms similar to the old IDX
# workflow even though we're scanning GRIB messages directly.
# ============================================================

def grib_message_description(
    gid
):

    short_name = str(
        safe_codes_get(
            gid,
            "shortName",
            ""
        )
    )


    name = str(
        safe_codes_get(
            gid,
            "name",
            ""
        )
    )


    parameter_name = str(
        safe_codes_get(
            gid,
            "parameterName",
            ""
        )
    )


    type_level = str(
        safe_codes_get(
            gid,
            "typeOfLevel",
            ""
        )
    )


    level = safe_codes_get(
        gid,
        "level",
        ""
    )


    top_level = safe_codes_get(
        gid,
        "topLevel",
        ""
    )


    bottom_level = safe_codes_get(
        gid,
        "bottomLevel",
        ""
    )


    step_type = str(
        safe_codes_get(
            gid,
            "stepType",
            ""
        )
    )


    pieces = [

        short_name,
        short_name.upper(),

        name,
        parameter_name,

        type_level,

        str(
            level
        ),

        str(
            top_level
        ),

        str(
            bottom_level
        ),

        step_type,

    ]


    # --------------------------------------------------------
    # ADD OLD-GRIB/IDX STYLE ALIASES
    # --------------------------------------------------------

    s = short_name.lower()


    if s in (
        "u",
        "ugrd"
    ):

        pieces.extend(
            [
                "UGRD",
                "U component of wind"
            ]
        )


    if s in (
        "v",
        "vgrd"
    ):

        pieces.extend(
            [
                "VGRD",
                "V component of wind"
            ]
        )


    if s in (
        "t",
        "2t",
        "tmp"
    ):

        pieces.extend(
            [
                "TMP",
                "temperature"
            ]
        )


    if s in (
        "sp",
        "pres",
        "prmsl"
    ):

        pieces.extend(
            [
                "PRES",
                "pressure"
            ]
        )


    if (
        "refd"
        in s
    ):

        pieces.append(
            "REFD"
        )


    if (
        "refc"
        in s
    ):

        pieces.append(
            "REFC"
        )


    if (
        "mxuphl"
        in s
        or
        "updraft helicity"
        in name.lower()
    ):

        pieces.append(
            "MXUPHL"
        )


    # --------------------------------------------------------
    # HUMAN-READABLE LEVEL ALIASES
    # --------------------------------------------------------

    if (
        type_level
        ==
        "heightAboveGround"
    ):

        pieces.extend(
            [
                f"{level} m",
                f"{level} m above ground"
            ]
        )


    if (
        "isobaric"
        in
        type_level.lower()
    ):

        pieces.extend(
            [
                f"{level} mb",
                f"{level} hPa"
            ]
        )


    # Layer aliases helpful for UH
    if (
        top_level != ""
        and
        bottom_level != ""
    ):

        pieces.extend(
            [
                f"{top_level}-{bottom_level}",
                f"{bottom_level}-{top_level}",
                f"{top_level} - {bottom_level}",
                f"{bottom_level} - {top_level}"
            ]
        )


    return " | ".join(
        str(
            x
        )
        for x
        in pieces
    )


# ============================================================
# MATCH OLD TERM SETS AGAINST A GRIB MESSAGE
# ============================================================

def matches_term_set(
    description,
    terms
):

    desc_lower = (
        description
        .lower()
    )


    return all(
        term.lower()
        in
        desc_lower

        for term
        in terms
    )


# ============================================================
# DETERMINE GRIB GRID SHAPE
# ============================================================

def get_grib_shape(
    gid,
    values
):

    nx = safe_codes_get(
        gid,
        "Nx",
        None
    )


    ny = safe_codes_get(
        gid,
        "Ny",
        None
    )


    if (
        nx in (
            None,
            "",
            0
        )
        or
        ny in (
            None,
            "",
            0
        )
    ):

        nx = safe_codes_get(
            gid,
            "Ni",
            None
        )


        ny = safe_codes_get(
            gid,
            "Nj",
            None
        )


    if (
        nx in (
            None,
            "",
            0
        )
        or
        ny in (
            None,
            "",
            0
        )
    ):

        raise RuntimeError(
            "Could not determine "
            "GRIB grid dimensions."
        )


    nx = int(
        nx
    )


    ny = int(
        ny
    )


    if (
        nx
        *
        ny
        !=
        len(
            values
        )
    ):

        raise RuntimeError(
            f"GRIB dimensions "
            f"{ny}x{nx} do not equal "
            f"value count {len(values)}."
        )


    return (
        ny,
        nx
    )


# ============================================================
# EXTRACT ARRAY + LAT/LON FROM MESSAGE
# ============================================================

def message_to_arrays(
    gid
):

    values = np.asarray(
        codes_get_array(
            gid,
            "values"
        ),
        dtype=float
    )


    latitudes = np.asarray(
        codes_get_array(
            gid,
            "latitudes"
        ),
        dtype=float
    )


    longitudes = np.asarray(
        codes_get_array(
            gid,
            "longitudes"
        ),
        dtype=float
    )


    (
        ny,
        nx
    ) = get_grib_shape(
        gid,
        values
    )


    field = values.reshape(
        ny,
        nx
    )


    lat = latitudes.reshape(
        ny,
        nx
    )


    lon = longitudes.reshape(
        ny,
        nx
    )


    lon = to_lon180(
        lon
    )


    return (
        field,
        lat,
        lon
    )


# ============================================================
# FIELD SEARCH CONFIGURATION
# ============================================================

TWO_D_FIELD_SEARCHES = {

    "reflectivity": [

        [
            "REFD",
            "1000 m"
        ],

        [
            "REFC"
        ],

        [
            "REFD"
        ],

    ],


    "uh25": [

        [
            "MXUPHL",
            "5000-2000"
        ],

        [
            "MXUPHL",
            "5000 - 2000"
        ],

        [
            "MXUPHL"
        ],

    ],


    "uh03": [

        [
            "MXUPHL",
            "3000-0"
        ],

        [
            "MXUPHL",
            "3000 - 0"
        ],

    ],


    "sim_ir": [

        [
            "SBT123"
        ],

        [
            "SBT124"
        ],

        [
            "brightness"
        ],

        [
            "satellite"
        ],

    ],


    "t2": [

        [
            "TMP",
            "2 m above ground"
        ],

        [
            "TMP",
            "2 m"
        ],

        [
            "2 metre temperature"
        ],

    ],


    "psfc": [

        [
            "PRES",
            "surface"
        ],

        [
            "surface pressure"
        ],

    ],


    "u_stm": [

        [
            "UEID"
        ],

        [
            "USTM"
        ],

        [
            "BUNK",
            "U"
        ],

    ],


    "v_stm": [

        [
            "VEID"
        ],

        [
            "VSTM"
        ],

        [
            "BUNK",
            "V"
        ],

    ],

}


PRESSURE_FIELD_SEARCHES = {

    "u700": [

        [
            "UGRD",
            "700 mb"
        ],

    ],


    "v700": [

        [
            "VGRD",
            "700 mb"
        ],

    ],


    "u600": [

        [
            "UGRD",
            "600 mb"
        ],

    ],


    "v600": [

        [
            "VGRD",
            "600 mb"
        ],

    ],


    "u500": [

        [
            "UGRD",
            "500 mb"
        ],

    ],


    "v500": [

        [
            "VGRD",
            "500 mb"
        ],

    ],

}


# ============================================================
# SCAN ONE GRIB AND EXTRACT ALL REQUESTED FIELDS IN ONE PASS
# ============================================================

def extract_fields_from_grib(
    path,
    searches,
    optional_fields=None
):

    if optional_fields is None:

        optional_fields = set()


    results = {}


    print("")
    print(
        "Scanning GRIB:",
        path
    )


    with open(
        path,
        "rb"
    ) as f:

        message_number = 0


        while True:

            gid = codes_grib_new_from_file(
                f
            )


            if gid is None:

                break


            message_number += 1


            try:

                description = (
                    grib_message_description(
                        gid
                    )
                )


                # ---------------------------------------------
                # ONLY TEST FIELDS THAT HAVE NOT BEEN FOUND
                # ---------------------------------------------

                for field_name, term_sets in searches.items():

                    if (
                        field_name
                        in
                        results
                    ):

                        continue


                    matched = False


                    for terms in term_sets:

                        if matches_term_set(
                            description,
                            terms
                        ):

                            (
                                field,
                                lat,
                                lon
                            ) = message_to_arrays(
                                gid
                            )


                            results[
                                field_name
                            ] = {

                                "field":
                                    field,

                                "lat":
                                    lat,

                                "lon":
                                    lon,

                                "description":
                                    description,

                            }


                            print("")
                            print(
                                f"Matched "
                                f"{field_name}:"
                            )


                            print(
                                description
                            )


                            matched = True

                            break


                    if matched:

                        continue


                # ---------------------------------------------
                # STOP EARLY IF EVERYTHING IS FOUND
                # ---------------------------------------------

                if (
                    len(
                        results
                    )
                    ==
                    len(
                        searches
                    )
                ):

                    break


            finally:

                codes_release(
                    gid
                )


    # ========================================================
    # REPORT MISSING FIELDS
    # ========================================================

    missing_required = []


    for field_name in searches:

        if (
            field_name
            not in
            results
        ):

            if (
                field_name
                in
                optional_fields
            ):

                print(
                    f"Optional field "
                    f"{field_name} not found."
                )


            else:

                missing_required.append(
                    field_name
                )


    if missing_required:

        raise RuntimeError(
            "Required RRFS fields not found: "
            +
            ", ".join(
                missing_required
            )
        )


    return results


# ============================================================
# SPATIAL HELPERS
# ============================================================

def subset_2d(
    lat,
    lon,
    *fields
):

    mask = (
        np.isfinite(
            lat
        )
        &
        np.isfinite(
            lon
        )
        &
        (
            lon >= LON_MIN
        )
        &
        (
            lon <= LON_MAX
        )
        &
        (
            lat >= LAT_MIN
        )
        &
        (
            lat <= LAT_MAX
        )
    )


    if not np.any(
        mask
    ):

        raise RuntimeError(
            "No grid points found "
            "inside selected domain."
        )


    iy, ix = np.where(
        mask
    )


    iy0 = max(
        iy.min() - 2,
        0
    )


    iy1 = min(
        iy.max() + 3,
        lat.shape[
            0
        ]
    )


    ix0 = max(
        ix.min() - 2,
        0
    )


    ix1 = min(
        ix.max() + 3,
        lon.shape[
            1
        ]
    )


    return (

        lat[
            iy0:iy1,
            ix0:ix1
        ],

        lon[
            iy0:iy1,
            ix0:ix1
        ],

        [
            field[
                iy0:iy1,
                ix0:ix1
            ]

            for field
            in fields
        ]

    )


def interp_to_target_grid(
    src_lat,
    src_lon,
    src_field,
    tgt_lat,
    tgt_lon
):

    src_points = np.column_stack(
        (
            src_lon.ravel(),
            src_lat.ravel()
        )
    )


    src_values = np.asarray(
        src_field
    ).ravel()


    good = (
        np.isfinite(
            src_points[
                :,
                0
            ]
        )
        &
        np.isfinite(
            src_points[
                :,
                1
            ]
        )
        &
        np.isfinite(
            src_values
        )
    )


    out = griddata(
        src_points[
            good
        ],
        src_values[
            good
        ],
        (
            tgt_lon,
            tgt_lat
        ),
        method="linear"
    )


    if np.isnan(
        out
    ).any():

        out_nearest = griddata(
            src_points[
                good
            ],
            src_values[
                good
            ],
            (
                tgt_lon,
                tgt_lat
            ),
            method="nearest"
        )


        out = np.where(
            np.isnan(
                out
            ),
            out_nearest,
            out
        )


    return out


# ============================================================
# SHAPEFILE HELPERS
# ============================================================

def add_shapefile_outline(
    ax,
    shp_path,
    edgecolor="k",
    linewidth=1.2,
    zorder=6
):

    if not os.path.exists(
        shp_path
    ):

        print(
            "Missing shapefile:",
            shp_path
        )

        return


    gdf = (
        gpd
        .read_file(
            shp_path
        )
        .to_crs(
            epsg=4326
        )
    )


    gdf = gdf.cx[
        LON_MIN - 1:
        LON_MAX + 1,

        LAT_MIN - 1:
        LAT_MAX + 1
    ]


    ax.add_geometries(
        gdf.geometry,

        crs=
            ccrs.PlateCarree(),

        facecolor=
            "none",

        edgecolor=
            edgecolor,

        linewidth=
            linewidth,

        zorder=
            zorder
    )


def get_lbf_cwa_geom(
    cwa_shp_path
):

    if not os.path.exists(
        cwa_shp_path
    ):

        print(
            "Missing LBF CWA shapefile:",
            cwa_shp_path
        )

        return None


    reader = shpreader.Reader(
        cwa_shp_path
    )


    recs = list(
        reader.records()
    )


    geoms = [

        r.geometry

        for r in recs

        if (
            str(
                r.attributes.get(
                    "CWA",
                    ""
                )
            )
            .upper()
            ==
            "LBF"
        )

        or

        (
            str(
                r.attributes.get(
                    "WFO",
                    ""
                )
            )
            .upper()
            ==
            "LBF"
        )

    ]


    if not geoms:

        geoms = [
            r.geometry
            for r
            in recs
        ]


    return unary_union(
        geoms
    )


def add_counties_clipped_to_cwa(
    ax,
    counties_shp_path,
    cwa_geom,
    lw=1.0,
    color="black",
    zorder=6
):

    if (
        cwa_geom is None
        or
        not os.path.exists(
            counties_shp_path
        )
    ):

        return


    reader = shpreader.Reader(
        counties_shp_path
    )


    cwa_p = prep(
        cwa_geom
    )


    clipped = []


    for r in reader.records():

        g = r.geometry


        if cwa_p.intersects(
            g
        ):

            inter = g.intersection(
                cwa_geom
            )


            if not inter.is_empty:

                clipped.append(
                    inter
                )


    ax.add_geometries(
        clipped,

        crs=
            ccrs.PlateCarree(),

        facecolor=
            "none",

        edgecolor=
            color,

        linewidth=
            lw,

        zorder=
            zorder
    )


# ============================================================
# RUNS.JSON
# ============================================================

def upload_runs_json(
    init_dt,
    cycle_str,
    max_fhr
):

    old_runs = []


    try:

        obj = s3.get_object(
            Bucket=
                BUCKET,

            Key=
                (
                    f"{R2_PRODUCT_PATH}/"
                    f"runs.json"
                )
        )


        old_data = json.loads(
            obj[
                "Body"
            ]
            .read()
            .decode(
                "utf-8"
            )
        )


        old_runs = old_data.get(
            "runs",
            []
        )


    except Exception:

        old_runs = []


    new_run = {

        "id":
            cycle_str,

        "label":
            init_dt.strftime(
                "%Y-%m-%d %Hz"
            ),

        "max_fhr":
            max_fhr,

    }


    combined = [
        new_run
    ]


    for r in old_runs:

        if isinstance(
            r,
            str
        ):

            rid = r


            try:

                rhour = int(
                    rid
                    .split(
                        "_"
                    )[
                        1
                    ]
                    .replace(
                        "z",
                        ""
                    )
                )


            except Exception:

                rhour = 0


            combined.append({

                "id":
                    rid,

                "label":
                    rid.replace(
                        "_",
                        " "
                    ),

                "max_fhr":
                    (
                        60
                        if
                        rhour
                        in
                        LONG_CYCLE_HOURS
                        else
                        18
                    ),

            })


        elif (
            r.get(
                "id"
            )
            !=
            cycle_str
        ):

            combined.append(
                r
            )


    runs_json = {

        "runs":
            combined[
                :4
            ]

    }


    with open(
        "runs.json",
        "w"
    ) as f:

        json.dump(
            runs_json,
            f,
            indent=2
        )


    upload_to_r2(
        "runs.json",

        (
            f"{R2_PRODUCT_PATH}/"
            f"runs.json"
        ),

        content_type=
            "application/json"
    )


    print(
        "Uploaded runs.json "
        "with last 4 RRFS runs."
    )


# ============================================================
# FIND RRFS INIT
# ============================================================

init_dt = (
    find_latest_available_rrfs_cycle()
)


cycle_str = (
    init_dt.strftime(
        "%Y%m%d_%Hz"
    )
)


if (
    init_dt.hour
    in
    LONG_CYCLE_HOURS
):

    MAX_FHR = (
        MAX_FHR_LONG
    )


else:

    MAX_FHR = (
        MAX_FHR_SHORT
    )


upload_runs_json(
    init_dt,
    cycle_str,
    MAX_FHR
)


OUTDIR = os.path.join(
    OUTDIR_BASE,
    cycle_str
)


os.makedirs(
    OUTDIR,
    exist_ok=True
)


fhrs = range(
    START_FHR,
    MAX_FHR + 1
)


print("")
print("=" * 70)


print(
    "Using RRFS init:",
    init_dt.strftime(
        "%Y-%m-%d %HZ"
    )
)


print(
    "Forecast hours:",
    list(
        fhrs
    )
)


print(
    "Output directory:",
    OUTDIR
)


print(
    "Domains:",
    list(
        DOMAINS.keys()
    )
)


print("=" * 70)


lbf_geom = get_lbf_cwa_geom(
    LBF_CWA_SHP
)


# ============================================================
# LOAD ALL RRFS FIELDS FOR ONE FORECAST HOUR
# ============================================================

def load_rrfs_fields_once(
    fhr
):

    print("")
    print("=" * 70)


    print(
        f"Loading RRFS | "
        f"Init "
        f"{init_dt:%Y-%m-%d %HZ} | "
        f"F{fhr:03d}"
    )


    print("=" * 70)


    two_d_path = None

    prs_path = None


    try:

        # ====================================================
        # DOWNLOAD FULL 2DFLD GRIB
        # ====================================================

        two_d_path = download_full_grib(
            init_dt,
            fhr,
            "2dfld"
        )


        # ====================================================
        # EXTRACT ALL 2-D FIELDS IN ONE PASS
        # ====================================================

        two_d = extract_fields_from_grib(

            two_d_path,

            TWO_D_FIELD_SEARCHES,

            optional_fields={
                "uh25",
                "uh03",
                "sim_ir",
                "u_stm",
                "v_stm"
            }
        )


        # ====================================================
        # REQUIRED MAIN GRID
        # ====================================================

        refl = np.asarray(
            two_d[
                "reflectivity"
            ][
                "field"
            ],
            dtype=float
        )


        lat = np.asarray(
            two_d[
                "reflectivity"
            ][
                "lat"
            ],
            dtype=float
        )


        lon = np.asarray(
            two_d[
                "reflectivity"
            ][
                "lon"
            ],
            dtype=float
        )


        refl = np.where(
            refl
            >=
            REF_LEVELS[
                0
            ],
            refl,
            np.nan
        )


        # ====================================================
        # OPTIONAL UH
        # ====================================================

        if (
            "uh25"
            in
            two_d
        ):

            uh25 = np.asarray(
                two_d[
                    "uh25"
                ][
                    "field"
                ],
                dtype=float
            )


        else:

            uh25 = np.full_like(
                refl,
                np.nan
            )


        if (
            "uh03"
            in
            two_d
        ):

            uh03 = np.asarray(
                two_d[
                    "uh03"
                ][
                    "field"
                ],
                dtype=float
            )


        else:

            uh03 = np.full_like(
                refl,
                np.nan
            )


        # ====================================================
        # OPTIONAL SIM IR
        # ====================================================

        if (
            "sim_ir"
            in
            two_d
        ):

            ir_c = k_to_c(
                two_d[
                    "sim_ir"
                ][
                    "field"
                ]
            )


        else:

            ir_c = np.full_like(
                refl,
                np.nan
            )


        # ====================================================
        # 2-M TEMP / SURFACE PRESSURE
        # ====================================================

        t2_k = np.asarray(
            two_d[
                "t2"
            ][
                "field"
            ],
            dtype=float
        )


        ps_pa = np.asarray(
            two_d[
                "psfc"
            ][
                "field"
            ],
            dtype=float
        )


        # ====================================================
        # THETA COLD POOL
        # ====================================================

        theta = (
            t2_k
            *
            (
                100000.0
                /
                ps_pa
            )**0.286
        )


        theta_bg = gaussian_filter(
            theta,
            sigma=18
        )


        theta_prime = (
            theta
            -
            theta_bg
        )


        # ====================================================
        # DOWNLOAD PRESSURE-LEVEL GRIB
        # ====================================================

        prs_path = download_full_grib(
            init_dt,
            fhr,
            "prslev"
        )


        # ====================================================
        # EXTRACT 700/600/500 WINDS IN ONE PASS
        # ====================================================

        prs = extract_fields_from_grib(
            prs_path,
            PRESSURE_FIELD_SEARCHES
        )


        pr_lat = np.asarray(
            prs[
                "u700"
            ][
                "lat"
            ],
            dtype=float
        )


        pr_lon = np.asarray(
            prs[
                "u700"
            ][
                "lon"
            ],
            dtype=float
        )


        u46_pr = np.nanmean(

            np.stack(
                [

                    prs[
                        "u700"
                    ][
                        "field"
                    ],

                    prs[
                        "u600"
                    ][
                        "field"
                    ],

                    prs[
                        "u500"
                    ][
                        "field"
                    ],

                ]
            ),

            axis=0

        )


        v46_pr = np.nanmean(

            np.stack(
                [

                    prs[
                        "v700"
                    ][
                        "field"
                    ],

                    prs[
                        "v600"
                    ][
                        "field"
                    ],

                    prs[
                        "v500"
                    ][
                        "field"
                    ],

                ]
            ),

            axis=0

        )


        # ====================================================
        # INTERPOLATE PRESSURE WINDS TO 2DFLD GRID
        # ====================================================

        u46_native = interp_to_target_grid(
            pr_lat,
            pr_lon,
            u46_pr,
            lat,
            lon
        )


        v46_native = interp_to_target_grid(
            pr_lat,
            pr_lon,
            v46_pr,
            lat,
            lon
        )


        # ====================================================
        # STORM MOTION
        # ====================================================

        if (
            "u_stm"
            in
            two_d
            and
            "v_stm"
            in
            two_d
        ):

            print(
                "Using RRFS storm motion."
            )


            stm_lat = np.asarray(
                two_d[
                    "u_stm"
                ][
                    "lat"
                ],
                dtype=float
            )


            stm_lon = np.asarray(
                two_d[
                    "u_stm"
                ][
                    "lon"
                ],
                dtype=float
            )


            u_stm_native = interp_to_target_grid(
                stm_lat,
                stm_lon,

                two_d[
                    "u_stm"
                ][
                    "field"
                ],

                lat,
                lon
            )


            v_stm_native = interp_to_target_grid(
                stm_lat,
                stm_lon,

                two_d[
                    "v_stm"
                ][
                    "field"
                ],

                lat,
                lon
            )


            sr_u46 = (
                u46_native
                -
                u_stm_native
            )


            sr_v46 = (
                v46_native
                -
                v_stm_native
            )


            storm_motion_source = (
                "RRFS storm motion"
            )


        else:

            print(
                "RRFS storm motion not found. "
                "Using manual storm motion."
            )


            (
                storm_u_scalar,
                storm_v_scalar

            ) = wind_from_dir_speed_to_uv(
                MANUAL_STORM_MOTION_FROM_DEG,

                kt_to_ms(
                    MANUAL_STORM_MOTION_SPEED_KT
                )
            )


            sr_u46 = (
                u46_native
                -
                np.full_like(
                    refl,
                    storm_u_scalar
                )
            )


            sr_v46 = (
                v46_native
                -
                np.full_like(
                    refl,
                    storm_v_scalar
                )
            )


            storm_motion_source = (
                "Manual storm motion"
            )


        # ====================================================
        # RETURN
        # ====================================================

        return {

            "lat":
                lat,

            "lon":
                lon,

            "refl":
                refl,

            "uh25":
                uh25,

            "uh03":
                uh03,

            "ir_c":
                ir_c,

            "theta_prime":
                theta_prime,

            "sr_u46":
                sr_u46,

            "sr_v46":
                sr_v46,

            "storm_motion_source":
                storm_motion_source,

        }


    finally:

        # ====================================================
        # DELETE HUGE GRIBS
        # ====================================================

        if DELETE_GRIB_AFTER_HOUR:

            for path in (
                two_d_path,
                prs_path
            ):

                if (
                    path
                    and
                    os.path.exists(
                        path
                    )
                ):

                    try:

                        os.remove(
                            path
                        )


                        print(
                            "Deleted large GRIB:",
                            path
                        )


                    except Exception as e:

                        print(
                            f"Could not delete "
                            f"{path}: {e}"
                        )


        gc.collect()


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_domain_from_fields(
    fields,
    domain_key,
    cfg,
    fhr
):

    global LON_MIN
    global LON_MAX
    global LAT_MIN
    global LAT_MAX


    (
        LON_MIN,
        LON_MAX,
        LAT_MIN,
        LAT_MAX

    ) = cfg[
        "extent"
    ]


    domain_outdir = os.path.join(
        OUTDIR,
        domain_key
    )


    os.makedirs(
        domain_outdir,
        exist_ok=True
    )


    print(
        f"Plotting "
        f"{domain_key.upper()} | "
        f"F{fhr:03d}"
    )


    try:

        lat = fields[
            "lat"
        ]


        lon = fields[
            "lon"
        ]


        refl = fields[
            "refl"
        ]


        uh25 = fields[
            "uh25"
        ]


        uh03 = fields[
            "uh03"
        ]


        ir_c = fields[
            "ir_c"
        ]


        theta_prime = fields[
            "theta_prime"
        ]


        sr_u46 = fields[
            "sr_u46"
        ]


        sr_v46 = fields[
            "sr_v46"
        ]


        (
            lat_sub,
            lon_sub,
            [
                refl_sub,
                uh25_sub,
                uh03_sub,
                ir_sub,
                theta_prime_sub,
                sr_u46_sub,
                sr_v46_sub,
            ]

        ) = subset_2d(
            lat,
            lon,
            refl,
            uh25,
            uh03,
            ir_c,
            theta_prime,
            sr_u46,
            sr_v46
        )


        # ====================================================
        # REFLECTIVITY
        # ====================================================

        refl_plot = gaussian_filter(
            np.nan_to_num(
                refl_sub,
                nan=0.0
            ),
            sigma=0.5
        )


        refl_plot = np.where(
            refl_plot
            >=
            5,
            refl_plot,
            np.nan
        )


        # ====================================================
        # UH
        # ====================================================

        uh25_plot = gaussian_filter(
            np.nan_to_num(
                uh25_sub,
                nan=0.0
            ),
            sigma=0.2
        )


        uh03_plot = gaussian_filter(
            np.nan_to_num(
                uh03_sub,
                nan=0.0
            ),
            sigma=0.2
        )


        uh_combined = np.where(
            (
                uh25_plot
                >=
                75
            )
            |
            (
                uh03_plot
                >=
                50
            ),
            1,
            np.nan
        )


        # ====================================================
        # COLD POOL
        # ====================================================

        theta_prime_smooth = gaussian_filter(
            theta_prime_sub,
            sigma=2.5
        )


        theta_cp_mask = np.ma.masked_where(
            theta_prime_smooth
            >
            -2.0,
            theta_prime_smooth
        )


        # ====================================================
        # SIM IR
        # ====================================================

        ir_mask = None


        if np.isfinite(
            ir_sub
        ).any():

            ir_smooth = gaussian_filter(
                ir_sub,
                sigma=4.0
            )


            ir_mask = np.ma.masked_where(
                ir_smooth
                >
                -40,
                ir_smooth
            )


        # ====================================================
        # FIGURE
        # ====================================================

        plt.close(
            "all"
        )


        plt.rcParams[
            "hatch.color"
        ] = "#b7d6ff"


        plt.rcParams[
            "hatch.linewidth"
        ] = 0.7


        plt.rcParams[
            "contour.negative_linestyle"
        ] = "solid"


        fig = plt.figure(
            figsize=(
                14,
                10
            )
        )


        ax = plt.axes(
            projection=
                ccrs.PlateCarree()
        )


        ax.set_extent(
            cfg[
                "extent"
            ],
            crs=
                ccrs.PlateCarree()
        )


        ax.add_feature(
            cfeature.LAND,
            facecolor=
                "white",
            zorder=
                0
        )


        # ====================================================
        # IR
        # ====================================================

        if ir_mask is not None:

            ax.contourf(
                lon_sub,
                lat_sub,
                ir_mask,

                levels=[
                    -130,
                    -40
                ],

                colors=[
                    "#d0d0d0"
                ],

                alpha=
                    0.35,

                transform=
                    ccrs.PlateCarree(),

                zorder=
                    2
            )


        # ====================================================
        # COLD POOL
        # ====================================================

        ax.contourf(
            lon_sub,
            lat_sub,
            theta_cp_mask,

            levels=[
                -30,
                -2
            ],

            colors=
                "none",

            hatches=[
                "///"
            ],

            transform=
                ccrs.PlateCarree(),

            zorder=
                3
        )


        ax.contour(
            lon_sub,
            lat_sub,
            theta_prime_smooth,

            levels=[
                -2
            ],

            colors=
                "#b7d6ff",

            linewidths=
                1.2,

            transform=
                ccrs.PlateCarree(),

            zorder=
                4
        )


        # ====================================================
        # REFLECTIVITY
        # ====================================================

        pm = ax.contourf(
            lon_sub,
            lat_sub,
            refl_plot,

            levels=
                bounds,

            cmap=
                cmap,

            norm=
                norm,

            extend=
                "neither",

            transform=
                ccrs.PlateCarree(),

            zorder=
                5
        )


        # ====================================================
        # UH FILL
        # ====================================================

        if np.isfinite(
            uh_combined
        ).any():

            ax.contourf(
                lon_sub,
                lat_sub,
                uh_combined,

                levels=[
                    0.5,
                    1.5
                ],

                colors=[
                    "#8f8f8f"
                ],

                alpha=
                    0.55,

                transform=
                    ccrs.PlateCarree(),

                zorder=
                    8
            )


        # ====================================================
        # 2-5 KM UH
        # ====================================================

        if (
            np.nanmax(
                uh25_plot
            )
            >=
            75
        ):

            ax.contour(
                lon_sub,
                lat_sub,
                uh25_plot,

                levels=[
                    75
                ],

                colors=
                    "#4a4a4a",

                linewidths=
                    0.9,

                transform=
                    ccrs.PlateCarree(),

                zorder=
                    9
            )


        # ====================================================
        # 0-3 KM UH
        # ====================================================

        if (
            np.nanmax(
                uh03_plot
            )
            >=
            50
        ):

            ax.contour(
                lon_sub,
                lat_sub,
                uh03_plot,

                levels=[
                    50
                ],

                colors=
                    "black",

                linewidths=
                    0.9,

                transform=
                    ccrs.PlateCarree(),

                zorder=
                    10
            )


        # ====================================================
        # SR WIND BARBS
        # ====================================================

        if PLOT_SR_WIND_BARBS:

            barb_skip = cfg[
                "barb_skip"
            ]


            ax.barbs(
                lon_sub[
                    ::barb_skip,
                    ::barb_skip
                ],

                lat_sub[
                    ::barb_skip,
                    ::barb_skip
                ],

                ms_to_kt(
                    sr_u46_sub[
                        ::barb_skip,
                        ::barb_skip
                    ]
                ),

                ms_to_kt(
                    sr_v46_sub[
                        ::barb_skip,
                        ::barb_skip
                    ]
                ),

                length=
                    5,

                linewidth=
                    0.7,

                color=
                    "black",

                transform=
                    ccrs.PlateCarree(),

                zorder=
                    23
            )


        # ====================================================
        # STATES / COUNTIES
        # ====================================================

        add_shapefile_outline(
            ax,
            STATE_SHP,
            edgecolor="black",
            linewidth=1.4,
            zorder=13
        )


        add_shapefile_outline(
            ax,
            COUNTY_SHP,
            edgecolor="lightgray",
            linewidth=0.35,
            zorder=12
        )


        # ====================================================
        # LBF CWA
        # ====================================================

        if lbf_geom is not None:

            add_counties_clipped_to_cwa(
                ax,
                COUNTY_SHP,
                lbf_geom,
                lw=1.0,
                color="black",
                zorder=13
            )


            ax.add_geometries(
                [
                    lbf_geom
                ],

                crs=
                    ccrs.PlateCarree(),

                facecolor=
                    "none",

                edgecolor=
                    "black",

                linewidth=
                    3.5,

                zorder=
                    14
            )


            ax.add_geometries(
                [
                    lbf_geom
                ],

                crs=
                    ccrs.PlateCarree(),

                facecolor=
                    "none",

                edgecolor=
                    "white",

                linewidth=
                    1.8,

                zorder=
                    15
            )


        # ====================================================
        # TITLES
        # ====================================================

        valid_dt = (
            init_dt
            +
            timedelta(
                hours=fhr
            )
        )


        valid_title = (
            f"F{fhr:03d} Valid: "
            f"{valid_dt:%a %Y-%m-%d %HZ}"
        )


        init_title = (
            f"Init: "
            f"{init_dt:%a %Y-%m-%d %HZ} "
            "RRFS"
        )


        main_title = (
            "RRFS | Refl, "
            "2-5km UH > 75, "
            "0-3km UH > 50, "
            "θ Cold Pools, "
            "4-6 km SR Winds"
        )


        ax.text(
            0.0,
            1.042,

            main_title,

            transform=
                ax.transAxes,

            ha=
                "left",

            va=
                "bottom",

            fontsize=
                cfg[
                    "title_size"
                ],

            fontweight=
                "bold"
        )


        ax.text(
            0.0,
            1.005,

            valid_title,

            transform=
                ax.transAxes,

            ha=
                "left",

            va=
                "bottom",

            fontsize=
                cfg[
                    "subtitle_size"
                ],

            fontweight=
                "bold"
        )


        ax.text(
            1.0,
            1.005,

            init_title,

            transform=
                ax.transAxes,

            ha=
                "right",

            va=
                "bottom",

            fontsize=
                cfg[
                    "subtitle_size"
                ],

            fontweight=
                "bold"
        )


        # ====================================================
        # COLORBAR
        # ====================================================

        divider = make_axes_locatable(
            ax
        )


        cax = divider.append_axes(
            "bottom",

            size=
                "3%",

            pad=
                0.25,

            axes_class=
                plt.Axes
        )


        cbar = plt.colorbar(
            pm,

            cax=
                cax,

            orientation=
                "horizontal",

            ticks=
                REF_LEVELS,

            drawedges=
                True
        )


        cbar.set_label(
            "Reflectivity (dBZ)",
            fontsize=10,
            weight="bold"
        )


        cbar.ax.xaxis.set_label_position(
            "top"
        )


        cbar.ax.tick_params(
            axis=
                "x",
            which=
                "both",
            length=
                0
        )


        # ====================================================
        # LOGO
        # ====================================================

        if os.path.exists(
            LOGO_PATH
        ):

            logo = mpimg.imread(
                LOGO_PATH
            )


            logo_ax = ax.inset_axes(
                [
                    0.82,
                    0.84,
                    0.165,
                    0.155
                ],

                transform=
                    ax.transAxes,

                zorder=
                    50
            )


            logo_ax.imshow(
                logo
            )


            logo_ax.axis(
                "off"
            )


        # ====================================================
        # OFFICE LABEL
        # ====================================================

        ax.text(
            0.902,
            0.835,

            "NWS North Platte, NE",

            transform=
                ax.transAxes,

            ha=
                "center",

            va=
                "top",

            fontsize=
                10,

            fontweight=
                "bold",

            color=
                "black",

            zorder=
                51,

            path_effects=[
                pe.withStroke(
                    linewidth=
                        2.5,

                    foreground=
                        "white"
                )
            ]
        )


        # ====================================================
        # CREATOR CREDIT
        # ====================================================

        ax.text(
            0.01,
            0.015,

            "Plot created by: Matthew Labenz",

            transform=
                ax.transAxes,

            ha=
                "left",

            va=
                "bottom",

            fontsize=
                8,

            weight=
                "bold",

            color=
                "black",

            zorder=
                40,

            path_effects=[
                pe.withStroke(
                    linewidth=
                        2.5,

                    foreground=
                        "white"
                )
            ]
        )


        # ====================================================
        # SAVE
        # ====================================================

        outname = os.path.join(
            domain_outdir,
            f"rrfs_lbf_f{fhr:03d}.png"
        )


        plt.savefig(
            outname,
            dpi=140,
            bbox_inches="tight"
        )


        plt.close(
            fig
        )


        print(
            "Saved:",
            outname
        )


        filename = os.path.basename(
            outname
        )


        remote_key = (
            f"{R2_PRODUCT_PATH}/"
            f"{cycle_str}/"
            f"{domain_key}/"
            f"{filename}"
        )


        upload_to_r2(
            outname,
            remote_key
        )


    except Exception as e:

        print(
            f"Failed "
            f"{domain_key.upper()} "
            f"F{fhr:03d}: "
            f"{e}"
        )


# ============================================================
# MAIN LOOP
# ============================================================

successful_fhrs = []

failed_fhrs = []


for fhr in fhrs:

    print("")
    print("#" * 70)


    print(
        f"STARTING RRFS F{fhr:03d}"
    )


    print("#" * 70)


    try:

        fields = load_rrfs_fields_once(
            fhr
        )


        for (
            domain_key,
            cfg

        ) in DOMAINS.items():

            plot_domain_from_fields(
                fields,
                domain_key,
                cfg,
                fhr
            )


        successful_fhrs.append(
            fhr
        )


        del fields

        gc.collect()


    except Exception as e:

        failed_fhrs.append(
            (
                fhr,
                str(
                    e
                )
            )
        )


        print("")
        print(
            f"FAILED F{fhr:03d}: "
            f"{e}"
        )


        gc.collect()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("")
print("=" * 70)
print("RRFS PROCESSING SUMMARY")
print("=" * 70)


if successful_fhrs:

    print(
        "Successful forecast hours:"
    )


    print(
        ", ".join(
            f"F{x:03d}"
            for x
            in successful_fhrs
        )
    )


else:

    print(
        "Successful forecast hours: none"
    )


print("")


if failed_fhrs:

    print(
        "Failed forecast hours:"
    )


    for (
        fhr,
        error

    ) in failed_fhrs:

        print(
            f"F{fhr:03d}: "
            f"{error}"
        )


else:

    print(
        "Failed forecast hours: none"
    )


print("")
print(
    "Done. Uploaded RRFS "
    "reflectivity/UH to R2:"
)


print(
    R2_PRODUCT_PATH
)


print("=" * 70)
