# -*- coding: utf-8 -*-

# ============================================================
# HRRR WIND GUST + DCAPE + 10-M WIND
#
# SITE PRODUCT
#
# FILLED:
#   HRRR Surface Wind Gust (mph)
#
# CONTOURS:
#   DCAPE every 100 J/kg
#
# BARBS:
#   Instantaneous 10-m wind (kt)
#
# DOMAIN:
#   Regional only
#
# OUTPUT:
#   runs/cams/hrrr/gust_dcape/
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import json
import zipfile
import time
import requests
import boto3
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.image as mpimg

from matplotlib.colors import ListedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from scipy.ndimage import gaussian_filter

from shapely.ops import unary_union
from shapely.prepared import prep

from datetime import datetime, timedelta

from herbie import Herbie

import metpy.calc as mpcalc
from metpy.units import units


# ============================================================
# BASE PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


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
            "ContentType": content_type
        }
    )

    print(
        "Uploaded to R2:",
        remote_key
    )


# ============================================================
# ASSETS
# ============================================================

zip_path = os.path.join(
    BASE_DIR,
    "assets",
    "c_18mr25.zip"
)

extract_path = os.path.join(
    BASE_DIR,
    "assets"
)

if os.path.exists(zip_path):
    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_ref:
        zip_ref.extractall(
            extract_path
        )


# ============================================================
# DOMAIN
#
# REGIONAL ONLY
# ============================================================

DOMAINS = {

    "regional": {
        "label": "Regional",
        "extent": [
            -107.5,
            -93.0,
            38.5,
            44.2
        ],
        "title_size": 13,
        "subtitle_size": 11,
        "barb_skip": 20,
    },

}


# ============================================================
# SHAPEFILES / LOGO
# ============================================================

COUNTY_SHP = os.path.join(
    BASE_DIR,
    "assets",
    "cb_2018_us_county_500k.shp"
)

STATE_SHP = os.path.join(
    BASE_DIR,
    "assets",
    "cb_2018_us_state_500k.shp"
)

LBF_CWA_SHP = os.path.join(
    BASE_DIR,
    "assets",
    "c_18mr25.shp"
)

LOGO_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "NOAANWSLogos.png"
)


# ============================================================
# PRODUCT SETTINGS
# ============================================================

# Gust/DCAPE are valid at F000
START_FHR = 0

PLOT_10M_WIND_BARBS = True

# HRRR ~3 km.
# 6 means roughly ~18 km DCAPE profile spacing.
DCAPE_STRIDE = 6

# Slight smoothing to make the DCAPE contours cleaner.
DCAPE_SMOOTH_SIGMA = 0.7

MIN_GUST_MPH = 15.0

PLOT_CITY_LABELS = False


# ============================================================
# STATIONS
# ============================================================

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
# GUST COLOR TABLE
# ============================================================

GUST_BOUNDS = np.arange(
    15,
    75,
    1
)

GUST_COLORS = [
    "#ffffff",
    "#f1f8ff",
    "#def0fd",
    "#cae6fc",
    "#b7defb",
    "#a4d5fa",
    "#92cdf8",
    "#8ab6ef",
    "#839fe6",
    "#7c87dd",

    "#7570d4",
    "#6e59cb",
    "#8566ce",
    "#9c72d1",
    "#b27fd5",
    "#ca8bd8",
    "#e298db",
    "#dc8cd5",
    "#d580cf",
    "#cf74c9",

    "#c969c3",
    "#c35dbd",
    "#bb4fb5",
    "#b342ad",
    "#ab35a5",
    "#a3289d",
    "#9b1d95",
    "#a21c80",
    "#a91c6a",
    "#b11c55",

    "#b81c41",
    "#c01c2e",
    "#c42032",
    "#c72435",
    "#d33441",
    "#d73b45",
    "#df494c",
    "#e35050",
    "#e75754",
    "#e97559",

    "#ec935d",
    "#efb262",
    "#f3d167",
    "#f7f16b",
    "#f0e765",
    "#eadd60",
    "#e4d35a",
    "#dec954",
    "#d8bf4e",
    "#d1b548",

    "#cbab42",
    "#c5a13c",
    "#bf9737",
    "#b98e31",
    "#b3842b",
    "#ad7a26",
    "#a77021",
    "#a1661c",
    "#9b5c17"
]

gust_cmap = ListedColormap(
    GUST_COLORS,
    name="gust_bins"
)

gust_norm = BoundaryNorm(
    GUST_BOUNDS,
    gust_cmap.N,
    clip=True
)


# ============================================================
# DCAPE CONTOUR STYLE
#
# EVERY 100 J/KG
# 500 -> 3000 J/KG
#
# YELLOW/GOLD -> ORANGE -> RED -> DARK RED
# ============================================================

DCAPE_LEVELS = np.arange(
    500,
    3100,
    100
)

DCAPE_COLORS = [
    "#f8d500",   # 500
    "#f6cc00",   # 600
    "#f4c200",   # 700
    "#f2b900",   # 800
    "#f0af00",   # 900

    "#eea400",   # 1000
    "#ec9900",   # 1100
    "#ea8d00",   # 1200
    "#e78000",   # 1300
    "#e57200",   # 1400

    "#e26300",   # 1500
    "#df5300",   # 1600
    "#dc4200",   # 1700
    "#d93200",   # 1800
    "#d52100",   # 1900

    "#d01000",   # 2000
    "#ca0000",   # 2100
    "#bf0000",   # 2200
    "#b50000",   # 2300
    "#aa0000",   # 2400

    "#9f0000",   # 2500
    "#940000",   # 2600
    "#890000",   # 2700
    "#7e0000",   # 2800
    "#730000",   # 2900
    "#680000",   # 3000
]


# Thicker contour every 500 J/kg
DCAPE_WIDTHS = []

for level in DCAPE_LEVELS:

    if level % 500 == 0:
        DCAPE_WIDTHS.append(2.0)

    else:
        DCAPE_WIDTHS.append(1.15)


# ============================================================
# GENERAL HELPERS
# ============================================================

def url_exists(
    url,
    timeout=12
):

    try:

        r = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True
        )

        return r.status_code == 200

    except Exception:

        return False


# ============================================================
# FIND LATEST HRRR CYCLE
# ============================================================

def find_latest_hrrr_cycle(
    max_back_hours=36
):

    now = datetime.utcnow().replace(
        minute=0,
        second=0,
        microsecond=0
    )

    for back in range(
        max_back_hours + 1
    ):

        dt = (
            now
            -
            timedelta(hours=back)
        )

        cycle_date = dt.strftime(
            "%Y%m%d"
        )

        cycle_hour = dt.hour

        test_url = (
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            f"hrrr.{cycle_date}/"
            "conus/"
            f"hrrr.t{cycle_hour:02d}z."
            "wrfsfcf00.grib2"
        )

        if url_exists(
            test_url
        ):

            print(
                f"Latest HRRR cycle found: "
                f"{cycle_date} "
                f"{cycle_hour:02d}Z"
            )

            return (
                cycle_date,
                cycle_hour
            )

    raise RuntimeError(
        "Could not find a recent HRRR cycle."
    )


# ============================================================
# UNIT / COORDINATE HELPERS
# ============================================================

def to_lon180(
    lon
):

    return (
        (
            np.asarray(lon)
            +
            180
        )
        %
        360
    ) - 180


def ms_to_kt(
    ms
):

    return (
        ms
        *
        1.94384449244
    )


def ms_to_mph(
    ms
):

    return (
        ms
        *
        2.23693629205
    )


def get_lat_lon(
    da
):

    if (
        "latitude" in da.coords
        and
        "longitude" in da.coords
    ):

        lat = np.asarray(
            da.latitude.values
        )

        lon = to_lon180(
            da.longitude.values
        )

    elif (
        "lat" in da.coords
        and
        "lon" in da.coords
    ):

        lat = np.asarray(
            da.lat.values
        )

        lon = to_lon180(
            da.lon.values
        )

    else:

        raise RuntimeError(
            "Could not find latitude/"
            "longitude coordinates."
        )

    return (
        lat,
        lon
    )


# ============================================================
# HRRR FIELD LOADER
# ============================================================

def hrrr_field(
    cycle_date,
    cycle_hour,
    fhr,
    product,
    search,
    label
):

    init_dt = datetime.strptime(
        f"{cycle_date}{cycle_hour:02d}",
        "%Y%m%d%H"
    )

    priority_sets = [
        ["aws"],
        ["google"],
        ["azure"],
        ["nomads"],
    ]

    last_err = None

    for priority in priority_sets:

        try:

            print(
                f"Trying {label} | "
                f"F{fhr:03d} | "
                f"{search} | "
                f"source={priority[0]}"
            )

            H = Herbie(
                init_dt,
                model="hrrr",
                product=product,
                fxx=fhr,
                priority=priority,
                verbose=False
            )

            ds = H.xarray(
                search,
                remove_grib=False
            )

            if isinstance(
                ds,
                list
            ):

                ds = ds[0]

            if len(
                ds.data_vars
            ) == 0:

                raise RuntimeError(
                    f"No variables found for "
                    f"{label} with search "
                    f"{search}"
                )

            var = list(
                ds.data_vars
            )[0]

            return (
                ds[var]
                .squeeze()
            )

        except Exception as e:

            print(
                f"Failed {label} from "
                f"{priority[0]}: {e}"
            )

            last_err = e

    raise RuntimeError(
        f"Could not open {label} "
        f"F{fhr:03d} after all sources. "
        f"Last error: {last_err}"
    )


# ============================================================
# DOMAIN INDEX HELPER
# ============================================================

def get_domain_indices(
    lat,
    lon,
    extent,
    padding=3
):

    lon_min, lon_max, lat_min, lat_max = (
        extent
    )

    mask = (
        np.isfinite(lat)
        &
        np.isfinite(lon)
        &
        (lon >= lon_min)
        &
        (lon <= lon_max)
        &
        (lat >= lat_min)
        &
        (lat <= lat_max)
    )

    if not np.any(
        mask
    ):

        raise RuntimeError(
            "No HRRR grid points found "
            "inside selected domain."
        )

    iy, ix = np.where(
        mask
    )

    iy0 = max(
        iy.min() - padding,
        0
    )

    iy1 = min(
        iy.max() + padding + 1,
        lat.shape[0]
    )

    ix0 = max(
        ix.min() - padding,
        0
    )

    ix1 = min(
        ix.max() + padding + 1,
        lon.shape[1]
    )

    return (
        slice(
            iy0,
            iy1
        ),
        slice(
            ix0,
            ix1
        )
    )


# ============================================================
# SHAPEFILE OUTLINE
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
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder
    )


# ============================================================
# LBF CWA GEOMETRY
# ============================================================

def get_lbf_cwa_geom(
    cwa_shp_path
):

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
            ).upper()
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
            ).upper()
            ==
            "LBF"
        )
    ]

    if not geoms:

        geoms = [
            r.geometry
            for r in recs
        ]

    return unary_union(
        geoms
    )


# ============================================================
# COUNTIES CLIPPED TO LBF CWA
# ============================================================

def add_counties_clipped_to_cwa(
    ax,
    counties_shp_path,
    cwa_geom,
    lw=1.0,
    color="black",
    zorder=6
):

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
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        zorder=zorder
    )


# ============================================================
# CITY LABELS
# ============================================================

def plot_city_labels(
    ax,
    cities,
    zorder=40,
    fontsize=9
):

    for name, (
        lon,
        lat
    ) in cities.items():

        ax.text(
            lon,
            lat,
            name,

            transform=
                ccrs.PlateCarree(),

            fontsize=
                fontsize,

            color=
                "black",

            ha=
                "center",

            va=
                "center",

            zorder=
                zorder,

            path_effects=[
                pe.withStroke(
                    linewidth=3,
                    foreground="white"
                )
            ]
        )


# ============================================================
# PRESSURE DIMENSION
# ============================================================

def get_pressure_dimension(
    da
):

    if "isobaricInhPa" in da.dims:

        p_dim = (
            "isobaricInhPa"
        )

        pressure = np.asarray(
            da[p_dim].values,
            dtype=float
        )

    elif "isobaricInPa" in da.dims:

        p_dim = (
            "isobaricInPa"
        )

        pressure = (
            np.asarray(
                da[p_dim].values,
                dtype=float
            )
            /
            100.0
        )

    else:

        raise RuntimeError(
            "Could not determine "
            "pressure dimension. "
            f"Dimensions: {da.dims}"
        )

    return (
        p_dim,
        pressure
    )


# ============================================================
# DCAPE CALCULATION
# ============================================================

def calculate_dcape_grid(
    temp_da,
    dewpoint_da,
    psfc,
    t2,
    td2,
    lat,
    lon,
    domain_extent,
    stride=6
):

    print("")
    print("=" * 70)
    print("CALCULATING DCAPE")
    print("=" * 70)

    # --------------------------------------------------------
    # PRESSURE COORDINATE
    # --------------------------------------------------------

    (
        p_dim,
        pressure_hpa

    ) = get_pressure_dimension(
        temp_da
    )

    # --------------------------------------------------------
    # 1000-400 MB
    # --------------------------------------------------------

    use_levels = (
        (pressure_hpa <= 1000.0)
        &
        (pressure_hpa >= 400.0)
    )

    p = pressure_hpa[
        use_levels
    ]

    level_indices = np.where(
        use_levels
    )[0]

    # --------------------------------------------------------
    # HIGH PRESSURE -> LOW PRESSURE
    # --------------------------------------------------------

    sort_order = np.argsort(
        p
    )[::-1]

    p = p[
        sort_order
    ]

    level_indices = level_indices[
        sort_order
    ]

    # --------------------------------------------------------
    # DOMAIN SUBSET
    # --------------------------------------------------------

    ys, xs = get_domain_indices(
        lat,
        lon,
        domain_extent,
        padding=2
    )

    # --------------------------------------------------------
    # PRESSURE HORIZONTAL DIMENSIONS
    # --------------------------------------------------------

    horizontal_dims = [
        d
        for d in temp_da.dims
        if d != p_dim
    ]

    if len(
        horizontal_dims
    ) != 2:

        raise RuntimeError(
            "Unexpected HRRR pressure "
            f"dimensions: {temp_da.dims}"
        )

    y_dim = (
        horizontal_dims[0]
    )

    x_dim = (
        horizontal_dims[1]
    )

    # --------------------------------------------------------
    # PRESSURE PROFILES
    # --------------------------------------------------------

    T_reg = (
        temp_da
        .isel({
            p_dim:
                level_indices,

            y_dim:
                ys,

            x_dim:
                xs
        })
        .transpose(
            p_dim,
            y_dim,
            x_dim
        )
        .values
    )

    Td_reg = (
        dewpoint_da
        .isel({
            p_dim:
                level_indices,

            y_dim:
                ys,

            x_dim:
                xs
        })
        .transpose(
            p_dim,
            y_dim,
            x_dim
        )
        .values
    )

    # --------------------------------------------------------
    # SURFACE DATA ON SAME HRRR GRID
    # --------------------------------------------------------

    psfc_reg = psfc[
        ys,
        xs
    ]

    t2_reg = t2[
        ys,
        xs
    ]

    td2_reg = td2[
        ys,
        xs
    ]

    lat_reg = lat[
        ys,
        xs
    ]

    lon_reg = lon[
        ys,
        xs
    ]

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if (
        T_reg.shape[1:]
        !=
        psfc_reg.shape
    ):

        raise RuntimeError(
            "Pressure and surface grids do "
            "not have matching regional "
            f"shapes: {T_reg.shape[1:]} vs "
            f"{psfc_reg.shape}"
        )

    # --------------------------------------------------------
    # SAMPLED GRID
    # --------------------------------------------------------

    ny, nx = (
        psfc_reg.shape
    )

    iy = np.arange(
        0,
        ny,
        stride
    )

    ix = np.arange(
        0,
        nx,
        stride
    )

    dcape = np.full(
        (
            len(iy),
            len(ix)
        ),
        np.nan,
        dtype=float
    )

    lat_dcape = lat_reg[
        np.ix_(
            iy,
            ix
        )
    ]

    lon_dcape = lon_reg[
        np.ix_(
            iy,
            ix
        )
    ]

    total = (
        len(iy)
        *
        len(ix)
    )

    count = 0

    print(
        f"DCAPE sampled grid: "
        f"{len(iy)} x {len(ix)} "
        f"= {total:,} profiles"
    )

    # ========================================================
    # PROFILE LOOP
    # ========================================================

    for jy, y in enumerate(
        iy
    ):

        for jx, x in enumerate(
            ix
        ):

            count += 1

            if (
                count % 250
                ==
                0
            ):

                print(
                    f"DCAPE profiles: "
                    f"{count:,}/{total:,}"
                )

            try:

                # ------------------------------------------------
                # SURFACE
                # ------------------------------------------------

                ps = (
                    float(
                        psfc_reg[
                            y,
                            x
                        ]
                    )
                    /
                    100.0
                )

                ts = float(
                    t2_reg[
                        y,
                        x
                    ]
                )

                tds = float(
                    td2_reg[
                        y,
                        x
                    ]
                )

                if not (
                    np.isfinite(ps)
                    and
                    np.isfinite(ts)
                    and
                    np.isfinite(tds)
                ):

                    continue

                # ------------------------------------------------
                # PROFILE
                # ------------------------------------------------

                temp_profile = T_reg[
                    :,
                    y,
                    x
                ]

                td_profile = Td_reg[
                    :,
                    y,
                    x
                ]

                # ------------------------------------------------
                # REMOVE BELOW-GROUND LEVELS
                # ------------------------------------------------

                valid = (
                    np.isfinite(
                        temp_profile
                    )
                    &
                    np.isfinite(
                        td_profile
                    )
                    &
                    (
                        p
                        <
                        ps - 1.0
                    )
                )

                p_valid = p[
                    valid
                ]

                t_valid = temp_profile[
                    valid
                ]

                td_valid = td_profile[
                    valid
                ]

                if len(
                    p_valid
                ) < 8:

                    continue

                # Need full 700-500 mb source layer
                if (
                    np.nanmax(
                        p_valid
                    )
                    <
                    700
                ):

                    continue

                if (
                    np.nanmin(
                        p_valid
                    )
                    >
                    500
                ):

                    continue

                # ------------------------------------------------
                # ADD MODEL SURFACE
                # ------------------------------------------------

                profile_p = np.concatenate(
                    [
                        [ps],
                        p_valid
                    ]
                )

                profile_t = np.concatenate(
                    [
                        [ts],
                        t_valid
                    ]
                )

                profile_td = np.concatenate(
                    [
                        [tds],
                        td_valid
                    ]
                )

                # ------------------------------------------------
                # SORT HIGH -> LOW PRESSURE
                # ------------------------------------------------

                order = np.argsort(
                    profile_p
                )[::-1]

                profile_p = profile_p[
                    order
                ]

                profile_t = profile_t[
                    order
                ]

                profile_td = profile_td[
                    order
                ]

                # ------------------------------------------------
                # REMOVE DUPLICATE PRESSURE LEVELS
                # ------------------------------------------------

                _, unique_index = np.unique(
                    profile_p,
                    return_index=True
                )

                unique_index = np.sort(
                    unique_index
                )

                profile_p = profile_p[
                    unique_index
                ]

                profile_t = profile_t[
                    unique_index
                ]

                profile_td = profile_td[
                    unique_index
                ]

                # ------------------------------------------------
                # METPY UNITS
                # ------------------------------------------------

                p_q = (
                    profile_p
                    *
                    units.hPa
                )

                T_q = (
                    profile_t
                    *
                    units.kelvin
                )

                Td_q = (
                    profile_td
                    *
                    units.kelvin
                )

                # ------------------------------------------------
                # DCAPE
                # ------------------------------------------------

                result = mpcalc.downdraft_cape(
                    p_q,
                    T_q,
                    Td_q
                )

                dcape_q = (
                    result[0]
                )

                value = float(
                    dcape_q
                    .to(
                        "joule / kilogram"
                    )
                    .magnitude
                )

                if np.isfinite(
                    value
                ):

                    dcape[
                        jy,
                        jx
                    ] = max(
                        value,
                        0.0
                    )

            except Exception:

                continue

    # ========================================================
    # SMOOTH DCAPE
    # ========================================================

    valid_mask = (
        np.isfinite(
            dcape
        )
        .astype(
            float
        )
    )

    dcape_filled = np.nan_to_num(
        dcape,
        nan=0.0
    )

    smoothed_data = gaussian_filter(
        dcape_filled,
        sigma=DCAPE_SMOOTH_SIGMA
    )

    smoothed_weight = gaussian_filter(
        valid_mask,
        sigma=DCAPE_SMOOTH_SIGMA
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):

        dcape_smooth = (
            smoothed_data
            /
            smoothed_weight
        )

    dcape_smooth[
        smoothed_weight < 0.05
    ] = np.nan

    if np.any(
        np.isfinite(
            dcape_smooth
        )
    ):

        print(
            "Maximum regional DCAPE: "
            f"{np.nanmax(dcape_smooth):.0f} "
            "J/kg"
        )

    return (
        lon_dcape,
        lat_dcape,
        dcape_smooth
    )


# ============================================================
# GET LATEST HRRR CYCLE
# ============================================================

cycle_date, cycle_hour = (
    find_latest_hrrr_cycle()
)

cycle_str = (
    f"{cycle_date}_"
    f"{cycle_hour:02d}z"
)


# ============================================================
# MAX FORECAST HOUR
# ============================================================

if cycle_hour in [
    0,
    6,
    12,
    18
]:

    MAX_FHR = 48

else:

    MAX_FHR = 18


# ============================================================
# PRODUCT PATH
# ============================================================

PRODUCT_PATH = (
    "runs/cams/hrrr/"
    "gust_dcape"
)


# ============================================================
# LOAD EXISTING RUNS.JSON
# ============================================================

old_runs = []

try:

    obj = s3.get_object(
        Bucket=BUCKET,
        Key=f"{PRODUCT_PATH}/runs.json"
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


# ============================================================
# BUILD NEW RUN
# ============================================================

new_run = {
    "id":
        cycle_str,

    "label":
        (
            f"{cycle_date[:4]}-"
            f"{cycle_date[4:6]}-"
            f"{cycle_date[6:8]} "
            f"{cycle_hour:02d}z"
        ),

    "max_fhr":
        MAX_FHR
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

        rhour = int(
            rid
            .split("_")[1]
            .replace(
                "z",
                ""
            )
        )

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
                    48
                    if rhour in [
                        0,
                        6,
                        12,
                        18
                    ]
                    else
                    18
                )
        })

    elif (
        r.get("id")
        !=
        cycle_str
    ):

        combined.append(
            r
        )


# ============================================================
# KEEP LAST 4 RUNS
# ============================================================

runs_json = {
    "runs":
        combined[:4]
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
    f"{PRODUCT_PATH}/runs.json",
    content_type="application/json"
)


print(
    "Uploaded gust/DCAPE runs.json."
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTDIR = os.path.join(
    "site",
    "runs",
    "hrrr",
    "gust_dcape",
    cycle_str
)


os.makedirs(
    OUTDIR,
    exist_ok=True
)


os.makedirs(
    "site",
    exist_ok=True
)


fhrs = range(
    START_FHR,
    MAX_FHR + 1
)


lbf_geom = get_lbf_cwa_geom(
    LBF_CWA_SHP
)


print(
    "Forecast hours:",
    list(fhrs)
)


print(
    "Output directory:",
    OUTDIR
)


# ============================================================
# LOAD FIELDS ONCE PER FORECAST HOUR
# ============================================================

def load_hrrr_fields_once(
    fhr
):

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        f"Loading HRRR Gust/DCAPE | "
        f"HRRR {cycle_date} "
        f"{cycle_hour:02d}Z "
        f"F{fhr:03d}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # GUST
    # ========================================================

    gust_da = hrrr_field(
        cycle_date,
        cycle_hour,
        fhr,
        "sfc",
        ":GUST:surface:",
        "surface wind gust"
    )

    lat, lon = get_lat_lon(
        gust_da
    )

    gust_ms = np.asarray(
        gust_da.values,
        dtype=float
    )

    gust_mph = ms_to_mph(
        gust_ms
    )

    # ========================================================
    # 10-M U WIND
    # ========================================================

    u10_da = hrrr_field(
        cycle_date,
        cycle_hour,
        fhr,
        "sfc",
        ":UGRD:10 m above ground:",
        "10-m U wind"
    )

    # ========================================================
    # 10-M V WIND
    # ========================================================

    v10_da = hrrr_field(
        cycle_date,
        cycle_hour,
        fhr,
        "sfc",
        ":VGRD:10 m above ground:",
        "10-m V wind"
    )

    u10 = np.asarray(
        u10_da.values,
        dtype=float
    )

    v10 = np.asarray(
        v10_da.values,
        dtype=float
    )

    # ========================================================
    # SURFACE PRESSURE
    # ========================================================

    ps_da = hrrr_field(
        cycle_date,
        cycle_hour,
        fhr,
        "sfc",
        ":PRES:surface:",
        "surface pressure"
    )

    psfc = np.asarray(
        ps_da.values,
        dtype=float
    )

    # ========================================================
    # 2-M TEMPERATURE
    # ========================================================

    t2_da = hrrr_field(
        cycle_date,
        cycle_hour,
        fhr,
        "sfc",
        ":TMP:2 m above ground:",
        "2-m temperature"
    )

    t2 = np.asarray(
        t2_da.values,
        dtype=float
    )

    # ========================================================
    # 2-M DEWPOINT
    # ========================================================

    td2_da = hrrr_field(
        cycle_date,
        cycle_hour,
        fhr,
        "sfc",
        ":DPT:2 m above ground:",
        "2-m dewpoint"
    )

    td2 = np.asarray(
        td2_da.values,
        dtype=float
    )

    # ========================================================
    # PRESSURE-LEVEL TEMPERATURE
    # ========================================================

    temp_da = hrrr_field(
        cycle_date,
        cycle_hour,
        fhr,
        "prs",
        r":TMP:[0-9]+ mb:",
        "pressure-level temperature"
    )

    # ========================================================
    # PRESSURE-LEVEL DEWPOINT
    # ========================================================

    dewpoint_da = hrrr_field(
        cycle_date,
        cycle_hour,
        fhr,
        "prs",
        r":DPT:[0-9]+ mb:",
        "pressure-level dewpoint"
    )

    # ========================================================
    # REGIONAL DCAPE
    # ========================================================

    regional_extent = (
        DOMAINS[
            "regional"
        ][
            "extent"
        ]
    )

    (
        dcape_lon,
        dcape_lat,
        dcape

    ) = calculate_dcape_grid(
        temp_da,
        dewpoint_da,
        psfc,
        t2,
        td2,
        lat,
        lon,
        regional_extent,
        stride=DCAPE_STRIDE
    )

    return {
        "lat":
            lat,

        "lon":
            lon,

        "gust_mph":
            gust_mph,

        "u10":
            u10,

        "v10":
            v10,

        "dcape_lon":
            dcape_lon,

        "dcape_lat":
            dcape_lat,

        "dcape":
            dcape,
    }


# ============================================================
# PLOT DOMAIN
# ============================================================

def plot_domain_from_fields(
    fields,
    domain_key,
    cfg,
    fhr
):

    global LON_MIN, LON_MAX, LAT_MIN, LAT_MAX

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

        gust_mph = fields[
            "gust_mph"
        ]

        u10 = fields[
            "u10"
        ]

        v10 = fields[
            "v10"
        ]

        dcape_lon = fields[
            "dcape_lon"
        ]

        dcape_lat = fields[
            "dcape_lat"
        ]

        dcape = fields[
            "dcape"
        ]

        # ====================================================
        # SUBSET DOMAIN
        # ====================================================

        ys, xs = get_domain_indices(
            lat,
            lon,
            cfg[
                "extent"
            ],
            padding=2
        )

        lat_sub = lat[
            ys,
            xs
        ]

        lon_sub = lon[
            ys,
            xs
        ]

        gust_sub = gust_mph[
            ys,
            xs
        ]

        u10_sub = u10[
            ys,
            xs
        ]

        v10_sub = v10[
            ys,
            xs
        ]

        # ====================================================
        # MASK GUSTS BELOW 15 MPH
        # ====================================================

        gust_plot = np.ma.masked_less(
            gust_sub,
            MIN_GUST_MPH
        )

        # ====================================================
        # MAXIMUM DISPLAYED-DOMAIN GUST
        # ====================================================

        max_gust = float(
            np.nanmax(
                gust_sub
            )
        )

        print(
            f"Regional max gust: "
            f"{max_gust:.1f} mph"
        )

        # ====================================================
        # FIGURE
        # ====================================================

        plt.close(
            "all"
        )

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
        # GUST FILL
        # ====================================================

        pm = ax.contourf(
            lon_sub,
            lat_sub,
            gust_plot,

            levels=
                GUST_BOUNDS,

            cmap=
                gust_cmap,

            norm=
                gust_norm,

            extend=
                "max",

            transform=
                ccrs.PlateCarree(),

            zorder=
                3
        )

        # ====================================================
        # DCAPE
        # ====================================================

        if np.any(
            np.isfinite(
                dcape
            )
        ):

            # ------------------------------------------------
            # WHITE HALO UNDER CONTOURS
            # ------------------------------------------------

            halo_widths = [
                width + 1.3
                for width in DCAPE_WIDTHS
            ]

            ax.contour(
                dcape_lon,
                dcape_lat,
                dcape,

                levels=
                    DCAPE_LEVELS,

                colors=
                    "white",

                linewidths=
                    halo_widths,

                alpha=
                    0.72,

                transform=
                    ccrs.PlateCarree(),

                zorder=
                    6
            )

            # ------------------------------------------------
            # COLORED DCAPE
            # ------------------------------------------------

            dc = ax.contour(
                dcape_lon,
                dcape_lat,
                dcape,

                levels=
                    DCAPE_LEVELS,

                colors=
                    DCAPE_COLORS,

                linewidths=
                    DCAPE_WIDTHS,

                transform=
                    ccrs.PlateCarree(),

                zorder=
                    7
            )

            # ------------------------------------------------
            # LABEL ALL 100 J/KG CONTOURS
            # ------------------------------------------------

            labels = ax.clabel(
                dc,

                levels=
                    DCAPE_LEVELS,

                inline=
                    True,

                inline_spacing=
                    3,

                fmt=
                    lambda x:
                    f"{int(x)}",

                fontsize=
                    7.2
            )

            for label in labels:

                label.set_fontweight(
                    "bold"
                )

                label.set_path_effects([
                    pe.withStroke(
                        linewidth=
                            2.3,

                        foreground=
                            "white"
                    )
                ])

        # ====================================================
        # 10-M WIND BARBS
        # ====================================================

        if PLOT_10M_WIND_BARBS:

            barb_skip = cfg.get(
                "barb_skip",
                20
            )

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
                    u10_sub[
                        ::barb_skip,
                        ::barb_skip
                    ]
                ),

                ms_to_kt(
                    v10_sub[
                        ::barb_skip,
                        ::barb_skip
                    ]
                ),

                length=
                    5,

                linewidth=
                    0.65,

                color=
                    "black",

                barb_increments={
                    "half": 5,
                    "full": 10,
                    "flag": 50
                },

                transform=
                    ccrs.PlateCarree(),

                zorder=
                    10
            )

        # ====================================================
        # STATE / COUNTY BORDERS
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
        # LBF COUNTIES
        # ====================================================

        add_counties_clipped_to_cwa(
            ax,
            COUNTY_SHP,
            lbf_geom,
            lw=1.0,
            color="black",
            zorder=13
        )

        # ====================================================
        # LBF CWA OUTLINE
        # ====================================================

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
        # CITY LABELS
        # ====================================================

        if PLOT_CITY_LABELS:

            plot_city_labels(
                ax,
                STATIONS,
                zorder=40,
                fontsize=9
            )

        # ====================================================
        # TIME
        # ====================================================

        init_dt = datetime.strptime(
            f"{cycle_date}"
            f"{cycle_hour:02d}",
            "%Y%m%d%H"
        )

        valid_dt = (
            init_dt
            +
            timedelta(
                hours=fhr
            )
        )

        # ====================================================
        # TITLES
        # ====================================================

        main_title = (
            "HRRR | Surface Wind Gust, "
            "DCAPE & 10-m Wind"
        )

        valid_title = (
            f"F{fhr:03d} Valid: "
            f"{valid_dt:%a %Y-%m-%d %Hz}"
        )

        init_title = (
            f"Init: "
            f"{init_dt:%a %Y-%m-%d %Hz} "
            "HRRR"
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
        # MAX GUST BOX
        #
        # NO STAR / NO LOCATION MARKER
        # ====================================================

        ax.text(
            0.015,
            0.975,

            "MAX GUST\n"
            f"{max_gust:.0f} mph",

            transform=
                ax.transAxes,

            ha=
                "left",

            va=
                "top",

            fontsize=
                11,

            fontweight=
                "bold",

            color=
                "black",

            bbox=dict(
                boxstyle=
                    "round,pad=0.35",

                facecolor=
                    "white",

                edgecolor=
                    "black",

                linewidth=
                    1.1,

                alpha=
                    0.90
            ),

            zorder=
                50
        )

        # ====================================================
        # COLORBAR
        # ====================================================

        from mpl_toolkits.axes_grid1 import (
            make_axes_locatable
        )

        divider = make_axes_locatable(
            ax
        )

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

            ticks=[
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
                70
            ],

            drawedges=False
        )

        cbar.set_label(
            "Surface Wind Gust (mph)",
            fontsize=10,
            weight="bold"
        )

        cbar.ax.xaxis.set_label_position(
            "top"
        )

        cbar.ax.tick_params(
            axis="x",
            which="both",
            length=0
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
                9,

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
            f"hrrr_gust_dcape_f{fhr:03d}.png"
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

        # ====================================================
        # UPLOAD
        # ====================================================

        remote_key = (
            f"{PRODUCT_PATH}/"
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
            f"F{fhr:03d}: {e}"
        )


# ============================================================
# RETRY SETTINGS
# ============================================================

MAX_FHR_ATTEMPTS = 3

RETRY_WAIT_SECONDS = 20

successful_fhrs = []

failed_fhrs = []


# ============================================================
# MAIN LOOP
# ============================================================

for fhr in fhrs:

    fhr_success = False

    for attempt in range(
        1,
        MAX_FHR_ATTEMPTS + 1
    ):

        try:

            print(
                "\n"
                +
                "=" * 70
            )

            print(
                f"Processing HRRR "
                f"F{fhr:03d} | "
                f"Attempt "
                f"{attempt}/"
                f"{MAX_FHR_ATTEMPTS}"
            )

            print(
                "=" * 70
            )

            fields = load_hrrr_fields_once(
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

            fhr_success = True

            successful_fhrs.append(
                fhr
            )

            print(
                f"Successfully completed "
                f"HRRR F{fhr:03d}"
            )

            break

        except Exception as e:

            print(
                f"HRRR F{fhr:03d} failed "
                f"on attempt "
                f"{attempt}/"
                f"{MAX_FHR_ATTEMPTS}: "
                f"{e}"
            )

            if (
                attempt
                <
                MAX_FHR_ATTEMPTS
            ):

                print(
                    f"Waiting "
                    f"{RETRY_WAIT_SECONDS} "
                    "seconds before retrying "
                    f"F{fhr:03d}..."
                )

                time.sleep(
                    RETRY_WAIT_SECONDS
                )

    if not fhr_success:

        failed_fhrs.append(
            fhr
        )

        print(
            f"Skipping HRRR "
            f"F{fhr:03d} after "
            f"{MAX_FHR_ATTEMPTS} "
            "failed attempts."
        )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    "HRRR GUST/DCAPE PROCESSING SUMMARY"
)

print(
    "=" * 70
)

print(
    "Successful forecast hours:",
    [
        f"F{fhr:03d}"
        for fhr in successful_fhrs
    ]
)

print(
    "Failed forecast hours:",
    [
        f"F{fhr:03d}"
        for fhr in failed_fhrs
    ]
)
