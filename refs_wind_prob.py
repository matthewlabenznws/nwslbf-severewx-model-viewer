# ============================================================
# REFS | R2 Meso-Ensemble Probability Product
# Damaging Wind Probability
# Fill: 10-m WIND > 25.72 m/s probability
# Contours: Composite Reflectivity > 40 dBZ probability
# Uploads PNGs to runs/mesoensprob/refs/wind_prob/
# ============================================================

import os, re, json, zipfile, requests, boto3
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSET_DIR = os.path.join(BASE_DIR, "assets")
COUNTY_SHP = os.path.join(ASSET_DIR, "cb_2018_us_county_500k.shp")
STATE_SHP = os.path.join(ASSET_DIR, "cb_2018_us_state_500k.shp")
LBF_CWA_SHP = os.path.join(ASSET_DIR, "c_18mr25.shp")
LOGO_PATH = os.path.join(ASSET_DIR, "NOAANWSLogos.png")

zip_path = os.path.join(ASSET_DIR, "c_18mr25.zip")
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ASSET_DIR)

DATA_DIR = os.path.join(BASE_DIR, "refs_wind_prob_subsets")

SECTION_KEY = "mesoensprob"
MODEL_KEY = "refs"
PRODUCT_KEY = "wind_prob"

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

BUCKET = os.environ["R2_BUCKET"]
ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    config=Config(signature_version="s3v4"),
    region_name="auto",
)

def upload_to_r2(local_file, remote_key, content_type="image/png"):
    s3.upload_file(
        local_file,
        BUCKET,
        remote_key,
        ExtraArgs={"ContentType": content_type}
    )
    print("Uploaded to R2:", remote_key)

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

def add_spc_severe_domain():
    try:
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "returnGeometry": "true",
            "outSR": "4326",
        }

        r = requests.get(SPC_DAY1_CAT_URL, params=params, timeout=30)
        r.raise_for_status()

        gdf = gpd.GeoDataFrame.from_features(
            r.json()["features"],
            crs="EPSG:4326"
        )

        risk_col = None

        for col in gdf.columns:
            vals = gdf[col].astype(str).str.upper()
            if vals.isin(SPC_RISK_ORDER.keys()).any():
                risk_col = col
                break

        if risk_col is None:
            print("SPC severe domain skipped: no risk column")
            return

        gdf["risk"] = gdf[risk_col].astype(str).str.upper()
        gdf["risk_rank"] = gdf["risk"].map(SPC_RISK_ORDER)

        severe = gdf[gdf["risk_rank"] >= SPC_RISK_ORDER[MIN_SPC_RISK]].copy()

        if severe.empty:
            print("SPC severe domain skipped: no SLGT+ risk")
            return

        highest = severe[severe["risk_rank"] == severe["risk_rank"].max()].copy()
        highest["_area"] = highest.to_crs(epsg=5070).geometry.area.values

        main_poly = highest.loc[highest["_area"].idxmax()]

        main_gdf = gpd.GeoDataFrame(
            [main_poly],
            geometry="geometry",
            crs="EPSG:4326"
        )

        centroid_ll = gpd.GeoSeries(
            main_gdf.to_crs(epsg=5070).geometry.centroid,
            crs="EPSG:5070"
        ).to_crs(epsg=4326).iloc[0]

        DOMAINS["spc_severe"] = {
            "label": f"SPC {main_poly['risk']} Risk",
            "extent": [
                centroid_ll.x - SEVERE_DOMAIN_WIDTH / 2,
                centroid_ll.x + SEVERE_DOMAIN_WIDTH / 2,
                centroid_ll.y - SEVERE_DOMAIN_HEIGHT / 2,
                centroid_ll.y + SEVERE_DOMAIN_HEIGHT / 2,
            ],
            "title_size": 11,
            "subtitle_size": 11,
        }

        print(f"Added SPC severe domain: {main_poly['risk']}")

    except Exception as e:
        print(f"SPC severe domain skipped due to error: {e}")

add_spc_severe_domain()

MODEL_LABEL = "REFS"

VALID_CYCLES = [0, 6, 12, 18]

START_FHR = 1
MAX_FHR = 60

CYCLE_DELAY_MINUTES = 90

PROB_LEVELS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
PROB_TICKS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

REFC_CONTOURS = [10, 20, 30, 40, 50, 60, 70]

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
            f"{label} is not 2D after squeeze. Shape={arr.shape}"
        )

    return arr

def normalize_probability(arr):
    arr = np.asarray(arr, dtype=float)

    if np.nanmax(arr) <= 1.01:
        arr = arr * 100.0

    return arr

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

def grib_url(init_dt, fhr):
    ymd = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")

    fname = f"refs.t{hh}z.prob.f{fhr:03d}.conus.grib2"

    return (
        f"https://noaa-rrfs-pds.s3.amazonaws.com/"
        f"rrfs_public/refs.{ymd}/{hh}/enspost/{fname}"
    )

def url_exists(url, timeout=15):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False

def find_latest_cycle(max_back_hours=96):
    now = datetime.now(timezone.utc) - timedelta(minutes=CYCLE_DELAY_MINUTES)

    for back in range(max_back_hours + 1):
        dt = now - timedelta(hours=back)

        if dt.hour not in VALID_CYCLES:
            continue

        dt = dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)

        test_url = grib_url(dt, 1) + ".idx"

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
            parsed.append({
                "i": i,
                "line": line,
                "msg_num": int(parts[0]),
                "start": int(parts[1]),
            })
        except Exception:
            continue

    for j in range(len(parsed)):
        if j < len(parsed) - 1:
            parsed[j]["end"] = parsed[j + 1]["start"] - 1
        else:
            parsed[j]["end"] = None

    return parsed

def find_idx_match(parsed, all_terms, label):
    terms = [t.lower() for t in all_terms]
    matches = []

    for item in parsed:
        line_lower = item["line"].lower()

        if all(term in line_lower for term in terms):
            matches.append(item)

    if not matches:
        sample = "\n".join([p["line"] for p in parsed[:200]])
        raise RuntimeError(
            f"Could not find {label} using terms {all_terms}.\n"
            f"First 200 IDX lines:\n{sample}"
        )

    match = matches[0]

    print(f"Matched {label}:")
    print(match["line"])

    return match

def download_byte_range(grib_url_in, start, end, outpath):
    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        print("Using cached subset:", outpath)
        return outpath

    headers = {}

    if end is None:
        headers["Range"] = f"bytes={start}-"
    else:
        headers["Range"] = f"bytes={start}-{end}"

    print("Downloading byte range:", headers["Range"])

    r = requests.get(grib_url_in, headers=headers, stream=True, timeout=120)
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

def idx_field(init_dt, fhr, term_sets, label):
    g_url = grib_url(init_dt, fhr)
    idx_url = g_url + ".idx"

    lines = read_idx(idx_url)
    parsed = parse_idx_lines(lines)

    last_error = None

    for terms in term_sets:
        try:
            match = find_idx_match(parsed, terms, label)

            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")

            outpath = os.path.join(
                DATA_DIR,
                f"refs_prob_{init_dt:%Y%m%d_%H}z_f{fhr:03d}_"
                f"{safe_label}_{match['msg_num']}.grib2"
            )

            download_byte_range(
                g_url,
                match["start"],
                match["end"],
                outpath
            )

            return open_subset_grib(outpath, label)

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not open {label}. Last error: {last_error}")

def subset_2d(lat, lon, *fields):
    mask = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lon >= LON_MIN)
        & (lon <= LON_MAX)
        & (lat >= LAT_MIN)
        & (lat <= LAT_MAX)
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
        [f[iy0:iy1, ix0:ix1] for f in fields],
    )

def upload_runs_json(init_dt, cycle_str, max_fhr):
    old_runs = []

    try:
        obj = s3.get_object(
            Bucket=BUCKET,
            Key=f"{R2_PRODUCT_PATH}/runs.json"
        )

        old_data = json.loads(obj["Body"].read().decode("utf-8"))
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

    with open("runs.json", "w") as f:
        json.dump({"runs": combined[:4]}, f, indent=2)

    upload_to_r2(
        "runs.json",
        f"{R2_PRODUCT_PATH}/runs.json",
        content_type="application/json"
    )

    print("Uploaded runs.json with last 4 REFS runs.")

init_dt = find_latest_cycle()
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

def load_fields_once(fhr):
    print("\n" + "=" * 70)
    print(f"Loading {MODEL_LABEL} | Init {init_dt:%Y-%m-%d %HZ} | F{fhr:03d}")
    print("=" * 70)

    wind_da = idx_field(
        init_dt,
        fhr,
        [
            ["WIND", "10 m above ground", "prob >25.72"],
            ["WIND", "10 m", "prob >25.72"],
            ["WIND", "prob >25.72"],
        ],
        "10m wind >25.72 probability"
    )

    lat, lon = get_lat_lon(wind_da)

    wind_prob = normalize_probability(
        ensure_2d_field(wind_da, "10m wind >25.72 probability")
    )

    refc_da = idx_field(
        init_dt,
        fhr,
        [
            ["REFC", "prob >40"],
            ["REFC", "entire atmosphere", "prob >40"],
        ],
        "REFC >40 probability"
    )

    refc_prob = normalize_probability(
        ensure_2d_field(refc_da, "REFC >40 probability")
    )

    return {
        "lat": lat,
        "lon": lon,
        "wind_prob": wind_prob,
        "refc_prob": refc_prob,
    }

def plot_domain(fields, domain_key, cfg, fhr):
    global LON_MIN, LON_MAX, LAT_MIN, LAT_MAX

    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = cfg["extent"]

    domain_outdir = os.path.join(OUTDIR, domain_key)
    os.makedirs(domain_outdir, exist_ok=True)

    print(f"Plotting {domain_key.upper()} | F{fhr:03d}")

    lat = fields["lat"]
    lon = fields["lon"]
    wind_prob = fields["wind_prob"]
    refc_prob = fields["refc_prob"]

    lat_sub, lon_sub, [
        wind_sub,
        refc_sub,
    ] = subset_2d(
        lat,
        lon,
        wind_prob,
        refc_prob
    )

    wind_plot = gaussian_filter(np.nan_to_num(wind_sub, nan=0.0), sigma=0.7)
    refc_plot = gaussian_filter(np.nan_to_num(refc_sub, nan=0.0), sigma=0.7)

    wind_plot = np.where(wind_plot >= 5, wind_plot, np.nan)
    refc_plot = np.where(refc_plot >= 5, refc_plot, np.nan)

    plt.close("all")
    plt.rcParams["contour.negative_linestyle"] = "solid"

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(cfg["extent"], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)

    pm = ax.contourf(
        lon_sub,
        lat_sub,
        wind_plot,
        levels=PROB_LEVELS,
        cmap=plt.cm.OrRd,
        extend="max",
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    if np.isfinite(refc_plot).any() and np.nanmax(refc_plot) >= 10:
        cs = ax.contour(
            lon_sub,
            lat_sub,
            refc_plot,
            levels=REFC_CONTOURS,
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
        add_counties_clipped_to_cwa(
            ax,
            COUNTY_SHP,
            lbf_geom,
            lw=1.0,
            color="black",
            zorder=13
        )

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

    main_title = (
        f"{MODEL_LABEL} | Fill: 10-m Wind > 25.72 m/s Probability | "
        "Black Contours: Composite Reflectivity > 40 dBZ Probability"
    )

    ax.text(
        0.0,
        1.042,
        main_title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=cfg["title_size"],
        fontweight="bold"
    )

    ax.text(
        0.0,
        1.005,
        f"F{fhr:03d} Valid: {valid_dt:%a %Y-%m-%d %HZ}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=cfg["subtitle_size"],
        fontweight="bold"
    )

    ax.text(
        1.0,
        1.005,
        f"Init: {init_dt:%a %Y-%m-%d %HZ} {MODEL_LABEL}",
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
        "Probability of 10-m Wind > 25.72 m/s (~58 mph) (%)",
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

    outname = os.path.join(domain_outdir, f"refs_wind_prob_f{fhr:03d}.png")

    plt.savefig(outname, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", outname)

    remote_key = (
        f"{R2_PRODUCT_PATH}/"
        f"{cycle_str}/"
        f"{domain_key}/"
        f"{os.path.basename(outname)}"
    )

    upload_to_r2(outname, remote_key)

for fhr in fhrs:
    try:
        fields = load_fields_once(fhr)

        for domain_key, cfg in DOMAINS.items():
            plot_domain(fields, domain_key, cfg, fhr)

    except Exception as e:
        print(f"FAILED F{fhr:03d}: {e}")

print("Done. Uploaded REFS wind probability to R2:", R2_PRODUCT_PATH)
