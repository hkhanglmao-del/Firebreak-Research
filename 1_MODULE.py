"""
mo_hinh.py - Mô hình lan cháy (CA) chạy không cần giao diện, phục vụ kiểm chứng với vụ cháy thật.

Tách từ app Firebreak (bản 1_MODULE.py, ngày 15/11/2026). Giữ nguyên logic dữ liệu nền và CA.
Thay đổi so với app:
  1. Bỏ giao diện Streamlit và phần lập đường băng (min-cut, lịch thi công, vẽ hình).
  2. build_env_ca(): giống fetch_and_build_api_environment() của app, nhưng bỏ các lớp chỉ dùng
     cho lập đường băng (giờ công, thời gian tiếp cận, hillshade). Bắt buộc có DEM và WorldCover
     cục bộ (báo lỗi thay vì tự chuyển sang dữ liệu thay thế), trừ khi chọn DEM API cho thí nghiệm.
  3. run_ca_step_vectorized(): thêm k_wind, k_wind_up, k_slope; mặc định bằng giá trị gốc
     (0.085; 0.25; 0.035) nên không truyền gì thì chạy y hệt app.
  4. Thêm timeline_from_csv(), save_env(), load_env().
Soạn với hỗ trợ của Claude (Anthropic).
"""
import json
import time
from pathlib import Path as FilePath

import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path
from scipy.ndimage import binary_dilation, distance_transform_edt, label, zoom

# ==============================================================================
# HẰNG SỐ
# ==============================================================================
GRID_SHAPE = (540, 540)
APP_DIR = FilePath(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
LOCAL_DEM_DIR = DATA_DIR / "dem"
LOCAL_WORLDCOVER_DIR = DATA_DIR / "worldcover"
OSM_CACHE_DIR = DATA_DIR / "osm_cache"
LAT_SPAN_DEG = 0.15          # Khung vĩ độ 0.15 độ (~16.6 km)
LON_SPAN_DEG = 0.15          # Khung kinh độ 0.15 độ
NOMINAL_PIXEL_M = 30.0       # Kích thước pixel danh nghĩa 30 m
CA_STEP_TO_HOURS = 0.5       # 1 bước CA = 0.5 giờ

STATE_EMPTY     = 0  # Đường / nước / không có thực bì
STATE_TREE      = 1  # Thực bì chưa cháy
STATE_BURNING   = 2  # Đang cháy
STATE_BURNED    = 3  # Đã cháy
STATE_FIREBREAK = 4  # Đường băng

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


# ==============================================================================
# DỮ LIỆU NỀN: DEM, WORLDCOVER, OSM (giữ nguyên từ app)
# ==============================================================================
def compute_projected_pixel_scales(lat_center):
    """Kích thước thực (m) của 1 pixel theo vĩ độ trên WGS84."""
    meters_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * np.radians(lat_center))
    meters_per_deg_lon = 111412.84 * np.cos(np.radians(lat_center))
    dy_m = (LAT_SPAN_DEG * meters_per_deg_lat) / GRID_SHAPE[0]
    dx_m = (LON_SPAN_DEG * meters_per_deg_lon) / GRID_SHAPE[1]
    pixel_area_ha = (dy_m * dx_m) / 10000.0
    return float(dy_m), float(dx_m), float(pixel_area_ha)


def _load_local_geotiff_grid(folder, lat_center, lon_center, categorical=False, required_filename_tokens=None):
    """Chiếu lại các tile GeoTIFF cục bộ về lưới GRID_SHAPE. Trả về (grid, valid_mask, files_used)."""
    folder = FilePath(folder)
    paths = sorted(set(folder.glob("*.tif")) | set(folder.glob("*.tiff"))) if folder.exists() else []
    if required_filename_tokens:
        tokens = tuple(str(token).casefold() for token in required_filename_tokens)
        paths = [path for path in paths if all(token in path.name.casefold() for token in tokens)]
    if not paths:
        return None, None, []

    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject, transform_bounds
    except ImportError as exc:
        raise RuntimeError(
            "GeoTIFF files were found, but rasterio is missing. Install it with: pip install rasterio"
        ) from exc

    south, north = lat_center - LAT_SPAN_DEG / 2.0, lat_center + LAT_SPAN_DEG / 2.0
    west, east = lon_center - LON_SPAN_DEG / 2.0, lon_center + LON_SPAN_DEG / 2.0
    rows, cols = GRID_SHAPE
    dst_transform = from_bounds(west, south, east, north, cols, rows)
    dtype = np.uint16 if categorical else np.float32
    fill_value = 0 if categorical else np.nan
    out = np.full(GRID_SHAPE, fill_value, dtype=dtype)
    valid = np.zeros(GRID_SHAPE, dtype=bool)
    used = []

    for path in paths:
        try:
            with rasterio.open(path) as ds:
                if ds.count < 1 or ds.crs is None:
                    continue
                b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=11)
                if b[2] <= west or b[0] >= east or b[3] <= south or b[1] >= north:
                    continue
                tmp = np.full(GRID_SHAPE, fill_value, dtype=dtype)
                reproject(
                    source=rasterio.band(ds, 1),
                    destination=tmp,
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    src_nodata=ds.nodata,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=fill_value,
                    resampling=Resampling.nearest if categorical else Resampling.bilinear,
                )
                tmp_valid = (tmp != 0) if categorical else np.isfinite(tmp)
                take = tmp_valid & (~valid)
                out[take] = tmp[take]
                valid |= tmp_valid
                used.append(path.name)
        except Exception as exc:
            raise RuntimeError(f"Could not read GeoTIFF {path.name}: {exc}") from exc

    if not used or not np.any(valid):
        return None, None, []
    return out, valid, used


def load_local_dem_grid(lat_center, lon_center, water_mask):
    """Đọc DEM cục bộ; chỉ cho phép thiếu dữ liệu ở ô nước."""
    dem, valid, used = _load_local_geotiff_grid(
        LOCAL_DEM_DIR, lat_center, lon_center, categorical=False
    )
    if dem is None:
        return None, None
    missing_land = (~valid) & (~np.asarray(water_mask, dtype=bool))
    if np.any(missing_land):
        pct = 100.0 * float(np.mean(valid))
        raise RuntimeError(
            f"The local DEM covers only {pct:.1f}% of the grid and has missing cells outside water. "
            f"Add the missing tiles covering the full map extent to {LOCAL_DEM_DIR}."
        )
    if np.any(~valid):
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        dem[~valid] = dem[tuple(nearest[:, ~valid])]
    return dem.astype(np.float32), "Local GeoTIFF: " + ", ".join(used)


def load_local_worldcover_grid(lat_center, lon_center, water_mask):
    """Đọc mã lớp ESA WorldCover; từ chối nếu phủ không đủ phần đất liền."""
    classes, valid, used = _load_local_geotiff_grid(
        LOCAL_WORLDCOVER_DIR, lat_center, lon_center, categorical=True,
        required_filename_tokens=("2021", "v200", "map")
    )
    if classes is None:
        if LOCAL_WORLDCOVER_DIR.exists() and any(
            path.suffix.casefold() in {".tif", ".tiff"} for path in LOCAL_WORLDCOVER_DIR.iterdir()
        ):
            raise RuntimeError(
                f"No WorldCover 2021 v200 Map tile covering the map area was found in {LOCAL_WORLDCOVER_DIR}. "
                "Place only GeoTIFF files for the correct Map layer and version in this folder; do not use InputQuality."
            )
        return None, None, "", 0.0
    water = np.asarray(water_mask, dtype=bool) | (classes == 80)
    land_domain = ~water
    coverage = float(np.mean(valid[land_domain])) * 100.0 if np.any(land_domain) else 100.0
    if coverage < 99.0:
        raise RuntimeError(
            f"ESA WorldCover covers only {coverage:.1f}% of cells not identified as water. "
            f"Add all relevant 10 m tiles to {LOCAL_WORLDCOVER_DIR}; "
            "NoData cells must not silently be treated as forest."
        )
    return classes.astype(np.uint16), water, "ESA WorldCover 2021 v200: " + ", ".join(used), coverage


def draw_thick_line(mask, r0, c0, r1, c1, thickness=2):
    rows, cols = mask.shape
    num_pts = int(max(abs(r1 - r0), abs(c1 - c0)) * 2) + 2
    rs = np.clip(np.round(np.linspace(r0, r1, num_pts)).astype(int), 0, rows - 1)
    cs = np.clip(np.round(np.linspace(c0, c1, num_pts)).astype(int), 0, cols - 1)
    for dr in range(-thickness, thickness + 1):
        for dc in range(-thickness, thickness + 1):
            if dr**2 + dc**2 <= thickness**2:
                rr = np.clip(rs + dr, 0, rows - 1)
                cc = np.clip(cs + dc, 0, cols - 1)
                mask[rr, cc] = True


def fetch_api_dem_grid(lat_center, lon_center, grid_n=20):
    """Lấy 20x20 mẫu độ cao SRTM 90 m từ OpenTopoData rồi nội suy lên 540x540 (dùng cho thí nghiệm DEM API)."""
    south, north = lat_center - LAT_SPAN_DEG / 2.0, lat_center + LAT_SPAN_DEG / 2.0
    west, east = lon_center - LON_SPAN_DEG / 2.0, lon_center + LON_SPAN_DEG / 2.0

    lats = np.linspace(north, south, grid_n)
    lons = np.linspace(west, east, grid_n)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()
    elev_results = []

    url = "https://api.opentopodata.org/v1/srtm90m"
    headers = {"User-Agent": "FirebreakOptimizationResearch/2.0 (Hanoi-VN)"}
    batch_size = 100

    with requests.Session() as session:
        for i in range(0, len(flat_lats), batch_size):
            batch_lats = flat_lats[i:i + batch_size]
            batch_lons = flat_lons[i:i + batch_size]
            locations = "|".join(
                f"{lat:.5f},{lon:.5f}"
                for lat, lon in zip(batch_lats, batch_lons)
            )
            try:
                resp = session.post(
                    url,
                    data={"locations": locations, "interpolation": "bilinear"},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"OpenTopoData SRTM 90m API request failed (batch {i // batch_size + 1}): {exc}"
                ) from exc
            except ValueError as exc:
                raise RuntimeError("OpenTopoData API returned invalid JSON.") from exc

            if payload.get("status") != "OK":
                raise RuntimeError(f"OpenTopoData API error: {payload}")

            results = payload.get("results", [])
            if len(results) != len(batch_lats):
                raise RuntimeError("OpenTopoData returned an unexpected number of elevation samples.")
            batch_elev = [item.get("elevation") for item in results]
            if any(value is None for value in batch_elev):
                raise RuntimeError("OpenTopoData returned missing elevations; refusing to synthesize terrain.")
            elev_results.extend(batch_elev)

            if i + batch_size < len(flat_lats):
                time.sleep(1.1)

    coarse_dem = np.asarray(elev_results, dtype=np.float32).reshape((grid_n, grid_n))
    scale_r = GRID_SHAPE[0] / grid_n
    scale_c = GRID_SHAPE[1] / grid_n
    dem_540 = zoom(coarse_dem, (scale_r, scale_c), order=3).astype(np.float32)
    return dem_540[:GRID_SHAPE[0], :GRID_SHAPE[1]]


def fetch_api_osm_landscape(lat_center, lon_center):
    """Đường, khu dân cư, mặt nước từ OpenStreetMap (có lưu bộ nhớ đệm trong data/osm_cache)."""
    south, north = lat_center - LAT_SPAN_DEG / 2.0, lat_center + LAT_SPAN_DEG / 2.0
    west, east   = lon_center - LON_SPAN_DEG / 2.0, lon_center + LON_SPAN_DEG / 2.0
    rows, cols   = GRID_SHAPE

    query = f"""
    [out:json][timeout:60];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track"]({south},{west},{north},{east});
      way["landuse"~"residential|commercial|industrial|retail|village|farmyard"]({south},{west},{north},{east});
      way["natural"="water"]({south},{west},{north},{east});
      way["natural"="coastline"]({south},{west},{north},{east});
      way["waterway"~"riverbank|river|canal|stream"]({south},{west},{north},{east});
      way["landuse"="reservoir"]({south},{west},{north},{east});
      way["water"~"lake|reservoir|pond|basin"]({south},{west},{north},{east});
      node["place"~"city|town|village|hamlet|isolated_dwelling|suburb"]({south},{west},{north},{east});
    );
    out geom;
    """

    headers = {
        "User-Agent": "FirebreakOptimizationResearch/2.0 (HighSchool-Academic-Research; Hanoi-VN)",
        "Accept-Encoding": "gzip, deflate"
    }

    data = None
    used_mirror = ""
    last_err = None
    cache_file = None
    if OSM_CACHE_DIR is not None:
        cache_file = FilePath(OSM_CACHE_DIR) / f"osm_{south:.4f}_{west:.4f}_{north:.4f}_{east:.4f}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("query") == query:
                data = cached["response"]
                used_mirror = f"local cache, fetched {cached.get('fetched', '?')}"

    if data is None:
        for mirror in OVERPASS_MIRRORS:
            try:
                resp = requests.post(mirror, data={"data": query}, headers=headers, timeout=(10, 75))
                if resp.status_code in (429, 504):
                    last_err = f"HTTP {resp.status_code} ({mirror})"
                    continue
                resp.raise_for_status()
                candidate = resp.json()
                remark = str(candidate.get("remark", ""))
                if "elements" not in candidate or "error" in remark.lower() or "timed out" in remark.lower():
                    raise RuntimeError(f"incomplete Overpass answer: {remark or 'no elements'}")
                data = candidate
                used_mirror = mirror.split("//")[1].split("/")[0]
                break
            except Exception as e:
                last_err = e
                continue
        if data is None:
            raise RuntimeError(f"All Overpass API servers are currently unavailable: {last_err}")
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({
                "fetched": time.strftime("%Y-%m-%d %H:%M"), "mirror": used_mirror,
                "bbox": [south, west, north, east], "query": query, "response": data,
            }), encoding="utf-8")

    road_mask = np.zeros(GRID_SHAPE, dtype=bool)
    access_mask = np.zeros(GRID_SHAPE, dtype=bool)  # chỉ để tiếp cận; vẫn cháy được
    stream_mask = np.zeros(GRID_SHAPE, dtype=bool)  # chỉ để hiển thị; không chặn lửa
    building_mask = np.zeros(GRID_SHAPE, dtype=bool)
    water_mask = np.zeros(GRID_SHAPE, dtype=bool)
    coastline_mask = np.zeros(GRID_SHAPE, dtype=bool)
    coastline_water_seeds = np.zeros(GRID_SHAPE, dtype=bool)

    def latlon_to_rc(lat, lon):
        r = (north - float(lat)) / (north - south) * rows - 0.5
        c = (float(lon) - west) / (east - west) * cols - 0.5
        return float(r), float(c)

    def clip_segment_to_grid(p0, p1):
        r0, c0 = p0
        r1, c1 = p1
        dr, dc = r1 - r0, c1 - c0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dr, r0), (dr, rows - 1 - r0), (-dc, c0), (dc, cols - 1 - c0)):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            t = q / p
            if p < 0:
                if t > t1:
                    return None
                t0 = max(t0, t)
            else:
                if t < t0:
                    return None
                t1 = min(t1, t)
        if t0 > t1:
            return None
        return ((r0 + t0 * dr, c0 + t0 * dc), (r0 + t1 * dr, c0 + t1 * dc))

    def clip_polygon_to_grid(points):
        poly = [tuple(map(float, p)) for p in points]
        boundaries = ((0, 0.0, True), (0, float(rows - 1), False),
                      (1, 0.0, True), (1, float(cols - 1), False))
        for axis, bound, keep_greater in boundaries:
            if not poly:
                break
            out = []
            prev = poly[-1]
            prev_inside = (prev[axis] >= bound) if keep_greater else (prev[axis] <= bound)
            for curr in poly:
                curr_inside = (curr[axis] >= bound) if keep_greater else (curr[axis] <= bound)
                if curr_inside != prev_inside:
                    denom = curr[axis] - prev[axis]
                    if abs(denom) > 1e-12:
                        t = (bound - prev[axis]) / denom
                        out.append((prev[0] + t * (curr[0] - prev[0]),
                                    prev[1] + t * (curr[1] - prev[1])))
                if curr_inside:
                    out.append(curr)
                prev, prev_inside = curr, curr_inside
            poly = out
        return np.asarray(poly, dtype=np.float64)

    road_count = 0
    bld_count = 0
    place_count = 0

    for el in data.get("elements", []):
        el_type = el.get("type")
        tags = el.get("tags", {})

        if el_type == "node" and "place" in tags:
            pr_f, pc_f = latlon_to_rc(el["lat"], el["lon"])
            if not (0 <= pr_f < rows and 0 <= pc_f < cols):
                continue
            place_count += 1
            bld_count += 1
            pr, pc = int(round(pr_f)), int(round(pc_f))
            ptype = tags.get("place", "village")
            rad = 15 if ptype in {"city", "town"} else (10 if ptype == "village" else 6)
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if dr**2 + dc**2 <= rad**2:
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            building_mask[nr, nc] = True

        elif el_type == "way" and "geometry" in el:
            geom = el["geometry"]
            pts = [latlon_to_rc(pt["lat"], pt["lon"])
                   for pt in geom if "lat" in pt and "lon" in pt]
            if len(pts) < 2:
                continue

            if "highway" in tags:
                road_count += 1
                hw_type = tags.get("highway", "")
                thick = 1 if hw_type in {"motorway", "trunk", "primary"} else 0
                target_mask = access_mask if hw_type in {"track", "service"} else road_mask
                for i in range(len(pts) - 1):
                    clipped = clip_segment_to_grid(pts[i], pts[i + 1])
                    if clipped is None:
                        continue
                    (r0, c0), (r1, c1) = clipped
                    draw_thick_line(target_mask, r0, c0, r1, c1, thickness=thick)

            elif tags.get("natural") == "coastline":
                for i in range(len(pts) - 1):
                    clipped = clip_segment_to_grid(pts[i], pts[i + 1])
                    if clipped is None:
                        continue
                    (r0, c0), (r1, c1) = clipped
                    draw_thick_line(coastline_mask, r0, c0, r1, c1, thickness=0)
                    dr, dc = r1 - r0, c1 - c0
                    norm = max(float(np.hypot(dr, dc)), 1e-6)
                    sr = int(round((r0 + r1) / 2.0 + 4.0 * dc / norm))
                    sc = int(round((c0 + c1) / 2.0 - 4.0 * dr / norm))
                    if 0 <= sr < rows and 0 <= sc < cols:
                        coastline_water_seeds[sr, sc] = True

            elif (tags.get("natural") == "water"
                  or tags.get("waterway") in {"riverbank", "river", "canal"}
                  or tags.get("landuse") == "reservoir"
                  or tags.get("water") in {"lake", "reservoir", "pond", "basin"}):
                if len(pts) >= 3 and (
                    np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 0.5
                ):
                    poly_rc = clip_polygon_to_grid(pts)
                    if len(poly_rc) >= 3:
                        min_r = max(0, int(np.floor(poly_rc[:, 0].min())))
                        max_r = min(rows - 1, int(np.ceil(poly_rc[:, 0].max())))
                        min_c = max(0, int(np.floor(poly_rc[:, 1].min())))
                        max_c = min(cols - 1, int(np.ceil(poly_rc[:, 1].max())))
                        if max_r >= min_r and max_c >= min_c:
                            rr, cc = np.meshgrid(
                                np.arange(min_r, max_r + 1), np.arange(min_c, max_c + 1), indexing="ij"
                            )
                            grid_pts = np.vstack((rr.ravel(), cc.ravel())).T
                            inside = Path(poly_rc).contains_points(grid_pts).reshape(rr.shape)
                            water_mask[min_r:max_r + 1, min_c:max_c + 1] |= inside
                else:
                    for i in range(len(pts) - 1):
                        clipped = clip_segment_to_grid(pts[i], pts[i + 1])
                        if clipped is None:
                            continue
                        (r0, c0), (r1, c1) = clipped
                        draw_thick_line(water_mask, r0, c0, r1, c1,
                                        thickness=1 if tags.get("waterway") == "river" else 0)

            elif tags.get("waterway") == "stream":
                for i in range(len(pts) - 1):
                    clipped = clip_segment_to_grid(pts[i], pts[i + 1])
                    if clipped is None:
                        continue
                    (r0, c0), (r1, c1) = clipped
                    draw_thick_line(stream_mask, r0, c0, r1, c1, thickness=0)

            elif "landuse" in tags and len(pts) >= 3:
                poly_rc = clip_polygon_to_grid(pts)
                if len(poly_rc) < 3:
                    continue
                bld_count += 1
                min_r = max(0, int(np.floor(poly_rc[:, 0].min())))
                max_r = min(rows - 1, int(np.ceil(poly_rc[:, 0].max())))
                min_c = max(0, int(np.floor(poly_rc[:, 1].min())))
                max_c = min(cols - 1, int(np.ceil(poly_rc[:, 1].max())))
                if max_r > min_r and max_c > min_c:
                    rr, cc = np.meshgrid(
                        np.arange(min_r, max_r + 1), np.arange(min_c, max_c + 1), indexing="ij"
                    )
                    grid_pts = np.vstack((rr.ravel(), cc.ravel())).T
                    path = Path(poly_rc)
                    inside = path.contains_points(grid_pts).reshape(rr.shape)
                    building_mask[min_r:max_r + 1, min_c:max_c + 1] |= inside

    if np.any(coastline_water_seeds) and np.any(coastline_mask):
        coast_barrier = binary_dilation(coastline_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
        components, _ = label(~coast_barrier, structure=np.ones((3, 3), dtype=np.uint8))
        seed_labels = components[coastline_water_seeds & (~coast_barrier)]
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size:
            counts = np.bincount(seed_labels, minlength=int(components.max()) + 1)
            counts[0] = 0
            ocean_label = int(np.argmax(counts))
            if ocean_label > 0:
                water_mask |= components == ocean_label
        water_mask |= coast_barrier & binary_dilation(water_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)

    building_mask[road_mask] = False
    building_mask[water_mask] = False
    water_mask[coastline_mask] = water_mask[coastline_mask] | binary_dilation(
        water_mask, structure=np.ones((3, 3), dtype=bool), iterations=1
    )[coastline_mask]
    access_mask &= ~(road_mask | water_mask)
    stream_mask &= ~water_mask
    return road_mask, building_mask, water_mask, road_count, bld_count, place_count, used_mirror, access_mask, stream_mask


def latlon_to_grid_rc(lat, lon, lat_center, lon_center):
    """(hàng, cột) của một điểm WGS84 trên lưới, cùng quy ước tâm pixel với GeoTIFF và OSM."""
    north = float(lat_center) + LAT_SPAN_DEG / 2.0
    west = float(lon_center) - LON_SPAN_DEG / 2.0
    r = (north - float(lat)) / LAT_SPAN_DEG * GRID_SHAPE[0] - 0.5
    c = (float(lon) - west) / LON_SPAN_DEG * GRID_SHAPE[1] - 0.5
    return int(round(r)), int(round(c))


# ==============================================================================
# THỜI TIẾT THEO GIỜ
# ==============================================================================
def timeline_from_csv(csv_path, t0_local):
    """Đọc file thời tiết theo giờ (từ thoi_tiet.py) thành timeline cho weather_values_at().
    t_h = số giờ tính từ thời điểm phát cháy t0_local (giờ Việt Nam, ví dụ "2024-02-19 12:00")."""
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.dropna(subset=["wind_speed", "wind_deg", "humidity", "prcp", "temp"])
    t0 = pd.Timestamp(t0_local)
    timeline = {"t_h": ((df["time"] - t0).dt.total_seconds() / 3600.0).tolist()}
    for k in ("wind_speed", "wind_deg", "humidity", "prcp", "temp"):
        timeline[k] = df[k].astype(float).tolist()
    return timeline


def weather_values_at(timeline, hour):
    """(wind_speed, wind_deg, humidity, prcp, temp) tại `hour` giờ kể từ T = 0 (giữ nguyên từ app).

    Nội suy tuyến tính tốc độ gió, độ ẩm, nhiệt độ; hướng gió nội suy theo vectơ đơn vị
    (350° sang 10° đi qua 0°, không đi qua 180°); lượng mưa ghi ở giờ H là tổng (H-1, H].
    """
    t = np.asarray(timeline["t_h"], float)
    speed = float(np.interp(hour, t, timeline["wind_speed"]))
    humidity = float(np.interp(hour, t, timeline["humidity"]))
    temp = float(np.interp(hour, t, timeline["temp"]))
    rad = np.radians(np.asarray(timeline["wind_deg"], float))
    east = float(np.interp(hour, t, np.sin(rad)))
    north = float(np.interp(hour, t, np.cos(rad)))
    direction = float(np.degrees(np.arctan2(east, north)) % 360.0)
    i = int(np.clip(np.searchsorted(t, hour, side="left"), 0, len(t) - 1))
    return speed, direction, humidity, float(timeline["prcp"][i]), temp


# ==============================================================================
# DỰNG MÔI TRƯỜNG CHO CA
# ==============================================================================
def build_env_ca(lat_center, lon_center, use_local_dem=True):
    """Giống fetch_and_build_api_environment() của app, bỏ các lớp chỉ dùng cho lập đường băng."""
    dy_m, dx_m, pixel_area_ha = compute_projected_pixel_scales(lat_center)
    (road_mask, building_mask, water_mask, road_cnt, bld_cnt, place_cnt,
     mirror_used, access_mask, stream_mask) = fetch_api_osm_landscape(lat_center, lon_center)

    if use_local_dem:
        elevation, dem_source = load_local_dem_grid(lat_center, lon_center, water_mask)
        if elevation is None:
            raise RuntimeError(f"Không tìm thấy tile DEM trong {LOCAL_DEM_DIR}")
    else:
        elevation = fetch_api_dem_grid(lat_center, lon_center, grid_n=20)
        dem_source = "OpenTopoData SRTM90m: 20x20 elevation samples interpolated to a 540x540 grid (not a 30 m DEM)"

    dy, dx = np.gradient(elevation, dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    aspect_deg = (np.degrees(np.arctan2(-dx, dy)) % 360.0).astype(np.float32)

    slope_sin = np.sin(np.radians(np.clip(slope_deg, 0.0, 60.0))).astype(np.float32)
    aspect_flammability = (
        1.0 + 0.16 * slope_sin * np.cos(np.radians(aspect_deg - 202.5))
    ).astype(np.float32)

    worldcover_classes, wc_water_mask, landcover_source, landcover_coverage_pct = load_local_worldcover_grid(
        lat_center, lon_center, water_mask
    )
    if wc_water_mask is None:
        raise RuntimeError(f"Không tìm thấy tile WorldCover 2021 v200 Map trong {LOCAL_WORLDCOVER_DIR}")
    water_mask = wc_water_mask
    class_fuel = np.zeros(GRID_SHAPE, dtype=np.float32)
    class_fuel[worldcover_classes == 10] = 1.00
    class_fuel[worldcover_classes == 20] = 0.82
    class_fuel[worldcover_classes == 30] = 0.58
    class_fuel[worldcover_classes == 40] = 0.28
    class_fuel[worldcover_classes == 90] = 0.12
    class_fuel[worldcover_classes == 95] = 0.90
    class_fuel[worldcover_classes == 100] = 0.10
    fuel_matrix = class_fuel * aspect_flammability

    # Đường đi qua nước là cầu: không cháy được; giữ đúng logic của app.
    crossing = road_mask & water_mask
    bridge = binary_dilation(crossing, structure=np.ones((3, 3), dtype=bool)) & water_mask
    road_mask = road_mask | bridge
    water_mask = water_mask & ~road_mask
    fuel_matrix[water_mask] = 0.0
    fuel_matrix[building_mask] = 0.0
    fuel_matrix[road_mask] = 0.0

    return {
        "elevation": elevation,
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "aspect_flammability": aspect_flammability,
        "fuel_matrix": fuel_matrix,
        "road_mask": road_mask,
        "access_mask": access_mask,
        "stream_mask": stream_mask,
        "building_mask": building_mask,
        "water_mask": water_mask,
        "worldcover_classes": worldcover_classes,
        "dy_m": dy_m,
        "dx_m": dx_m,
        "pixel_area_ha": pixel_area_ha,
        "dem_source": dem_source,
        "landcover_source": landcover_source,
        "landcover_coverage_pct": landcover_coverage_pct,
        "road_cnt": road_cnt,
        "bld_cnt": bld_cnt,
        "place_cnt": place_cnt,
        "mirror_used": mirror_used,
    }


def save_env(env, path):
    np.savez_compressed(path, **env)


def load_env(path):
    """Đọc lại môi trường đã lưu, đổi các giá trị đơn về đúng kiểu."""
    with np.load(path, allow_pickle=False) as data:
        env = {k: data[k] for k in data.files}
    for k in ("dy_m", "dx_m", "pixel_area_ha", "landcover_coverage_pct"):
        env[k] = float(env[k])
    for k in ("dem_source", "landcover_source", "mirror_used"):
        env[k] = str(env[k])
    return env


# ==============================================================================
# CA: MỘT BƯỚC LAN CHÁY (giữ nguyên từ app, thêm 3 hệ số hiệu chỉnh)
# ==============================================================================
_WEYL_OFFSETS = (
    0.0, 0.6180339, 0.2360679, 0.8541019,
    0.4721359, 0.0901699, 0.7082039, 0.3262379
)


def run_ca_step_vectorized(
    grid, burn_time, env, wind_speed, wind_deg, humidity, prcp, temp,
    p_base, burn_duration, rng, active_barrier_mask=None, allow_spotting=True,
    blocked_spread_mask=None, k_wind=0.085, k_wind_up=0.25, k_slope=0.035
):
    """Một bước chuyển trạng thái CA trên địa hình 3D.
    k_wind: hệ số gió xuôi; k_wind_up: hệ số gió ngược; k_slope: hệ số dốc."""
    next_grid = grid.copy()
    next_burn_time = burn_time.copy()

    road_mask = np.asarray(env["road_mask"], dtype=bool)
    building_mask = np.asarray(env["building_mask"], dtype=bool)
    water_mask = np.asarray(env.get("water_mask", np.zeros(GRID_SHAPE, dtype=bool)), dtype=bool)
    if active_barrier_mask is not None:
        active_barrier_mask = np.asarray(active_barrier_mask, dtype=bool)
        constructed_pixels = (
            active_barrier_mask & (~road_mask) & (~building_mask) & (~water_mask)
            & (grid != STATE_BURNING) & (grid != STATE_BURNED)
        )
        next_grid[constructed_pixels] = STATE_FIREBREAK

    burning_mask = (grid == STATE_BURNING)

    # Một ma trận ngẫu nhiên cho mỗi bước để đồng bộ chuỗi số ngẫu nhiên (CRN)
    base_rand_grid = rng.random(GRID_SHAPE, dtype=np.float32)
    spot_jump_seed = int(rng.integers(0, 3))

    if not np.any(burning_mask):
        return next_grid, next_burn_time

    next_burn_time[burning_mask] += 1
    burned_out = burning_mask & (next_burn_time > burn_duration)
    next_grid[burned_out] = STATE_BURNED

    w_moisture = np.exp(-0.015 * max(0.0, humidity - 30.0))
    w_rain = np.exp(-2.0 * max(0.0, prcp))
    w_temp = np.exp(0.024 * (float(temp) - 20.0))

    wind_rad = np.radians(wind_deg)
    wind_vr = np.cos(wind_rad)
    wind_vc = -np.sin(wind_rad)

    dy_m, dx_m = env["dy_m"], env["dx_m"]
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    tree_mask = (
        (next_grid == STATE_TREE) & (env["fuel_matrix"] > 0.0)
        & (~road_mask) & (~building_mask) & (~water_mask)
    )
    spread_passable = (
        (env["fuel_matrix"] > 0.0) & (~road_mask) & (~building_mask) & (~water_mask)
    )
    if active_barrier_mask is not None:
        tree_mask &= (~active_barrier_mask)
        spread_passable &= (~active_barrier_mask)
    if blocked_spread_mask is not None:
        blocked_spread_mask = np.asarray(blocked_spread_mask, dtype=bool)
        tree_mask &= (~blocked_spread_mask)
        spread_passable &= (~blocked_spread_mask)

    elev = env["elevation"]

    for dir_idx, (dr, dc) in enumerate(neighbors):
        dist_2d_m = float(np.hypot(dr * dy_m, dc * dx_m))

        shifted_burning = np.zeros_like(burning_mask)
        src_r = slice(max(0, -dr), GRID_SHAPE[0] - max(0, dr))
        src_c = slice(max(0, -dc), GRID_SHAPE[1] - max(0, dc))
        dst_r = slice(max(0, dr), GRID_SHAPE[0] - max(0, -dr))
        dst_c = slice(max(0, dc), GRID_SHAPE[1] - max(0, -dc))

        shifted_burning[dst_r, dst_c] = burning_mask[src_r, src_c]
        candidate_mask = shifted_burning & tree_mask
        if dr != 0 and dc != 0:
            # Không cho lửa lách chéo qua góc giữa hai ô không cháy được
            side_a = spread_passable[src_r, dst_c]
            side_b = spread_passable[dst_r, src_c]
            corner_ok = np.zeros_like(candidate_mask)
            corner_ok[dst_r, dst_c] = (side_a & side_b)
            candidate_mask &= corner_ok
        if not np.any(candidate_mask):
            continue

        dz_cand = elev[dst_r, dst_c][candidate_mask[dst_r, dst_c]] - elev[src_r, src_c][candidate_mask[dst_r, dst_c]]

        dist_3d_m = np.hypot(dist_2d_m, dz_cand)
        dist_norm = dist_3d_m / NOMINAL_PIXEL_M

        local_slope_deg = np.degrees(np.arctan2(dz_cand, dist_2d_m))
        w_slope = np.exp(k_slope * local_slope_deg)                 # gốc: 0.035

        vec_norm = float(np.hypot(dr, dc))
        cos_wind = (dr * wind_vr + dc * wind_vc) / vec_norm
        if cos_wind >= 0:
            w_wind = np.exp(k_wind * wind_speed * cos_wind)         # gốc: 0.085
        else:
            w_wind = np.exp(k_wind_up * wind_speed * cos_wind)      # gốc: 0.25

        p_spread = (
            p_base
            * env["fuel_matrix"][candidate_mask]
            * w_slope
            * w_wind
            * w_moisture
            * w_rain
            * w_temp
            / (dist_norm ** 0.5)
        )
        p_spread = np.clip(p_spread, 0.0, 0.95)

        rand_vals = (base_rand_grid[candidate_mask] + _WEYL_OFFSETS[dir_idx]) % 1.0
        ignited = rand_vals < p_spread
        if np.any(ignited):
            cand_indices = np.argwhere(candidate_mask)
            ignited_coords = cand_indices[ignited]
            next_grid[ignited_coords[:, 0], ignited_coords[:, 1]] = STATE_BURNING
            next_burn_time[ignited_coords[:, 0], ignited_coords[:, 1]] = 1

    if wind_speed > 8.0 and allow_spotting:
        spot_rate = 0.02 * min(1.8, float(w_temp))
        spot_vals = (base_rand_grid[burning_mask] + 0.5142286) % 1.0
        burning_coords = np.argwhere(burning_mask)
        spotters = burning_coords[spot_vals < spot_rate]
        for r, c in spotters:
            jump_dist = 2 + ((int(r) * 7 + int(c) * 13 + spot_jump_seed) % 3)
            jr = r + int(round(wind_vr * jump_dist))
            jc = c + int(round(wind_vc * jump_dist))
            if 0 <= jr < GRID_SHAPE[0] and 0 <= jc < GRID_SHAPE[1]:
                if tree_mask[jr, jc]:
                    next_grid[jr, jc] = STATE_BURNING
                    next_burn_time[jr, jc] = 1

    return next_grid, next_burn_time


# ==============================================================================
# CHẠY TRỰC TIẾP: dựng và lưu môi trường cho ROI Hoàng Liên
# ==============================================================================
if __name__ == "__main__":
    LAT_C, LON_C = 22.240, 103.875           # tâm ROI Hoàng Liên (đã chốt 9/11/2026)

    env = build_env_ca(LAT_C, LON_C, use_local_dem=True)
    save_env(env, "research/env_hoang_lien.npz")
    print("=== Môi trường chính (DEM cục bộ) ===")
    print("Nguồn DEM:", env["dem_source"])
    print("Nguồn lớp phủ:", env["landcover_source"], f"({env['landcover_coverage_pct']:.1f}%)")
    print("Nguồn OSM:", env["mirror_used"])
    print(f"Độ cao: {env['elevation'].min():.0f} - {env['elevation'].max():.0f} m, "
          f"độ dốc trung bình {env['slope_deg'].mean():.1f} độ")
    print("Số ô cháy được:", int((env["fuel_matrix"] > 0).sum()), "/", GRID_SHAPE[0] * GRID_SHAPE[1])
    print("Số ô nước:", int(env["water_mask"].sum()),
          "| đường:", int(env["road_mask"].sum()),
          "| dân cư:", int(env["building_mask"].sum()))

    r0, c0 = latlon_to_grid_rc(22.24626, 103.87037, LAT_C, LON_C)   # tâm 3 điểm VIIRS đầu tiên
    print(f"Ô phát cháy: hàng {r0}, cột {c0}; nhiên liệu = {env['fuel_matrix'][r0, c0]:.2f}; "
          f"lớp WorldCover = {int(env['worldcover_classes'][r0, c0])}")

    # Môi trường dùng DEM thô từ API, cho thí nghiệm "DEM API" ở CH2
    env_api = build_env_ca(LAT_C, LON_C, use_local_dem=False)
    save_env(env_api, "research/env_hoang_lien_demapi.npz")
    print("=== Môi trường DEM API ===")
    print(f"Độ cao: {env_api['elevation'].min():.0f} - {env_api['elevation'].max():.0f} m, "
          f"độ dốc trung bình {env_api['slope_deg'].mean():.1f} độ")
    print("Đã lưu env_hoang_lien.npz và env_hoang_lien_demapi.npz")