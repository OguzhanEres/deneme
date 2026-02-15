
import cv2
import numpy as np
import os
import json
import time

import bisect
import pathlib
import re as _re

def _parse_dji_srt(srt_path: str):
    """Parse DJI .SRT and return sorted list of (t_sec, lat, lon, alt_m, yaw_deg|None).
    Robust to different DJI formats. Missing fields become None.
    """
    p = pathlib.Path(srt_path)
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8", errors="ignore")
    entries = []
    # Split by blank lines
    blocks = _re.split(r"\n\s*\n", txt.strip())
    time_re = _re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d\d\d)")
    # Examples seen:
    # GPS (lat, lon, alt) or GPS: 39.1234, 32.1234, 120.3
    gps_re1 = _re.compile(r"GPS\s*\(?\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*\)?", _re.IGNORECASE)
    gps_re2 = _re.compile(r"\bLat\b\s*[:=]\s*([-+]?\d+\.\d+).*?\bLon\b\s*[:=]\s*([-+]?\d+\.\d+).*?\bAlt\b\s*[:=]\s*([-+]?\d+\.?\d*)", _re.IGNORECASE|_re.DOTALL)
    # Yaw/Heading formats vary
    yaw_re = _re.compile(r"\b(?:Yaw|Heading|Hdg)\b\s*[:=]\s*([-+]?\d+\.?\d*)", _re.IGNORECASE)

    def to_sec(h,m,s,ms):
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

    for b in blocks:
        m = time_re.search(b)
        if not m:
            continue
        t0 = to_sec(m.group(1),m.group(2),m.group(3),m.group(4))
        lat=lon=alt=yaw=None
        g = gps_re1.search(b) or gps_re2.search(b)
        if g:
            try:
                lat = float(g.group(1)); lon = float(g.group(2)); alt = float(g.group(3))
            except Exception:
                pass
        y = yaw_re.search(b)
        if y:
            try:
                yaw = float(y.group(1))
            except Exception:
                pass
        if lat is not None and lon is not None:
            entries.append((t0, lat, lon, alt, yaw))
    entries.sort(key=lambda x: x[0])
    return entries

def _srt_lookup(entries, t_sec: float):
    """Return nearest SRT entry to t_sec."""
    if not entries:
        return None
    ts = [e[0] for e in entries]
    i = bisect.bisect_left(ts, t_sec)
    if i <= 0:
        return entries[0]
    if i >= len(entries):
        return entries[-1]
    before = entries[i-1]
    after = entries[i]
    return before if (t_sec - before[0]) <= (after[0] - t_sec) else after

# ---- GPU (OpenCL) Setup ----
# ---- GPU (OpenCL) Setup ----
_USE_GPU = False
# if cv2.ocl.haveOpenCL():
#     cv2.ocl.setUseOpenCL(True)
#     _USE_GPU = cv2.ocl.useOpenCL()
#     _dev = cv2.ocl.Device.getDefault()
#     print(f"[GPU] OpenCL enabled  –  {_dev.name()}")
# else:
#     print("[GPU] OpenCL not available – running on CPU")

def _to_umat(arr):
    """Upload numpy array to GPU (UMat). No-op if OpenCL is off."""
    if _USE_GPU:
        return cv2.UMat(arr)
    return arr

def _to_numpy(m):
    """Download UMat back to numpy. Safe for plain ndarray too."""
    if isinstance(m, cv2.UMat):
        return m.get()
    return m

# Constants
MAP_DIR = "processed_map"
VIDEO_PATH = "DJIG0022.mov"
COARSE_SCALES = [0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6]
# Use a smaller rotation range for faster testing, or full range if needed
ROTATION_RANGE = 6
ROTATION_STEP = 3
SEARCH_WINDOW = 1 # Adjacent tiles to search when locked
LOCK_WINDOW = 5 # Frames to keep lock
SKIP_SECONDS = 35 # Skip takeoff sequence

# --- Uniqueness gate thresholds (ratio-based + absolute) ---
# --- Uniqueness gate thresholds (ratio-based + absolute) ---
UNIQUENESS_MEDIAN_RATIO = 1.25   # median2/median1 must be >= this (INCREASED to reduce false positives)
UNIQUENESS_MEDIAN_DIFF  = 0.4    # OR median2 - median1 must be >= this (INCREASED)
UNIQUENESS_CORE_DIFF    = 0.05   # core_ratio1 - core_ratio2 must be >= this (INCREASED)
UNIQUENESS_SCORE_RATIO  = 1.10   # score1/score2 must be >= this (INCREASED)

# --- Orientation gate (tightened for GT mode) ---
ORIENTED_CORE_RATIO_MIN  = 0.10  # min fraction of edges with matching gradient direction
GRADIENT_ANGLE_THRESH_DEG = 15.0 # max |delta_theta| for oriented inlier

# --- Gray NCC secondary score ---
GRAY_NCC_WEIGHT   = 0.6          # weight for grayscale NCC (INCREASED for better texture discrimination)
EDGE_SCORE_WEIGHT = 0.4          # weight for edge/DT coarse score (DECREASED)

# --- Line-only verify ---
LINE_MIN_LENGTH = 200            # min line segment length (px) – raised for road/canal only
LINE_TOP_N      = 20             # keep only this many longest lines

# --- Multi-patch consistency (anchor + local refine) ---
MULTI_PATCH_ENABLED  = True      # enable 5-patch position consistency check
MULTI_PATCH_MARGIN   = 0.15      # fraction from each edge for corner patches
MULTI_PATCH_POS_THR  = 150        # max pos_std across patches (pixels) - TIGHTENED
MULTI_PATCH_ANG_THR  = 5.0       # max std(angle) across patches (degrees) - TIGHTENED
MULTI_PATCH_LOCAL_R  = 256       # search radius around anchor (map pixels)
MULTI_PATCH_SIZE     = 0.60      # patch size as fraction of frame dimension

# --- Coarse (x,y) spatial clustering ---
COARSE_TOP_K        = 20         # take more candidates for clustering
COARSE_CLUSTER_BIN  = 64         # quantization grid for (x,y) clustering

# --- Edge accumulation (multi-frame) ---
EDGE_ACCUM_N        = 5          # number of recent frames to OR together

# --- NEW: Temporal N-frame lock ---
TEMPORAL_LOCK_N = 3  # tracking lock
GT_LOCK_N = 2        # GT write lock             # consecutive consistent frames before GT write
TEMPORAL_MAX_JUMP_M = 30.0       # max jump in metres between consecutive frames
TEMPORAL_MAX_ANGLE_DIFF = 10.0   # max rotation change between consecutive frames

class DroneLocalizer:
    def __init__(self, map_dir):
        self.map_dir = map_dir
        self.tiles = []
        self.load_map_metadata()
        # --- Optional: DJI SRT GPS prior for clean GT (recommended for XFeat) ---
        self.srt_entries = []
        self._prior_px = None   # (global_px_x, global_px_y)
        self.prior_roi_px = 1100  # ~250m @0.23m/px, tune
        srt_guess = str(pathlib.Path(VIDEO_PATH).with_suffix(".SRT"))
        if os.path.exists(srt_guess):
            self.srt_entries = _parse_dji_srt(srt_guess)
            if self.srt_entries:
                print(f"[SRT] Loaded {len(self.srt_entries)} GPS samples from {srt_guess}")
            else:
                print(f"[SRT] Found {srt_guess} but could not parse GPS.")
        else:
            # No SRT present -> pure vision fallback
            pass

        
        self.state = "LOST" 
        self.lock_counter = 0
        
        # Last known good position
        self.last_tile_idx = -1
        self.last_scale = None
        self.last_pose = None # {x, y, scale, angle} relative to tile
        self.last_global_pos = None  # (gx, gy) in map pixel coords
        
        # Cache
        self.tile_cache = {}
        
        # Edge accumulation ring buffer
        self._edge_ring = []         # list of (edges_np, scale_info)

    def load_map_metadata(self):
        meta_path = os.path.join(self.map_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Map metadata not found at {meta_path}")
        
        with open(meta_path, "r") as f:
            data = json.load(f)
            self.tiles = data["tiles"]
            self.stride = data["stride"]
            self.tile_size = data["tile_size"]
            self.transform = data.get("transform")
            print(f"Loaded {len(self.tiles)} tiles.")
            
            # --- GeoTIFF transform sanity check ---
            if self.transform:
                a, b, c, d, e, f = self.transform
                pix_w = abs(a) if abs(a) > 1e-12 else abs(b)
                pix_h = abs(e) if abs(e) > 1e-12 else abs(d)
                if pix_w < 1e-12 and pix_h < 1e-12:
                    print("\n" + "="*70)
                    print("[WARNING] GeoTIFF transform has ZERO pixel size!")
                    print(f"  transform = {self.transform}")
                    print("  lat/lon output will be MEANINGLESS.")
                    print("  Fix: re-georeference your TIFF in QGIS or with gdal_translate.")
                    print("="*70 + "\n")
                else:
                    print(f"[GEO] pixel size ~ {pix_w:.10f} x {pix_h:.10f} deg/px")


    def pixel_to_geo(self, px_x: float, px_y: float):
        """Convert global pixel (x,y) to (lon,lat) using GeoTIFF affine."""
        if not self.transform:
            return None
        a,b,c,d,e,f = self.transform
        lon = a*px_x + b*px_y + c
        lat = d*px_x + e*px_y + f
        return lon, lat

    def geo_to_pixel(self, lon: float, lat: float):
        """Invert affine to get global pixel from (lon,lat)."""
        if not self.transform:
            return None
        a,b,c,d,e,f = self.transform
        # Solve:
        # lon = a*x + b*y + c
        # lat = d*x + e*y + f
        det = a*e - b*d
        if abs(det) < 1e-18:
            return None
        x = ( e*(lon - c) - b*(lat - f) ) / det
        y = ( -d*(lon - c) + a*(lat - f) ) / det
        return float(x), float(y)

    def tiles_in_roi(self, center_px, roi_px: int):
        """Return tile indices whose bbox intersects ROI box around center_px."""
        cx, cy = center_px
        x0 = cx - roi_px; y0 = cy - roi_px
        x1 = cx + roi_px; y1 = cy + roi_px
        idxs = []
        for i,t in enumerate(self.tiles):
            tx0 = t["x"]; ty0 = t["y"]
            tx1 = tx0 + self.tile_size
            ty1 = ty0 + self.tile_size
            if tx1 < x0 or tx0 > x1 or ty1 < y0 or ty0 > y1:
                continue
            idxs.append(i)
        return idxs
    def get_tile_data(self, tile_idx):
        if tile_idx in self.tile_cache:
            return self.tile_cache[tile_idx]
            
        tile_info = self.tiles[tile_idx]
        tile_id = tile_info["id"]
        path = os.path.join(self.map_dir, tile_id)
        
        dt = np.load(os.path.join(path, "dt.npy"))
        
        # DT inversion for coarse template matching
        # Larger sigma needed with sparser edges so the "glow" extends further
        sigma = 30.0
        dt_norm = np.exp(-dt / sigma) * 255
        inv_dt = dt_norm.astype(np.uint8)
        
        # Load Gray for Visualization
        gray_path = os.path.join(path, "gray.png")
        if os.path.exists(gray_path):
            gray = cv2.imread(gray_path, 0)
        else:
            gray = np.zeros_like(dt, dtype=np.uint8)
            
        # Load Lines
        lines_path = os.path.join(path, "lines.npy")
        if os.path.exists(lines_path):
            lines = np.load(lines_path)
        else:
            lines = None

        data = {"inv_dt": inv_dt, "dt": dt, "info": tile_info, "gray": gray, "lines": lines,
                # Pre-upload to GPU once – reused every frame
                "inv_dt_gpu": _to_umat(inv_dt),
                # Pre-compute Sobel theta on grayscale (used by orientation gate)
                "theta_map": self._precompute_theta_map(gray),
                }
        self.tile_cache[tile_idx] = data
        return data

    @staticmethod
    def _precompute_theta_map(gray):
        """Compute gradient direction map once per tile (on CPU, cached)."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.arctan2(gy, gx)  # -pi..pi

    def _build_roi_mask(self, h, w):
        """Static ROI mask: zeros out propeller/body regions.
        Adjust the polygon coordinates to match your airframe.
        Convention: 255 = keep, 0 = mask-out.
        Frame is already rotated 90° CW at this point.
        More aggressive: bottom 20% full strip + larger propeller arcs.
        """
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # --- bottom 20% full-width strip (body + landing gear) ---
        body_top = int(h * 0.80)
        mask[body_top:, :] = 0

        # --- left propeller arc (bigger) ---
        cv2.ellipse(mask, (int(w * 0.08), int(h * 0.75)),
                    (int(w * 0.15), int(h * 0.15)), 0, 0, 360, 0, -1)

        # --- right propeller arc (bigger) ---
        cv2.ellipse(mask, (int(w * 0.92), int(h * 0.75)),
                    (int(w * 0.15), int(h * 0.15)), 0, 0, 360, 0, -1)

        # --- top corners (sometimes props visible) ---
        cv2.ellipse(mask, (int(w * 0.08), int(h * 0.05)),
                    (int(w * 0.10), int(h * 0.08)), 0, 0, 360, 0, -1)
        cv2.ellipse(mask, (int(w * 0.92), int(h * 0.05)),
                    (int(w * 0.10), int(h * 0.08)), 0, 0, 360, 0, -1)

        return mask

    def preprocess_frame(self, frame):
        if frame is None: return None, None, None, 0, 0
        # --- GPU path: upload once, chain OpenCV calls on UMat ---
        gray_gpu = cv2.cvtColor(_to_umat(frame), cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gpu = clahe.apply(gray_gpu)
        # Sharpen
        gaussian_gpu = cv2.GaussianBlur(enhanced_gpu, (0, 0), 3.0)
        enhanced_gpu = cv2.addWeighted(enhanced_gpu, 1.5, gaussian_gpu, -0.5, 0)

        # Blur before Canny – tighter thresholds to reduce noise edges
        blurred_gpu = cv2.GaussianBlur(enhanced_gpu, (5, 5), 0)
        edges_gpu = cv2.Canny(blurred_gpu, 80, 200)
        edge_count_before = int(cv2.countNonZero(edges_gpu))

        # Apply propeller / body mask
        h, w = _to_numpy(edges_gpu).shape[:2] if isinstance(edges_gpu, cv2.UMat) else edges_gpu.shape[:2]
        roi_mask = self._build_roi_mask(h, w)
        edges_gpu = cv2.bitwise_and(edges_gpu, _to_umat(roi_mask))

        # Download to numpy for later point-wise ops
        enhanced = _to_numpy(enhanced_gpu)
        edges = _to_numpy(edges_gpu)

        # Remove small connected-component blobs (area < 30 px)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < 30:
                edges[labels == lbl] = 0
        edge_count_after = int(np.count_nonzero(edges))

        lsd = cv2.createLineSegmentDetector(0)
        lines, _, _, _ = lsd.detect(enhanced)

        return enhanced, edges, lines, edge_count_before, edge_count_after

    def _get_center_patch(self, edges, crop_factor=0.6):
        """
        Extract center patch for 'Anchor' coarse search.
        Returns:
            patch_edges: The cropped edge image
            offset_x, offset_y: The top-left coordinate of the crop in the original frame
        """
        h, w = edges.shape
        ph = int(h * crop_factor)
        pw = int(w * crop_factor)
        
        y1 = (h - ph) // 2
        x1 = (w - pw) // 2
        
        patch_edges = edges[y1:y1+ph, x1:x1+pw].copy()
        return patch_edges, x1, y1


    def coarse_search(self, frame_edges, top_k=5, prior_global_px=None, prior_roi_px=1100, frame_gray=None):
        """
        Match frame_edges against tiles.  GPU-accelerated via UMat.
        Uses combined edge/DT + grayscale NCC scoring for farmland robustness.
        Returns Top-K matches list: [{tile_idx, score, ...}, ...]
        """
        # --- ROI GATING (best for clean GT): if prior is available, restrict GLOBAL/LOST search to tiles overlapping ROI ---
        roi_tile_idxs = None
        if prior_global_px is not None:
            try:
                roi_tile_idxs = self.tiles_in_roi(prior_global_px, int(prior_roi_px))
                if len(roi_tile_idxs) == 0:
                    roi_tile_idxs = None
            except Exception:
                roi_tile_idxs = None

        candidates = []

        # Determine search indices
        if self.state == "LOCKED" and self.last_tile_idx != -1:
             current_tile = self.tiles[self.last_tile_idx]
             cx, cy = current_tile["x"], current_tile["y"]
             search_indices = []
             for i, t in enumerate(self.tiles):
                 if abs(t["x"] - cx) <= self.stride * (SEARCH_WINDOW + 1) and \
                    abs(t["y"] - cy) <= self.stride * (SEARCH_WINDOW + 1):
                     search_indices.append(i)
        else:
            search_indices = range(len(self.tiles))

        if roi_tile_idxs is not None:
            search_indices = roi_tile_idxs

        use_gray = frame_gray is not None

        # Precompute rotated templates (keep on GPU)
        templates = []
        h, w = frame_edges.shape
        frame_gpu = _to_umat(frame_edges)
        if use_gray:
            gray_gpu = _to_umat(frame_gray)

        for scale in COARSE_SCALES:
            scaled_w, scaled_h = int(w*scale), int(h*scale)
            if scaled_w == 0 or scaled_h == 0: continue

            resized_gpu = cv2.resize(frame_gpu, (scaled_w, scaled_h))
            if use_gray:
                gray_resized_gpu = cv2.resize(gray_gpu, (scaled_w, scaled_h))

            for angle in range(-ROTATION_RANGE, ROTATION_RANGE + 1, ROTATION_STEP):
                 M = cv2.getRotationMatrix2D((scaled_w//2, scaled_h//2), angle, 1.0)
                 rotated_gpu = cv2.warpAffine(resized_gpu, M, (scaled_w, scaled_h),
                                              flags=cv2.INTER_NEAREST,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=0)
                 # Downsampled template on GPU (0.25x)
                 small_tmpl_gpu = cv2.resize(rotated_gpu, (0, 0), fx=0.25, fy=0.25)

                 tmpl_entry = {
                     "img_gpu": small_tmpl_gpu,
                     "scale": scale,
                     "angle": angle,
                     "h": scaled_h,
                     "w": scaled_w,
                     "sh": _to_numpy(small_tmpl_gpu).shape[0],
                     "sw": _to_numpy(small_tmpl_gpu).shape[1],
                 }

                 if use_gray:
                     gray_rotated_gpu = cv2.warpAffine(gray_resized_gpu, M, (scaled_w, scaled_h),
                                                       flags=cv2.INTER_LINEAR,
                                                       borderMode=cv2.BORDER_CONSTANT,
                                                       borderValue=0)
                     tmpl_entry["gray_gpu"] = cv2.resize(gray_rotated_gpu, (0, 0), fx=0.25, fy=0.25)

                 templates.append(tmpl_entry)

        # Match against tiles (GPU matchTemplate)
        all_candidates = []

        for tile_idx in search_indices:
            tile_data = self.get_tile_data(tile_idx)
            inv_dt_gpu = tile_data["inv_dt_gpu"]

            # Downsample tile on GPU
            small_inv_dt_gpu = cv2.resize(inv_dt_gpu, (0,0), fx=0.25, fy=0.25)
            si_h = _to_numpy(small_inv_dt_gpu).shape[0]
            si_w = _to_numpy(small_inv_dt_gpu).shape[1]

            # Downsample tile gray for NCC
            if use_gray:
                small_gray_tile_gpu = cv2.resize(_to_umat(tile_data["gray"]), (0,0), fx=0.25, fy=0.25)

            for t in templates:
                if t["sh"] > si_h or t["sw"] > si_w:
                    continue

                # Edge/DT matchTemplate
                res_gpu = cv2.matchTemplate(small_inv_dt_gpu, t["img_gpu"],
                                            cv2.TM_CCOEFF_NORMED)
                _, edge_score, _, max_loc = cv2.minMaxLoc(res_gpu)

                # Grayscale NCC at the same location
                combined_score = edge_score
                if use_gray and "gray_gpu" in t:
                    gray_res = cv2.matchTemplate(small_gray_tile_gpu, t["gray_gpu"],
                                                 cv2.TM_CCOEFF_NORMED)
                    _, gray_score, _, gray_loc = cv2.minMaxLoc(gray_res)
                    combined_score = EDGE_SCORE_WEIGHT * edge_score + GRAY_NCC_WEIGHT * gray_score

                # Track best score for debug
                if not hasattr(self, '_best_score_seen'):
                    self._best_score_seen = 0.0
                if combined_score > self._best_score_seen:
                    self._best_score_seen = combined_score

                if combined_score > 0.1:
                    x = max_loc[0] * 4
                    y = max_loc[1] * 4
                    all_candidates.append({
                        "tile_idx": tile_idx,
                        "score": combined_score,
                        "x": x,
                        "y": y,
                        "scale": t["scale"],
                        "angle": t["angle"],
                        "w": t["w"],
                        "h": t["h"]
                    })

        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate: keep best score per ~global position (across overlapping tiles)
        seen = set()
        deduped = []
        for c in all_candidates:
            tile_info = self.tiles[c["tile_idx"]]
            gx = (tile_info["x"] + c["x"]) // 50
            gy = (tile_info["y"] + c["y"]) // 50
            key = (gx, gy, c["angle"])
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        if not hasattr(self, '_coarse_debug_done'):
            sample_dt = self.get_tile_data(0)["inv_dt"]
            top_score = all_candidates[0]["score"] if all_candidates else 0
            print(f"[COARSE DEBUG] raw={len(all_candidates)} deduped={len(deduped)} top_score={top_score:.3f} "
                  f"best_score_seen={self._best_score_seen:.4f} "
                  f"inv_dt min={sample_dt.min()} max={sample_dt.max()} mean={sample_dt.mean():.1f} "
                  f"templates={len(templates)} tiles={len(list(search_indices))}")
            self._coarse_debug_done = True

        # --- Spatial (x,y) clustering: pick dominant peak ---
        top_pool = deduped[:COARSE_TOP_K]
        if len(top_pool) >= 3:
            top_pool = self._cluster_coarse_candidates(top_pool)

        return top_pool[:top_k]

    # ------------------------------------------------------------------
    #  Local refinement: fine-tune position + angle after coarse search
    # ------------------------------------------------------------------
    def _refine_candidate(self, cand, frame_edges):
        """Refine coarse match by testing small (x,y) shifts and angle
        perturbations at full resolution against the DT.
        Returns a new candidate dict with refined x, y, angle."""
        tile_data = self.get_tile_data(cand["tile_idx"])
        dt = tile_data["dt"]

        h, w = frame_edges.shape
        scale = float(cand["scale"])
        base_x, base_y = int(cand["x"]), int(cand["y"])
        base_angle = float(cand["angle"])

        scaled_w, scaled_h = int(w * scale), int(h * scale)
        if scaled_w < 8 or scaled_h < 8:
            return cand

        best_med = float('inf')
        best_params = (base_x, base_y, base_angle)

        # Search grid: ±12px position, ±2° angle
        shifts = [-12, -8, -4, 0, 4, 8, 12]
        angle_deltas = [-2, -1, 0, 1, 2]

        for da in angle_deltas:
            angle = base_angle + da
            resized = cv2.resize(frame_edges, (scaled_w, scaled_h),
                                 interpolation=cv2.INTER_NEAREST)
            M = cv2.getRotationMatrix2D((scaled_w // 2, scaled_h // 2), angle, 1.0)
            rotated = cv2.warpAffine(resized, M, (scaled_w, scaled_h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rotated = (rotated > 0).astype(np.uint8) * 255
            edge_y, edge_x = np.where(rotated > 0)
            if edge_x.size < 500:
                continue

            for dx in shifts:
                for dy in shifts:
                    cx = base_x + dx
                    cy = base_y + dy
                    map_y = edge_y + cy
                    map_x = edge_x + cx
                    valid = ((map_x >= 0) & (map_x < dt.shape[1]) &
                             (map_y >= 0) & (map_y < dt.shape[0]))
                    vi = np.nonzero(valid)[0]
                    if vi.size < 500:
                        continue
                    dists = dt[map_y[vi], map_x[vi]].astype(np.float32)
                    sorted_d = np.sort(dists)
                    med = float(np.median(sorted_d[:min(2000, len(sorted_d))]))
                    if med < best_med:
                        best_med = med
                        best_params = (cx, cy, angle)

        refined = cand.copy()
        refined["x"] = best_params[0]
        refined["y"] = best_params[1]
        refined["angle"] = best_params[2]
        return refined

    # ------------------------------------------------------------------
    #  Coarse candidate spatial clustering
    # ------------------------------------------------------------------
    def _cluster_coarse_candidates(self, cands):
        """Bin candidates by (x//BIN, y//BIN), pick the densest bin,
        return only candidates in that bin (best score first)."""
        bins = {}
        for c in cands:
            key = (c["tile_idx"], c["x"] // COARSE_CLUSTER_BIN, c["y"] // COARSE_CLUSTER_BIN)
            bins.setdefault(key, []).append(c)
        # Find densest bin
        best_key = max(bins, key=lambda k: len(bins[k]))
        cluster = bins[best_key]
        # Sort by score within cluster
        cluster.sort(key=lambda x: x["score"], reverse=True)
        # Add best from other bins as fallback (for uniqueness gate)
        others = []
        for k, v in bins.items():
            if k != best_key:
                others.append(v[0])  # best from each other bin
        others.sort(key=lambda x: x["score"], reverse=True)
        return cluster + others

    # ------------------------------------------------------------------
    #  Temporal N-frame lock
    # ------------------------------------------------------------------
    def check_temporal_consistency(self, current_match, history,
                                   max_dist_m=None, max_angle_diff=None, required_n=None):
        """
        Returns True ONLY when the last n entries in *history*
        (including current_match appended conceptually) satisfy:
          - same tile or neighbour tile
          - lat/lon jump  < TEMPORAL_MAX_JUMP_M  (≈ metres)
          - rotation jump < TEMPORAL_MAX_ANGLE_DIFF
        history is expected to already contain previous accepted candidates.
        """
        if max_dist_m is None:
            max_dist_m = TEMPORAL_MAX_JUMP_M
        if max_angle_diff is None:
            max_angle_diff = TEMPORAL_MAX_ANGLE_DIFF

        if required_n is None:
            n = TEMPORAL_LOCK_N
        else:
            n = int(required_n)

        # Need at least N-1 previous + current = N
        if len(history) < n - 1:
            return False   # not enough frames yet → don't write GT

        # Build window = last (N-1) history + current
        window = history[-(n - 1):] + [current_match]

        for i in range(1, len(window)):
            prev = window[i - 1]
            curr = window[i]

            # --- Distance (degrees → metres, rough) ---
            d_lat = curr['lat'] - prev['lat']
            d_lon = curr['lon'] - prev['lon']
            # 1° lat ≈ 111 320 m, 1° lon ≈ 111 320 * cos(lat)
            cos_lat = np.cos(np.radians(curr['lat']))
            dist_m = np.sqrt((d_lat * 111320) ** 2 + (d_lon * 111320 * cos_lat) ** 2)
            if dist_m > max_dist_m:
                return False

            # --- Angle ---
            diff_angle = abs(curr['angle'] - prev['angle'])
            if diff_angle > 180:
                diff_angle = 360 - diff_angle
            if diff_angle > max_angle_diff:
                return False

        return True

    # ------------------------------------------------------------------
    #  Orientation (gradient direction) verification
    # ------------------------------------------------------------------
    def _compute_oriented_core_ratio(self, tile_data, frame_edges,
                                      match, frame_enhanced=None,
                                      angle_thresh_deg=None):
        """
        For each transformed frame-edge pixel that lands on / near a map edge:
        compare the Sobel gradient direction θ_frame vs θ_map.
        Returns the fraction whose |Δθ| < threshold  ("oriented inlier ratio").

        IMPORTANT: Sobel is computed on GRAYSCALE images (not binary edges, not DT).
        frame_enhanced = CLAHE-enhanced grayscale of frame.
        gray_map       = grayscale of the tile.
        """
        if angle_thresh_deg is None:
            angle_thresh_deg = GRADIENT_ANGLE_THRESH_DEG
        dt = tile_data["dt"]
        gray_map = tile_data["gray"]          # grayscale tile (NOT dt)

        # Use enhanced grayscale for frame gradient; fall back to edges if unavailable
        frame_src = frame_enhanced if frame_enhanced is not None else frame_edges

        h, w = frame_edges.shape
        scale = float(match["scale"])
        angle = float(match["angle"])
        x, y = int(match["x"]), int(match["y"])

        scaled_w, scaled_h = int(w * scale), int(h * scale)
        if scaled_w < 8 or scaled_h < 8:
            return 0.0

        # --- Sobel on map GRAYSCALE tile (cached!) ---
        theta_map = tile_data["theta_map"]

        # --- Sobel on frame GRAYSCALE (not binary!) ---
        gx_fr = cv2.Sobel(frame_src.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy_fr = cv2.Sobel(frame_src.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        theta_frame_raw = np.arctan2(gy_fr, gx_fr)

        # Scale + rotate frame edges (nearest) to locate edge positions
        resized = cv2.resize(frame_edges, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        M = cv2.getRotationMatrix2D((scaled_w // 2, scaled_h // 2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (scaled_w, scaled_h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated = (rotated > 0).astype(np.uint8) * 255

        # Also warp theta_frame with the same transform
        theta_fr_resized = cv2.resize(theta_frame_raw, (scaled_w, scaled_h),
                                       interpolation=cv2.INTER_NEAREST)
        theta_fr_rot = cv2.warpAffine(theta_fr_resized, M, (scaled_w, scaled_h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Compensate rotation offset (angle in degrees → radians)
        theta_fr_rot = theta_fr_rot + np.radians(angle)

        edge_y, edge_x = np.where(rotated > 0)
        if edge_x.size < 200:
            return 0.0

        map_y = edge_y + y
        map_x = edge_x + x
        valid = ((map_x >= 0) & (map_x < dt.shape[1]) &
                 (map_y >= 0) & (map_y < dt.shape[0]))
        vi = np.nonzero(valid)[0]
        if vi.size < 200:
            return 0.0

        # Only look at pixels near a map edge (DT < 2.0, tightened)
        dists = dt[map_y[vi], map_x[vi]]
        near = dists < 2.0
        vi_near = vi[near]
        if vi_near.size < 100:
            return 0.0

        th_f = theta_fr_rot[edge_y[vi_near], edge_x[vi_near]]
        th_m = theta_map[map_y[vi_near], map_x[vi_near]]

        # --- Proper angle wrapping (direction-agnostic: mod π) ---
        delta = np.abs(th_f - th_m)
        # First wrap to [0, 2π]
        delta = delta % (2.0 * np.pi)
        # Then to [0, π]  (opposite directions are the same line)
        delta = np.minimum(delta, 2.0 * np.pi - delta)
        delta = np.minimum(delta, np.pi - np.minimum(delta, np.abs(np.pi - delta)))
        # Simplified: smallest angle in [0, π/2]
        # a line at 10° and 190° should give 0°
        delta = delta % np.pi
        delta = np.minimum(delta, np.pi - delta)

        oriented_inlier = np.count_nonzero(delta < np.radians(angle_thresh_deg))
        return float(oriented_inlier / vi_near.size)

    # ------------------------------------------------------------------
    #  Gray NCC secondary score (block-wise for robustness)
    # ------------------------------------------------------------------
    @staticmethod
    def _ncc_of_pair(a, b, mask):
        """Pearson correlation on masked pixels. Both a, b are uint8 arrays."""
        n = np.count_nonzero(mask)
        if n < 64:
            return 0.0
        av = a[mask].astype(np.float64)
        bv = b[mask].astype(np.float64)
        av -= av.mean()
        bv -= bv.mean()
        d = np.sqrt(np.sum(av ** 2) * np.sum(bv ** 2))
        if d < 1e-8:
            return 0.0
        return float(np.sum(av * bv) / d)

    def _compute_gray_ncc(self, tile_data, frame_enhanced, match):
        """Block-wise NCC: split overlap into 3×3 sub-blocks, take trimmed mean."""
        gray_map = tile_data["gray"]
        if gray_map is None or np.count_nonzero(gray_map) < 100:
            return 0.0

        h, w = frame_enhanced.shape[:2]
        scale = float(match["scale"])
        angle = float(match["angle"])
        x, y = int(match["x"]), int(match["y"])

        scaled_w, scaled_h = int(w * scale), int(h * scale)
        if scaled_w < 64 or scaled_h < 64:
            return 0.0

        # Apply CLAHE to map patch too (match preprocessing of frame)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_map_eq = clahe.apply(gray_map)

        # Transform frame grayscale to tile coordinate system
        resized = cv2.resize(frame_enhanced, (scaled_w, scaled_h))
        M = cv2.getRotationMatrix2D((scaled_w // 2, scaled_h // 2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (scaled_w, scaled_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Clip to valid map region
        x1, y1 = max(0, x), max(0, y)
        x2 = min(gray_map.shape[1], x + scaled_w)
        y2 = min(gray_map.shape[0], y + scaled_h)
        pw, ph = x2 - x1, y2 - y1
        if pw < 64 or ph < 64:
            return 0.0

        map_patch = gray_map_eq[y1:y2, x1:x2]
        frame_patch = rotated[y1 - y:y2 - y, x1 - x:x2 - x]

        if map_patch.shape != frame_patch.shape or map_patch.size == 0:
            return 0.0

        # Global mask: exclude rotation border (black) pixels
        valid_mask = frame_patch > 5

        # 3×3 block-wise NCC → trimmed mean (drop worst block)
        bh, bw = ph // 3, pw // 3
        block_scores = []
        for br in range(3):
            for bc in range(3):
                ry1, ry2 = br * bh, (br + 1) * bh
                rx1, rx2 = bc * bw, (bc + 1) * bw
                m_blk = map_patch[ry1:ry2, rx1:rx2]
                f_blk = frame_patch[ry1:ry2, rx1:rx2]
                v_blk = valid_mask[ry1:ry2, rx1:rx2]
                s = self._ncc_of_pair(m_blk, f_blk, v_blk)
                block_scores.append(s)

        if len(block_scores) < 3:
            return 0.0
        block_scores.sort()
        # Trimmed mean: drop worst 2 blocks (likely border artifacts)
        trimmed = block_scores[2:]
        return float(np.mean(trimmed)) if trimmed else 0.0

    # ------------------------------------------------------------------
    #  Line-only DT verify (long lines = roads / canals)
    # ------------------------------------------------------------------
    def _compute_line_verify(self, tile_data, line_mask, match, frame_h, frame_w):
        """DT verify restricted to long-line pixels only (more discriminative)."""
        if line_mask is None:
            return float('inf'), 0.0

        dt = tile_data["dt"]
        scale = float(match["scale"])
        angle = float(match["angle"])
        x, y = int(match["x"]), int(match["y"])

        scaled_w, scaled_h = int(frame_w * scale), int(frame_h * scale)
        if scaled_w < 32 or scaled_h < 32:
            return float('inf'), 0.0

        resized = cv2.resize(line_mask, (scaled_w, scaled_h),
                             interpolation=cv2.INTER_NEAREST)
        M = cv2.getRotationMatrix2D((scaled_w // 2, scaled_h // 2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (scaled_w, scaled_h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        edge_y, edge_x = np.where(rotated > 0)
        if edge_x.size < 50:
            return float('inf'), 0.0

        map_y = edge_y + y
        map_x = edge_x + x
        valid = ((map_x >= 0) & (map_x < dt.shape[1]) &
                 (map_y >= 0) & (map_y < dt.shape[0]))
        vi = np.nonzero(valid)[0]
        if vi.size < 50:
            return float('inf'), 0.0

        dists = dt[map_y[vi], map_x[vi]].astype(np.float32)
        sorted_dists = np.sort(dists)
        best_dists = sorted_dists[:min(len(sorted_dists), 2000)]

        line_median = float(np.median(best_dists))
        core_ratio = float(np.count_nonzero(dists < 2.0) / dists.size)
        return line_median, core_ratio

    # ------------------------------------------------------------------
    #  Multi-patch: anchor + local refine
    # ------------------------------------------------------------------
    def _multi_patch_vote(self, enhanced, edges, top_cands, fh, fw):
        """Anchor + local-refine multi-patch consistency.

        Step A: Use the best coarse candidate as the **anchor** hypothesis
                (tile_idx, x, y, scale, angle).
        Step B: For 4 corner sub-patches, search ONLY within a small window
                around the anchor-predicted position on the map.
        Step C: Measure std of the implied full-frame origin across all
                patches.  If tight -> consistent -> True.
        """
        if len(top_cands) < 1:
            return True

        # Anchor is the best candidate (which came from center patch in coarse search)
        # top_cands entries are already adjusted to represent FULL FRAME top-left
        anchor_match = top_cands[0]["cand"]
        
        tile_idx = anchor_match["tile_idx"]
        tile_data = self.get_tile_data(tile_idx)
        inv_dt = tile_data["inv_dt"]
        tile_info = tile_data["info"]
        
        # Tile origin in global pixel space
        tile_ox = tile_info["x"]
        tile_oy = tile_info["y"]

        ref_scale = float(anchor_match["scale"])
        ref_angle = float(anchor_match["angle"])
        # Anchor x,y are full-frame top-left in tile coords
        anchor_tx = int(anchor_match["x"])
        anchor_ty = int(anchor_match["y"])

        # Parameters for local search
        R = MULTI_PATCH_LOCAL_R  # spatial radius (px)
        # Angle/Scale search ranges
        # User requested: +/- 5 deg, +/- 5% scale
        # We'll do a small grid search
        D_ANG = 5.0
        D_SCL = 0.05
        
        # Patch definitions (4 corners) + Center (already matched roughly, but we can re-verify or just use anchor)
        # Let's verify 4 corners + center. 
        # Center is redundant if we trust anchor, but good for consistency stats.
        m = 0.15 # margin for corners
        psz = 0.30 # patch size (smaller for corners to be distinct?)
        # User said: "Multipatch accept: pos_std <= 80 px"
        # User said: "4 patch'i sadece anchor çevresinde ara"
        # Let's use 5 patches: Center + 4 corners
        
        patches_defs = [
            (0.5, 0.5, "Center"),
            (m, m, "TL"),
            (m, 1.0-m, "BL"),
            (1.0-m, m, "TR"),
            (1.0-m, 1.0-m, "BR")
        ]
        
        patch_h = int(fh * psz)
        patch_w = int(fw * psz)
        
        # Implied global origin of full frame from each patch
        positions = [] 

        # We need to pre-compute rotated templates for the search range?
        # Doing full exhaustive search for 5 patches * Angle * Scale might be slow.
        # But we only search +/- 5 deg (maybe 3 steps: -5, 0, +5) and +/- 5% scale (0.95, 1.0, 1.05)
        # That's 3*3 = 9 variations per patch. fast enough.
        
        search_angles = [ref_angle - D_ANG, ref_angle, ref_angle + D_ANG]
        search_scales = [ref_scale * (1.0 - D_SCL), ref_scale, ref_scale * (1.0 + D_SCL)]

        for cx_frac, cy_frac, name in patches_defs:
            # 1. Extract patch from original frame edges
            cy = int(fh * cy_frac)
            cx = int(fw * cx_frac)
            y1 = max(0, cy - patch_h // 2)
            y2 = min(fh, cy + patch_h // 2)
            x1 = max(0, cx - patch_w // 2)
            x2 = min(fw, cx + patch_w // 2)
            
            if y2 - y1 < 32 or x2 - x1 < 32:
                continue

            sub_edges = edges[y1:y2, x1:x2]
            
            # Where does anchor say this patch should be (center of patch)?
            # We need the vector from Full-Frame-TL to Patch-TL (x1, y1)
            # rotated by ref_angle and scaled by ref_scale
            
            # Vector in frame coords:
            vec_x = x1
            vec_y = y1
            
            # Rotate vector
            # (Note: standard rotation matrix for image coord frame)
            # x' = x cos - y sin
            # y' = x sin + y cos
            # But wait, OpenCV warpAffine uses a center of rotation.
            # Here we just want the offset.
            # Easier: Just map (x1, y1) using the anchor transform.
            
            # We will search for the patch such that it implies a Frame-TL consistent with Anchor.
            # Local search logic:
            # 1. Prediction:
            #    Frame-TL = (anchor_tx, anchor_ty)
            #    Patch-TL-in-Tile = Frame-TL + RotateScale(x1, y1)
            
            rad = np.radians(ref_angle)
            c, s = np.cos(rad), np.sin(rad)
            
            # center of rotation for the frame was (scaled_w/2, scaled_h/2)...
            # Actually, let's look at how coarse match defined (x,y).
            # (x,y) is the top-left of the BOUNDING BOX of the rotated frame in the tile?
            # No, in coarse_search/matchTemplate:
            # result (x,y) is top-left of the template image in the tile.
            # The template was created by: resize frame -> warpAffine (center rotation) -> template
            # So (x,y) corresponds to the top-left of the black-padded rotated image.
            
            # This is complex to invert exactly. 
            # approximate: The patch is a small crop.
            # We generate candidates for the patch by varying angle/scale.
            # For each candidate, we look for it in window +/- R around expected center.
            
            # Expected center of patch in Tile:
            # Frame Center in Tile roughly: anchor_tx + scaled_w/2, anchor_ty + scaled_h/2
            # Patch Center relative to Frame Center: (cx - w/2, cy - h/2)
            # Rotate/Scale that and add to Frame Center.
            
            scaled_fw = fw * ref_scale
            scaled_fh = fh * ref_scale
            
            frame_center_x = anchor_tx + scaled_fw / 2
            frame_center_y = anchor_ty + scaled_fh / 2
            
            # Patch offset from frame center
            dx = (cx - fw/2) * ref_scale
            dy = (cy - fh/2) * ref_scale
            
            # Rotate offset
            rdx = dx * c - dy * s
            rdy = dx * s + dy * c
            
            expected_patch_cx = frame_center_x + rdx
            expected_patch_cy = frame_center_y + rdy
            
            # Now we search for the patch in ROI around (expected_patch_cx, expected_patch_cy)
            
            best_val = -1.0
            best_loc = None
            best_s = ref_scale
            best_a = ref_angle
            
            # Grid search angle/scale
            for s_cand in search_scales:
                for a_cand in search_angles:
                    # Prepare template
                    tsw = int((x2 - x1) * s_cand)
                    tsh = int((y2 - y1) * s_cand)
                    if tsw < 8 or tsh < 8: continue
                    
                    resized_patch = cv2.resize(sub_edges, (tsw, tsh), interpolation=cv2.INTER_NEAREST)
                    M = cv2.getRotationMatrix2D((tsw//2, tsh//2), a_cand, 1.0)
                    rotated_patch = cv2.warpAffine(resized_patch, M, (tsw, tsh), 
                                                   flags=cv2.INTER_NEAREST, 
                                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    
                    # ROI in inv_dt
                    # We expect the CENTER of this patch at (expected_patch_cx, expected_patch_cy)
                    # So Top-Left of this template should be at expected_center - (tsw/2, tsh/2)
                    
                    exp_tl_x = int(expected_patch_cx - tsw/2)
                    exp_tl_y = int(expected_patch_cy - tsh/2)
                    
                    roi_x1 = max(0, exp_tl_x - R)
                    roi_y1 = max(0, exp_tl_y - R)
                    roi_x2 = min(inv_dt.shape[1], exp_tl_x + tsw + R)
                    roi_y2 = min(inv_dt.shape[0], exp_tl_y + tsh + R)
                    
                    if roi_x2 <= roi_x1 + tsw or roi_y2 <= roi_y1 + tsh:
                        continue
                        
                    roi_dt = inv_dt[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    # Downsample for speed?
                    # Since window is small (R=256), maybe full res or 0.5x is fine.
                    # Let's use 0.5x for speed/robustness
                    ds = 0.5
                    tmpl_small = cv2.resize(rotated_patch, (0,0), fx=ds, fy=ds, interpolation=cv2.INTER_NEAREST)
                    roi_small = cv2.resize(roi_dt, (0,0), fx=ds, fy=ds, interpolation=cv2.INTER_LINEAR)
                    
                    if tmpl_small.shape[0] > roi_small.shape[0] or tmpl_small.shape[1] > roi_small.shape[1]:
                        continue
                        
                    res = cv2.matchTemplate(roi_small, tmpl_small, cv2.TM_CCORR_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    
                    if max_val > best_val:
                        best_val = max_val
                        # max_loc is top-left in roi_small
                        # Convert to full res
                        hit_x = roi_x1 + int(max_loc[0] / ds)
                        hit_y = roi_y1 + int(max_loc[1] / ds)
                        best_loc = (hit_x, hit_y)
                        best_s = s_cand
                        best_a = a_cand

            if best_val > 0.05 and best_loc is not None:
                # We found the patch. Now, what does this imply for GLOBAL Frame Position?
                # Revert logic: 
                # Patch-TL-Found = best_loc
                # Patch-Center-Found = best_loc + (tsw/2, tsh/2) approximately
                # Frame-Center-Implied = Patch-Center-Found - RotatedOffset
                
                # Re-calc offset with best_s, best_a
                tsw = int((x2 - x1) * best_s)
                tsh = int((y2 - y1) * best_s)
                patch_center_x = best_loc[0] + tsw/2
                patch_center_y = best_loc[1] + tsh/2
                
                rad = np.radians(best_a)
                c, s = np.cos(rad), np.sin(rad)
                dx = (cx - fw/2) * best_s
                dy = (cy - fh/2) * best_s
                rdx = dx * c - dy * s
                rdy = dx * s + dy * c
                
                implied_frame_cx = patch_center_x - rdx
                implied_frame_cy = patch_center_y - rdy
                
                # Convert to Global Pixel Coords
                # Tile Top-Left Global = (tile_ox, tile_oy)
                global_cx = tile_ox + implied_frame_cx
                global_cy = tile_oy + implied_frame_cy
                
                positions.append((global_cx, global_cy, best_val))

        if len(positions) < 3:
            return True, 0.0  # Not enough patches to verify, fallback to coarse

        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])

        std_x = np.std(xs)
        std_y = np.std(ys)
        pos_std = np.sqrt(std_x**2 + std_y**2)

        min_ncc = min(p[2] for p in positions)

        print(f"[MULTIPATCH] n={len(positions)} std_x={std_x:.1f} std_y={std_y:.1f} "
              f"pos_std={pos_std:.1f} thr={MULTI_PATCH_POS_THR} min_ncc={min_ncc:.3f}")

        return pos_std < MULTI_PATCH_POS_THR, pos_std

    def verify_match(self, match, frame_edges, frame_enhanced=None):
        """
        Verify the coarse match using full resolution DT.
        Measure distances from transformed frame edges to nearest map edge (DT).
        frame_enhanced: CLAHE-enhanced grayscale (used for orientation Sobel).
        Returns:
        median_dist, inlier_ratio, limit_count, grid_coverage, core_count, core_ratio, oriented_core_ratio
        """
        if not match:
            return float('inf'), 0.0, 0, 0.0, 0, 0.0, 0.0

        tile_data = self.get_tile_data(match["tile_idx"])
        dt = tile_data["dt"]

        h, w = frame_edges.shape
        scale = float(match["scale"])
        angle = float(match["angle"])
        x, y = int(match["x"]), int(match["y"])  # top-left in tile

        scaled_w, scaled_h = int(w * scale), int(h * scale)
        if scaled_w < 8 or scaled_h < 8:
            return float('inf'), 0.0, 0, 0.0, 0, 0.0, 0.0

        # --- edge-safe resize/rotate ---
        resized = cv2.resize(frame_edges, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        M = cv2.getRotationMatrix2D((scaled_w // 2, scaled_h // 2), angle, 1.0)
        rotated = cv2.warpAffine(
            resized, M, (scaled_w, scaled_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        rotated = (rotated > 0).astype(np.uint8) * 255

        edge_y, edge_x = np.where(rotated > 0)
        if edge_x.size < 200:
            return float('inf'), 0.0, 0, 0.0, 0, 0.0, 0.0

        map_y = edge_y + y
        map_x = edge_x + x

        valid = (map_x >= 0) & (map_x < dt.shape[1]) & (map_y >= 0) & (map_y < dt.shape[0])

        # --- Safe Indexing & Vectorized Coverage ---
        valid_idx = np.nonzero(valid)[0]
        map_xv = map_x[valid_idx]
        map_yv = map_y[valid_idx]

        if map_xv.size < 1000:
            return float('inf'), 0.0, 0, 0.0, 0, 0.0, 0.0

        dists = dt[map_yv, map_xv].astype(np.float32)

        # One-time DT sanity check
        if not hasattr(self, '_dt_sanity_done'):
            print(f"[DT SANITY] dt.dtype={dt.dtype} min={float(dt.min()):.1f} max={float(dt.max()):.1f} | "
                  f"dists min={float(dists.min()):.1f} max={float(dists.max()):.1f} med={float(np.median(dists)):.1f}")
            self._dt_sanity_done = True

        # Sort and take top 2000
        sorted_idx = np.argsort(dists)
        limit = min(dists.size, 2000)
        best_idx = sorted_idx[:limit]
        best_dists = dists[best_idx]

        # --- Primary metric: MEDIAN (more robust than mean) ---
        median_dist = float(np.median(best_dists))

        # Soft inliers (tightened: 4.0 instead of 5.0)
        soft_mask = dists < 4.0
        inlier_count = int(np.count_nonzero(soft_mask))
        inlier_ratio = float(inlier_count / dists.size)

        # Core inliers (tightened: 2.0 instead of 3.0)
        core_mask = dists < 2.0
        core_count = int(np.count_nonzero(core_mask))
        core_ratio = float(core_count / dists.size)

        # --- Vectorized Coverage on BEST SUBSET ---
        best_subset_idx = valid_idx[best_idx]
        best_edge_y = edge_y[best_subset_idx]
        best_edge_x = edge_x[best_subset_idx]

        grid_h, grid_w = 4, 4
        cell_h = max(1, scaled_h // grid_h)
        cell_w = max(1, scaled_w // grid_w)

        r_cell = (best_edge_y // cell_h).astype(int)
        c_cell = (best_edge_x // cell_w).astype(int)

        in_grid = (r_cell >= 0) & (r_cell < grid_h) & (c_cell >= 0) & (c_cell < grid_w)
        r_cell = r_cell[in_grid]
        c_cell = c_cell[in_grid]

        cell_ids = r_cell * grid_w + c_cell
        unique_cells = np.unique(cell_ids)
        grid_coverage = float(unique_cells.size / (grid_h * grid_w))

        # --- Orientation core ratio (Sobel on grayscale, not binary) ---
        oriented_core_ratio = self._compute_oriented_core_ratio(
            tile_data, frame_edges, match, frame_enhanced=frame_enhanced
        )

        return median_dist, inlier_ratio, limit, grid_coverage, core_count, core_ratio, oriented_core_ratio

    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Could not open video {VIDEO_PATH}")
            return

        cap.set(cv2.CAP_PROP_POS_MSEC, SKIP_SECONDS * 1000)
        
        gt_path = "gt.csv"
        gt_file = open(gt_path, "w")
        gt_file.write("frame_idx,timestamp,lat,lon,rotation,scale,score,score_final,gray_ncc,median_dist,inlier_count,coverage,tile_id,core_count,core_ratio,oriented_core_ratio,uniqueness,line_med,line_cr\n")

        all_path = "all_matches.csv"
        all_file = open(all_path, "w")
        all_file.write("frame_idx,timestamp,lat,lon,rotation,scale,score,score_final,gray_ncc,"
                        "median_dist,inlier_ratio,inlier_count,coverage,tile_id,"
                        "core_count,core_ratio,oriented_core_ratio,uniqueness,"
                        "line_med,line_cr,multipatch_std,is_gt,reject_reason,"
                        "state,edge_before,edge_after,streak_len,dt_search_time\n")
            
        print("Starting GT processing... Press 'q' to quit.")
        
        frame_idx = 0
        history = []          # accepted matches (for temporal lock)
        pending_streak = []   # candidate streak before GT is written
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if self.srt_entries:
                e = _srt_lookup(self.srt_entries, timestamp)
                if e is not None:
                    _t, lat, lon, alt, yaw = e
                    px = self.geo_to_pixel(lon, lat)
                    if px is not None:
                        self._prior_px = px
                    else:
                        self._prior_px = None
                else:
                    self._prior_px = None

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_idx += 1
            
            if frame.shape[1] > 2000:
                frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            t0 = time.time()
            enhanced, edges, frame_lines, edge_before, edge_after = self.preprocess_frame(frame)
            
            # --- Edge accumulation: OR last N frames ---
            self._edge_ring.append(edges.copy())
            if len(self._edge_ring) > EDGE_ACCUM_N:
                self._edge_ring.pop(0)
            if len(self._edge_ring) >= 3:
                accum_edges = self._edge_ring[0].copy()
                for _ae in self._edge_ring[1:]:
                    accum_edges = cv2.bitwise_or(accum_edges, _ae)
                # Slight dilate to link nearby edges
                accum_edges = cv2.dilate(accum_edges, np.ones((3, 3), np.uint8), iterations=1)
                # Re-thin
                accum_edges = cv2.ximgproc.thinning(accum_edges) if hasattr(cv2, 'ximgproc') else accum_edges
                edges_for_coarse = accum_edges
            else:
                edges_for_coarse = edges
            
            # Coarse Match: Top-K (using CENTER PATCH as Anchor)
            # user req: "Sadece center patch ile coarse->verify yap"
            center_edges, off_x, off_y = self._get_center_patch(edges_for_coarse, crop_factor=MULTI_PATCH_SIZE)
            
            # DEBUG: Check center patch content
            _nz = cv2.countNonZero(center_edges)
            print(f"[PATCH DEBUG] size={center_edges.shape} nz={_nz} off={off_x},{off_y}")

            # Extract center gray patch (same crop as edges)
            center_gray = enhanced[off_y:off_y+center_edges.shape[0], off_x:off_x+center_edges.shape[1]].copy()
            candidates_partial = self.coarse_search(center_edges, top_k=5, prior_global_px=self._prior_px, prior_roi_px=self.prior_roi_px, frame_gray=center_gray)
            
            # Convert partial-patch coords to full-frame coords
            candidates = []
            for c in candidates_partial:
                # The 'x' and 'y' in c are top-left of the template in the tile.
                # The template corresponds to the rotated *center patch*.
                # We want candidate to represent the rotated *full frame*.
                
                # Full frame top-left relative to Center Patch top-left?
                # It's negative offset.
                # But we have rotation/scale.
                
                # Logic:
                # coarse_search returns (x,y) = Top-Left of the *Patch's bounding box* in the tile.
                # Actually, matchTemplate returns top-left of the template.
                # The template is the rotated patch (black padded to bounding box).
                
                # We want `match["x"]` to be the Top-Left of the *Full Frame's bounding box* in the tile?
                # Let's check how `verify_match` and `_multi_patch_vote` usage works.
                
                # verify_match usage:
                #   scale, angle, x, y = match[...]
                #   scaled_w, scaled_h = full_frame_w * scale...
                #   M = getRotationMatrix2D(center, angle)
                #   warpAffine(..., M, ...)
                #   map_x = edge_x + x
                
                # So `match["x"], match["y"]` is the offset to add to the *rotated full frame* to place it on the map.
                # i.e., it is the top-left coordinate where the rotated full-frame image should be pasted.
                
                # We have `c["x"], c["y"]`: top-left where the *rotated center patch* is pasted.
                
                # We need to calculate the vector from (FullFrameTL) to (CenterPatchTL) in the *rotated/scaled* space.
                # Frame TL = (0,0)
                # Patch TL = (off_x, off_y)
                
                # Rotated/Scaled Frame TL is (0,0) concept in the output image space?
                # No, warpAffine with borderConstant=0 pads it.
                # `coarse_search` creates templates by:
                #   resized = cv2.resize(img, ...)
                #   rotated = cv2.warpAffine(resized, M, ...)
                # The rotation is around the center of the resized image.
                
                # This is tricky because the "top-left" of the resulting image depends on the rotation padding.
                # But `coarse_search` does:
                #   M = cv2.getRotationMatrix2D((scaled_w//2, scaled_h//2), angle, 1.0)
                #   rotated = cv2.warpAffine(..., (scaled_w, scaled_h))
                # So the result image has the SAME size as the bounding box of the non-rotated scaled image?
                # Wait, (scaled_w, scaled_h) passed to warpAffine is the size of the destination image.
                # In coarse_search: 
                #   scaled_w, scaled_h = int(w*scale), int(h*scale)
                #   rotated_gpu = cv2.warpAffine(..., (scaled_w, scaled_h))
                # So the destination image size is exactly the scaled frame size. 
                # IT IS NOT EXPANDED to fit the corners. The corners are clipped if rotation != 0?
                # NO. `cv2.warpAffine` maintains the canvas size if you pass (scaled_w, scaled_h).
                # If you rotate a rectangle by 45 degrees inside its own original bounds, you CLIP the corners.
                
                # FAILURE IN LOGIC DETECTED (in original code too?):
                # If rotation is small (+/- 15 deg), clipping is minimal.
                # But `verify_match` also does:
                #   M = cv2.getRotationMatrix2D((scaled_w // 2, scaled_h // 2), angle, 1.0)
                #   rotated = cv2.warpAffine(..., (scaled_w, scaled_h))
                # This definitively clips corners for large rotations.
                # For +/- 15 deg, it's probably okay-ish?
                
                # Assumption: Code assumes small scaling/rotation where clipping is negligible or acceptable.
                
                # So:
                # The coordinate system is "Top-Left of the Canvas".
                # The Canvas is size (scaled_w, scaled_h).
                # Center of rotation is (scaled_w/2, scaled_h/2).
                
                # We have a Match for the Center Patch.
                # Center Patch Canvas Size: (scaled_pw, scaled_ph)
                # Match["x"], Match["y"] = Top-Left of this canvas in Map.
                
                # We want Match for Full Frame.
                # Full Frame Canvas Size: (scaled_fw, scaled_fh)
                # We want Top-Left of THIS canvas.
                
                # The geometric relationship between the "Canvas Top-Lefts" depends on the relative position of the centers?
                # Center of CenterPatch Canvas should align with Center of FullFrame Canvas in the map?
                # IF the Center Patch is exactly centered in the Frame.
                # Yes, _get_center_patch clips symmetrically.
                
                # So:
                # Center of Patch Match = (c["x"] + scaled_pw/2, c["y"] + scaled_ph/2)
                # Center of Frame Match = (cand_x + scaled_fw/2, cand_y + scaled_fh/2)
                # These must be equal.
                
                # cand_x = c["x"] + scaled_pw/2 - scaled_fw/2
                # cand_y = c["y"] + scaled_ph/2 - scaled_fh/2
                
                scale = c["scale"]
                angle = c["angle"]

                sfw = int(edges_for_coarse.shape[1] * scale)
                sfh = int(edges_for_coarse.shape[0] * scale)

                rad = np.radians(angle)
                cosa = np.cos(rad)
                sina = np.sin(rad)

                dx = off_x * scale
                dy = off_y * scale

                rot_dx = dx * cosa - dy * sina
                rot_dy = dx * sina + dy * cosa

                full_x = int(c["x"] - rot_dx)
                full_y = int(c["y"] - rot_dy)

                c_full = c.copy()
                c_full["x"] = full_x
                c_full["y"] = full_y
                c_full["w"] = sfw
                c_full["h"] = sfh

            candidates.append(c_full)
            
            # --- DEBUG: Verify the Anchor (Full-frame) Candidate independently ---
            bad_anchor = False
            anchor_med = None
            if candidates:
                _a = candidates[0]  # already expanded to full-frame w/h
                anchor_med, _, _, _, _, _, _ = self.verify_match(_a, edges_for_coarse, frame_enhanced=None)
                if anchor_med is not None and anchor_med > 25.0:
                    bad_anchor = True
                    print(f"[ANCHOR DEBUG] BAD ANCHOR! Score={_a['score']:.3f} Med={anchor_med:.1f}px")
            # Verify Top-K (pass enhanced for Sobel orientation)
            verified_candidates = []
            fh, fw = edges.shape[:2]

            # Pre-compute long-line mask once per frame (top-N longest, >= LINE_MIN_LENGTH)
            long_line_mask = None
            if frame_lines is not None and len(frame_lines) > 0:
                # Compute lengths, filter, sort, take top-N
                _line_data = []
                for _ln in frame_lines:
                    _lx1, _ly1, _lx2, _ly2 = _ln[0]
                    _ll = np.sqrt((_lx2 - _lx1) ** 2 + (_ly2 - _ly1) ** 2)
                    if _ll >= LINE_MIN_LENGTH:
                        _line_data.append((_ll, int(_lx1), int(_ly1), int(_lx2), int(_ly2)))
                _line_data.sort(key=lambda x: x[0], reverse=True)
                _line_data = _line_data[:LINE_TOP_N]  # keep only longest
                if len(_line_data) >= 2:
                    long_line_mask = np.zeros((fh, fw), dtype=np.uint8)
                    for _, _lx1, _ly1, _lx2, _ly2 in _line_data:
                        cv2.line(long_line_mask, (_lx1, _ly1), (_lx2, _ly2), 255, 2)

            for cand in candidates:
                # Refine position + angle locally before full verify
                cand = self._refine_candidate(cand, edges)

                # Returns: median, inlier_ratio, limit, grid_coverage, core_count, core_ratio, oriented_core_ratio
                med, ratio, limit_count, cov, c_count, c_ratio, o_ratio = self.verify_match(cand, edges, frame_enhanced=enhanced)

                # Loose pre-filter
                if med > 15.0:
                    continue
                
                # --- Gray NCC secondary score ---
                tile_data_c = self.get_tile_data(cand["tile_idx"])
                gray_ncc = self._compute_gray_ncc(tile_data_c, enhanced, cand)

                # --- Line-only verify ---
                line_med, line_cr = self._compute_line_verify(
                    tile_data_c, long_line_mask, cand, fh, fw)

                # --- Combined score (edge coarse + gray NCC) ---
                score_final = (EDGE_SCORE_WEIGHT * cand["score"]
                               + GRAY_NCC_WEIGHT * max(0.0, gray_ncc))
                
                verified_candidates.append({
                    "cand": cand,
                    "quality": score_final,
                    "metrics": (med, ratio, limit_count, cov, c_count, c_ratio, o_ratio),
                    "gray_ncc": gray_ncc,
                    "line_med": line_med,
                    "line_cr": line_cr,
                    "score_final": score_final,
                })

            # Sort by DT alignment quality: core_ratio (desc) is the most
            # discriminative metric.  Tie-break with score_final.
            verified_candidates.sort(
                key=lambda x: (x["metrics"][5], x["score_final"]),
                reverse=True)

            # =============================================================
            # Multi-patch position consistency: do 5 sub-patches agree
            # on the same global (X,Y) position?
            # =============================================================
            multi_patch_ok = True  # default pass if disabled
            multipatch_pos_std = 0.0
            if MULTI_PATCH_ENABLED and len(verified_candidates) >= 1:
                try:
                    multi_patch_ok, multipatch_pos_std = self._multi_patch_vote(
                        enhanced, edges_for_coarse, verified_candidates, fh, fw)
                except Exception as e:
                    print(f"[MULTIPATCH ERROR] {e}")
                    import traceback
                    traceback.print_exc()
                    multi_patch_ok = False
                    multipatch_pos_std = 999.0
            
            best_match = None
            best_metrics = None
            uniqueness_reason = "OK"
            uniqueness_score = 2.0
            sf_ratio = None
            oriented_core_ratio = 0.0
            
            if len(verified_candidates) > 0:
                best = verified_candidates[0]
                best_match = best["cand"]
                best_metrics = best["metrics"]
                oriented_core_ratio = best_metrics[6]
                
                # ============================================================
                # Strict uniqueness gate (best vs second-best) + multi-patch
                # ============================================================
                if not multi_patch_ok:
                    uniqueness_reason = "AMBIG_MULTIPATCH"
                elif len(verified_candidates) >= 2:
                    second = verified_candidates[1]

                    # --- Skip uniqueness if top-2 are the same global position (tile overlap) ---
                    b_tile = self.tiles[best["cand"]["tile_idx"]]
                    s_tile = self.tiles[second["cand"]["tile_idx"]]
                    b_gx = b_tile["x"] + best["cand"]["x"]
                    b_gy = b_tile["y"] + best["cand"]["y"]
                    s_gx = s_tile["x"] + second["cand"]["x"]
                    s_gy = s_tile["y"] + second["cand"]["y"]
                    same_global_pos = (abs(b_gx - s_gx) < 100 and abs(b_gy - s_gy) < 100)

                    if same_global_pos:
                        # Same physical location on overlapping tiles - not ambiguous
                        uniqueness_score = 2.0
                    else:
                        # Use score_final (edge + gray NCC) for uniqueness
                        sf1 = best["score_final"]
                        sf2 = second["score_final"]
                        uniqueness_score = sf1 / sf2 if sf2 > 0.001 else 999.0
                    
                    m1 = best["metrics"][0]   # median_dist best
                    m2 = second["metrics"][0]  # median_dist 2nd
                    cr1 = best["metrics"][5]   # core_ratio best
                    cr2 = second["metrics"][5]  # core_ratio 2nd

                    # Line metrics for tie-breaking
                    lm1, lcr1 = best["line_med"], best["line_cr"]
                    lm2, lcr2 = second["line_med"], second["line_cr"]
                    _line_valid = (lm1 < float('inf') and lm2 < float('inf'))

                    # Ratio-based median check (handles small medians better)
                    median_ratio = m2 / m1 if m1 > 0.01 else 999.0

                    if not same_global_pos:
                        # 1) score_final ratio (includes gray NCC)
                        if uniqueness_score < UNIQUENESS_SCORE_RATIO:
                            if not (_line_valid and ((lm2 - lm1) > 1.0 or (lcr1 - lcr2) > 0.05)):
                                uniqueness_reason = "AMBIG_SCORE"
                        # 2) Edge median too close
                        if uniqueness_reason == "OK":
                            if (m2 - m1) < UNIQUENESS_MEDIAN_DIFF and median_ratio < UNIQUENESS_MEDIAN_RATIO:
                                if not (_line_valid and ((lm2 - lm1) > 1.0 or (lcr1 - lcr2) > 0.05)):
                                    uniqueness_reason = "AMBIG_MEDIAN"
                        # 3) Core ratio too close
                        if uniqueness_reason == "OK":
                            if (cr1 - cr2) < UNIQUENESS_CORE_DIFF:
                                if not (_line_valid and (lcr1 - lcr2) > 0.05):
                                    uniqueness_reason = "AMBIG_CORE"
                else:
                    uniqueness_score = 2.0  # only 1 candidate
            
            dt_search = time.time() - t0
            
            is_valid_gt = False
            median_dist, inlier_ratio, limit_count, grid_coverage = float('inf'), 0.0, 0, 0.0
            core_count, core_ratio = 0, 0.0

            # Always unpack metrics for logging (even if uniqueness fails)
            if best_match and best_metrics:
                median_dist, inlier_ratio, limit_count, grid_coverage, core_count, core_ratio, oriented_core_ratio = best_metrics

            # ============================================================
            # Temporal pipeline (always runs when we have a best_match)
            # TRACK/LOCK is allowed to be permissive; GT writing is strict.
            # ============================================================
            reject_reason = []
            if best_match and best_metrics:
                # --- relaxed gates for keeping a candidate in temporal buffer ---
                pass_count    = (limit_count >= 15)
                pass_accuracy = (median_dist <= 20.0)
                pass_cov      = (grid_coverage >= 0.35)
                pass_orient   = (oriented_core_ratio >= ORIENTED_CORE_RATIO_MIN)

                if pass_count and pass_accuracy and pass_cov and pass_orient and (not bad_anchor):
                    # --- Lat/Lon for temporal check ---
                    center_x = best_match["x"] + best_match["w"] / 2
                    center_y = best_match["y"] + best_match["h"] / 2
                    lat, lon = self.pixel_to_latlon(
                        self.get_tile_data(best_match["tile_idx"])["info"],
                        center_x, center_y
                    )

                    match_data = {
                        'lat': lat, 'lon': lon,
                        'angle': best_match['angle'],
                        'tile_idx': best_match['tile_idx']
                    }

                    pending_streak.append(match_data)
                    # keep enough history for GT lock window
                    if len(pending_streak) > GT_LOCK_N * 2:
                        pending_streak.pop(0)

                    # ==========================================================
                    # XFEAT ULTRA CLEAN GT POLICY
                    # ==========================================================
                    gt_ok = True

                    # 1) Uniqueness must be unambiguous
                    if uniqueness_reason != "OK":
                        gt_ok = False
                        reject_reason.append(uniqueness_reason)

                    # 2) DT median must be very low
                    if median_dist > 6.0:
                        gt_ok = False
                        reject_reason.append("MED")

                    # 3) Core ratio must be strong
                    if core_ratio < 0.20:
                        gt_ok = False
                        reject_reason.append("CORE")

                    # 4) Multipatch spatial std must be tight
                    if multipatch_pos_std > MULTI_PATCH_POS_THR:
                        gt_ok = False
                        reject_reason.append("MP_STD")

                    # 5) Temporal lock: at least 2 consecutive frames in streak
                    if len(pending_streak) < GT_LOCK_N:
                        gt_ok = False
                        reject_reason.append("TEMP")

                    # 6) Global position jump: check consistency in map coordinates
                    # (tile ID can change between overlapping tiles — that's fine
                    #  as long as the global position is consistent)
                    if self.last_global_pos is not None:
                        tile_info = self.tiles[best_match["tile_idx"]]
                        cur_gx = tile_info["x"] + best_match["x"]
                        cur_gy = tile_info["y"] + best_match["y"]
                        dx = cur_gx - self.last_global_pos[0]
                        dy = cur_gy - self.last_global_pos[1]
                        pos_jump = (dx**2 + dy**2) ** 0.5
                        # Max ~200px jump between consecutive GT frames
                        if pos_jump > 200:
                            gt_ok = False
                            reject_reason.append("POS_JUMP")

                    # 7) Scale drift: max 10%
                    if self.last_scale is not None:
                        drift = abs(best_match["scale"] - self.last_scale) / self.last_scale
                        if drift > 0.10:
                            gt_ok = False
                            reject_reason.append("SCALE_DRIFT")

                    # 8) Oriented core ratio
                    if oriented_core_ratio < 0.10:
                        gt_ok = False
                        reject_reason.append("ORI")

                    # 9) Grid coverage
                    if grid_coverage < 0.35:
                        gt_ok = False
                        reject_reason.append("COV")

                    # 10) Bad anchor
                    if bad_anchor:
                        gt_ok = False
                        reject_reason.append("ANCHOR")

                    # 11) Temporal consistency check (spatial jump between frames)
                    if gt_ok:
                        if not self.check_temporal_consistency(match_data, pending_streak[:-1], required_n=GT_LOCK_N):
                            gt_ok = False
                            reject_reason.append("TEMPORAL_JUMP")

                    if gt_ok:
                        is_valid_gt = True
                        history.append(match_data)
                        if len(history) > 30:
                            history.pop(0)
                else:
                    # failed relaxed gates -> break the temporal buffer
                    pending_streak.clear()
                    reject_reason.append("RELAXED_GATE")

            # --- DEBUG: Save comparison visualization (every frame with candidates) ---
            if len(verified_candidates) > 0:
                self._save_debug_comparison(
                    frame_idx, edges, verified_candidates,
                    uniqueness_reason, multipatch_pos_std,
                    is_valid_gt, reject_reason
                )

            output = frame.copy()

            # --- Write ALL frames to all_matches.csv ---
            _all_lat = ""
            _all_lon = ""
            _all_rotation = ""
            _all_scale = ""
            _all_score = ""
            _all_sf = ""
            _all_gncc = ""
            _all_med = ""
            _all_ir = ""
            _all_lc = ""
            _all_cov = ""
            _all_tile = ""
            _all_cc = ""
            _all_cr = ""
            _all_ocr = ""
            _all_uq = ""
            _all_lm = ""
            _all_lcr = ""
            _all_mpstd = f"{multipatch_pos_std:.2f}"
            _all_gt = "1" if is_valid_gt else "0"
            _all_reason = ",".join(reject_reason) if reject_reason else ("" if is_valid_gt else "NO_MATCH")
            _all_state = self.state
            _all_eb = str(edge_before)
            _all_ea = str(edge_after)
            _all_strk = str(len(pending_streak))
            _all_dt_time = f"{dt_search:.4f}"

            if best_match and best_metrics:
                _all_rotation = f"{best_match['angle']}"
                _all_scale = f"{best_match['scale']:.2f}"
                _all_score = f"{best_match['score']:.4f}"
                _all_med = f"{median_dist:.2f}"
                _all_ir = f"{inlier_ratio:.4f}"
                _all_lc = str(limit_count)
                _all_cov = f"{grid_coverage:.2f}"
                _all_tile = self.tiles[best_match['tile_idx']]['id']
                _all_cc = str(core_count)
                _all_cr = f"{core_ratio:.2f}"
                _all_ocr = f"{oriented_core_ratio:.3f}"
                _all_uq = f"{uniqueness_score:.3f}"

                if verified_candidates:
                    _all_sf = f"{verified_candidates[0]['score_final']:.4f}"
                    _all_gncc = f"{verified_candidates[0]['gray_ncc']:.4f}"
                    _all_lm = f"{verified_candidates[0]['line_med']:.2f}" if verified_candidates[0]['line_med'] < float('inf') else ""
                    _all_lcr = f"{verified_candidates[0]['line_cr']:.3f}"

                # lat/lon — compute if not already available
                try:
                    _center_x = best_match["x"] + best_match["w"] / 2
                    _center_y = best_match["y"] + best_match["h"] / 2
                    _lat_all, _lon_all = self.pixel_to_latlon(
                        self.get_tile_data(best_match["tile_idx"])["info"],
                        _center_x, _center_y
                    )
                    _all_lat = f"{_lat_all:.8f}"
                    _all_lon = f"{_lon_all:.8f}"
                except Exception:
                    pass

            # Escape reject_reason in case it contains commas
            _all_reason_safe = _all_reason.replace(",", ";")
            all_line = (f"{frame_idx},{timestamp:.3f},{_all_lat},{_all_lon},"
                        f"{_all_rotation},{_all_scale},{_all_score},{_all_sf},{_all_gncc},"
                        f"{_all_med},{_all_ir},{_all_lc},{_all_cov},{_all_tile},"
                        f"{_all_cc},{_all_cr},{_all_ocr},{_all_uq},"
                        f"{_all_lm},{_all_lcr},{_all_mpstd},{_all_gt},{_all_reason_safe},"
                        f"{_all_state},{_all_eb},{_all_ea},{_all_strk},{_all_dt_time}\n")
            all_file.write(all_line)
            all_file.flush()
            
            if is_valid_gt:
                self.state = "LOCKED"
                self.lock_counter = LOCK_WINDOW
                self.last_tile_idx = best_match["tile_idx"]
                self.last_scale = best_match["scale"]
                # Track global position for POS_JUMP check
                _gt_tile = self.tiles[best_match["tile_idx"]]
                self.last_global_pos = (
                    _gt_tile["x"] + best_match["x"],
                    _gt_tile["y"] + best_match["y"]
                )

                # --- Visual tracking chain: use last GT as prior for next frame ---
                # Critical when no GPS/SRT available - narrows search to nearby tiles
                self._prior_px = self.last_global_pos
                
                _sf = verified_candidates[0]['score_final']
                _gncc = verified_candidates[0]['gray_ncc']
                _lm = verified_candidates[0]['line_med']
                _lcr = verified_candidates[0]['line_cr']
                line = (f"{frame_idx},{timestamp:.3f},{lat:.8f},{lon:.8f},"
                        f"{best_match['angle']},{best_match['scale']:.2f},"
                        f"{best_match['score']:.4f},{_sf:.4f},"
                        f"{_gncc:.4f},{median_dist:.2f},"
                        f"{limit_count},{grid_coverage:.2f},"
                        f"{self.tiles[best_match['tile_idx']]['id']},"
                        f"{core_count},{core_ratio:.2f},"
                        f"{oriented_core_ratio:.3f},{uniqueness_score:.3f},"
                        f"{_lm:.2f},{_lcr:.3f}\n")
                gt_file.write(line)
                gt_file.flush()
                
                color = (0, 255, 0)
                cv2.putText(output, f"GT: SAVED ({lat:.6f}, {lon:.6f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(output, f"S:{best_match['score']:.2f} U:{uniqueness_score:.2f} Med:{median_dist:.2f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(output, f"Ori:{oriented_core_ratio:.2f} Cov:{grid_coverage:.0%} Streak:{len(pending_streak)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                self.visualize_match(frame, frame_lines, best_match)
                
            else:
                self.lock_counter -= 1
                if self.lock_counter <= 0:
                    self.state = "SEARCHING" 
                
                color = (0, 0, 255)
                reason = ",".join(reject_reason) if reject_reason else "NO_MATCH"

                cv2.putText(output, f"GT: {reason}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if best_match:
                     cv2.putText(output, f"S:{best_match['score']:.2f} U:{uniqueness_score:.2f} Med:{median_dist:.2f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                     cv2.putText(output, f"Ori:{oriented_core_ratio:.2f} Cov:{grid_coverage:.0%} Streak:{len(pending_streak)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                     
                     self.visualize_match(frame, frame_lines, best_match)
            
            # --- Edge mask diagnostic ---
            mask_pct = (1.0 - edge_after / max(edge_before, 1)) * 100
            cv2.putText(output, f"Edges: {edge_before}->{edge_after} (mask -{mask_pct:.0f}%)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(output, f"Time: {dt_search:.3f}s Frame: {frame_idx}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # --- Diagnostic console log (per-frame) ---
            diag_parts = [f"F{frame_idx}"]
            if best_match:
                tile_name = self.tiles[best_match['tile_idx']]['id']
                diag_parts.append(f"tile={tile_name}")
                diag_parts.append(f"sf={verified_candidates[0]['score_final']:.3f}")
                diag_parts.append(f"ncc={verified_candidates[0]['gray_ncc']:.3f}")
                if len(verified_candidates) >= 2:
                    sf2 = verified_candidates[1]['score_final']
                    ncc2 = verified_candidates[1]['gray_ncc']
                    m2_val = verified_candidates[1]['metrics'][0]
                    cr2_val = verified_candidates[1]['metrics'][5]
                    tile2_name = self.tiles[verified_candidates[1]['cand']['tile_idx']]['id']
                    diag_parts.append(f"sf2={sf2:.3f}")
                    diag_parts.append(f"ncc2={ncc2:.3f}")
                    diag_parts.append(f"sf1/sf2={((sf_ratio if sf_ratio is not None else uniqueness_score)):.3f}")
                    diag_parts.append(f"uq={uniqueness_score:.3f}")
                    diag_parts.append(f"med1={best_metrics[0]:.2f}")
                    diag_parts.append(f"med2={m2_val:.2f}")
                    diag_parts.append(f"cr1={best_metrics[5]:.3f}")
                    diag_parts.append(f"cr2={cr2_val:.3f}")
                    diag_parts.append(f"t2={tile2_name}")
                    # Line metrics
                    _lm1d = verified_candidates[0]['line_med']
                    _lm2d = verified_candidates[1]['line_med']
                    _lcr1d = verified_candidates[0]['line_cr']
                    _lcr2d = verified_candidates[1]['line_cr']
                    if _lm1d < float('inf'):
                        diag_parts.append(f"lm1={_lm1d:.2f}")
                    if _lm2d < float('inf'):
                        diag_parts.append(f"lm2={_lm2d:.2f}")
                    diag_parts.append(f"lcr1={_lcr1d:.3f}")
                    diag_parts.append(f"lcr2={_lcr2d:.3f}")
                else:
                    diag_parts.append(f"med={best_metrics[0]:.2f}")
                    diag_parts.append(f"cr={best_metrics[5]:.3f}")
                diag_parts.append(f"ori={oriented_core_ratio:.3f}")
                diag_parts.append(f"cov={grid_coverage:.2f}")
            diag_parts.append(f"edg={edge_before}/{edge_after}")
            diag_parts.append(f"strk={len(pending_streak)}")
            diag_parts.append(f"mpstd={multipatch_pos_std:.1f}")
            if is_valid_gt:
                diag_parts.append("GT_SAVED_ULTRA")
            elif reject_reason:
                diag_parts.append(f"REJECT_ULTRA:{','.join(reject_reason)}")
            else:
                diag_parts.append("REJECT")
            print(" | ".join(diag_parts))

            cv2.imshow("Drone Localization (GT Mode)", output)
            if cv2.waitKey(1) == ord('q'):
                break
                
        cap.release()
        gt_file.close()
        all_file.close()
        print(f"[DONE] All matches saved to {all_path}")
        print(f"[DONE] GT matches saved to {gt_path}")
        cv2.destroyAllWindows()

    def match_lines_geometric(self, frame_lines_transformed, map_lines):
        """
        Find matches between transformed frame lines and map lines.
        Returns list of (frame_line_idx, map_line_idx)
        """
        if map_lines is None or len(map_lines) == 0:
            return []
        if frame_lines_transformed is None or len(frame_lines_transformed) == 0:
            return []
            
        # Simple N*M check (optimize with spatial index later if needed)
        # Criteria:
        # 1. Distance between midpoints is small
        # 2. Angle difference is small
        
        matches = []
        
        # map_lines: (N, 1, 4)
        map_lines_flat = map_lines.reshape(-1, 4)
        # Precompute map midpoints and angles
        map_mids = (map_lines_flat[:, :2] + map_lines_flat[:, 2:]) / 2
        map_vecs = map_lines_flat[:, 2:] - map_lines_flat[:, :2]
        map_angles = np.arctan2(map_vecs[:, 1], map_vecs[:, 0])
        
        # Frame lines
        fr_lines_flat = frame_lines_transformed.reshape(-1, 4)
        fr_mids = (fr_lines_flat[:, :2] + fr_lines_flat[:, 2:]) / 2
        fr_vecs = fr_lines_flat[:, 2:] - fr_lines_flat[:, :2]
        fr_angles = np.arctan2(fr_vecs[:, 1], fr_vecs[:, 0])
        
        DIST_THRESH = 20.0 # pixels
        ANGLE_THRESH = np.radians(15) # 15 degrees
        
        for i, (fm, fa) in enumerate(zip(fr_mids, fr_angles)):
            # Distances to all map lines
            dists = np.linalg.norm(map_mids - fm, axis=1)
            
            # Filter by distance
            potential_idxs = np.where(dists < DIST_THRESH)[0]
            
            best_idx = -1
            min_dist = float('inf')
            
            for p_idx in potential_idxs:
                # Check angle
                ma = map_angles[p_idx]
                diff = abs(fa - ma)
                # Handle wrap around
                diff = min(diff, 2*np.pi - diff)
                # Also lines calculate mod pi (direction agnostic?) 
                # LSD lines usually have direction, but road lines don't.
                # Let's check mod pi
                diff_pi = diff % np.pi
                diff_pi = min(diff_pi, np.pi - diff_pi)
                
                if diff_pi < ANGLE_THRESH:
                    if dists[p_idx] < min_dist:
                        min_dist = dists[p_idx]
                        best_idx = p_idx
            
            if best_idx != -1:
                matches.append((i, best_idx))
                
        return matches

    def _save_debug_comparison(self, frame_idx, frame_edges, verified_candidates,
                               uniqueness_reason, multipatch_std, is_valid_gt, reject_reasons):
        """
        Save debug visualization comparing top-2 candidates side-by-side.
        Helps identify false positives and ambiguous matches.
        """
        import os
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)

        if len(verified_candidates) == 0:
            return

        # Get top-2 candidates
        best = verified_candidates[0]
        second = verified_candidates[1] if len(verified_candidates) >= 2 else None

        # Create canvas: [Frame | Tile-1 | Tile-2]
        h_f, w_f = frame_edges.shape
        canvas_h = h_f * 2
        canvas_w = w_f * 3
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # --- Top-left: Frame edges (GREEN) ---
        frame_vis = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2BGR)
        frame_vis[frame_edges > 0] = [0, 255, 0]  # green edges
        canvas[0:h_f, 0:w_f] = frame_vis
        cv2.putText(canvas, "FRAME EDGES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2)

        # --- Top-middle: Best candidate tile edges (CYAN) ---
        tile1_data = self.get_tile_data(best["cand"]["tile_idx"])
        tile1_gray = tile1_data["gray"]

        # Extract ROI from tile matching the frame
        x1, y1 = best["cand"]["x"], best["cand"]["y"]
        w1, h1 = best["cand"]["w"], best["cand"]["h"]

        # Compute edges from tile gray for visualization
        tile1_edges = cv2.Canny(tile1_gray[y1:y1+h1, x1:x1+w1], 80, 200)
        tile1_edges_resized = cv2.resize(tile1_edges, (w_f, h_f))

        tile1_vis = cv2.cvtColor(tile1_edges_resized, cv2.COLOR_GRAY2BGR)
        tile1_vis[tile1_edges_resized > 0] = [255, 255, 0]  # cyan edges
        canvas[0:h_f, w_f:w_f*2] = tile1_vis

        # Metrics text for best
        m1 = best["metrics"]
        score1 = best["score_final"]
        text1 = [
            f"BEST (Tile {best['cand']['tile_idx']})",
            f"Score: {score1:.3f}",
            f"Median: {m1[0]:.1f}px",
            f"Core: {m1[5]:.2f}",
            f"Orient: {m1[6]:.2f}",
            f"Cov: {m1[3]:.2f}"
        ]
        for i, txt in enumerate(text1):
            cv2.putText(canvas, txt, (w_f + 10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # --- Top-right: Second candidate tile edges (MAGENTA) if exists ---
        if second is not None:
            tile2_data = self.get_tile_data(second["cand"]["tile_idx"])
            tile2_gray = tile2_data["gray"]

            x2, y2 = second["cand"]["x"], second["cand"]["y"]
            w2, h2 = second["cand"]["w"], second["cand"]["h"]

            tile2_edges = cv2.Canny(tile2_gray[y2:y2+h2, x2:x2+w2], 80, 200)
            tile2_edges_resized = cv2.resize(tile2_edges, (w_f, h_f))

            tile2_vis = cv2.cvtColor(tile2_edges_resized, cv2.COLOR_GRAY2BGR)
            tile2_vis[tile2_edges_resized > 0] = [255, 0, 255]  # magenta edges
            canvas[0:h_f, w_f*2:w_f*3] = tile2_vis

            # Metrics text for second
            m2 = second["metrics"]
            score2 = second["score_final"]
            text2 = [
                f"2ND (Tile {second['cand']['tile_idx']})",
                f"Score: {score2:.3f}",
                f"Median: {m2[0]:.1f}px",
                f"Core: {m2[5]:.2f}",
                f"Orient: {m2[6]:.2f}",
                f"Cov: {m2[3]:.2f}"
            ]
            for i, txt in enumerate(text2):
                cv2.putText(canvas, txt, (w_f*2 + 10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # --- Bottom: Summary info ---
        summary_y = h_f + 30
        uniqueness_color = (0, 255, 0) if uniqueness_reason == "OK" else (0, 0, 255)
        gt_color = (0, 255, 0) if is_valid_gt else (0, 165, 255)

        summary_lines = [
            f"Frame {frame_idx}  |  Uniqueness: {uniqueness_reason}  |  MultiPatch STD: {multipatch_std:.1f}px",
            f"GT: {'YES' if is_valid_gt else 'NO'}  |  Reject: {', '.join(reject_reasons) if reject_reasons else 'N/A'}",
            f"Candidates: {len(verified_candidates)}"
        ]

        for i, txt in enumerate(summary_lines):
            color = gt_color if i == 1 else uniqueness_color if i == 0 else (255, 255, 255)
            cv2.putText(canvas, txt, (10, summary_y + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save to file
        filename = os.path.join(debug_dir, f"frame_{frame_idx:05d}_debug.jpg")
        cv2.imwrite(filename, canvas)

        # Also save rejected GT frames separately for analysis
        if not is_valid_gt and len(verified_candidates) > 0:
            reject_dir = os.path.join(debug_dir, "rejected_gt")
            os.makedirs(reject_dir, exist_ok=True)
            reject_file = os.path.join(reject_dir, f"frame_{frame_idx:05d}_REJECT.jpg")
            cv2.imwrite(reject_file, canvas)

    def visualize_match(self, frame_img, frame_lines, match):
        if not match: return

        tile_data = self.get_tile_data(match["tile_idx"])
        # Map image (BGR)
        map_img = cv2.cvtColor(tile_data["gray"], cv2.COLOR_GRAY2BGR)
        
        # Frame image (ensure BGR)
        if len(frame_img.shape) == 2:
            frame_vis = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
        else:
            frame_vis = frame_img.copy()
            
        # Composite View
        # We want [Frame | Map]
        # Heights might match or not. Pad Frame to Map height? Map is 2048 usually. Frame 1080.
        # Let's align tops.
        h_f, w_f = frame_vis.shape[:2]
        h_m, w_m = map_img.shape[:2]
        
        # Create canvas
        h_canvas = max(h_f, h_m)
        w_canvas = w_f + w_m
        canvas = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8)
        
        # Place Frame
        canvas[0:h_f, 0:w_f] = frame_vis
        
        # Place Map
        # If we just place map at x=w_f, we use map coords directly + w_f
        canvas[0:h_m, w_f:w_f+w_m] = map_img
        
        # Draw Map Lines (Blue) on Map side
        map_lines = tile_data.get("lines")
        if map_lines is not None:
            for line in map_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(canvas, (int(x1)+w_f, int(y1)), (int(x2)+w_f, int(y2)), (255, 0, 0), 1)

        # Draw Frame Lines (Green) on Frame side
        fr_lines_trans = None # For geometric check
        if frame_lines is not None:
            for line in frame_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
            # We need transformed frame lines for logic matching, but we draw connectors from ORIGINAL frame lines.
            # Re-calculate transformed lines for matching purposes
            scale = match["scale"]
            angle = match["angle"]
            x, y = match["x"], match["y"]
            
            pts = []
            for line in frame_lines:
                 x1, y1, x2, y2 = line[0]
                 pts.append([x1, y1])
                 pts.append([x2, y2])
            pts = np.array(pts, dtype=np.float32)
            pts = pts * scale
            center = (w_f * scale / 2, h_f * scale / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            pts_reshaped = pts.reshape(-1, 1, 2)
            trans_pts = cv2.transform(pts_reshaped, M)
            trans_pts = trans_pts.reshape(-1, 2)
            trans_pts[:, 0] += x
            trans_pts[:, 1] += y
            fr_lines_trans = trans_pts.reshape(-1, 1, 4)

        # Draw Connectors (Yellow)
        if fr_lines_trans is not None and map_lines is not None:
            matches = self.match_lines_geometric(fr_lines_trans, map_lines)
            
            # Subsample matches to avoid clutter?
            # matches = matches[::2] 
            
            for (fr_idx, map_idx) in matches:
                # Frame point (Original Frame Coords)
                f_l_orig = frame_lines[fr_idx][0]
                f_mid = ((f_l_orig[0]+f_l_orig[2])/2, (f_l_orig[1]+f_l_orig[3])/2)
                pt_frame = (int(f_mid[0]), int(f_mid[1]))
                
                # Map point (Map Coords + Offset)
                m_l = map_lines[map_idx][0]
                m_mid = ((m_l[0]+m_l[2])/2, (m_l[1]+m_l[3])/2)
                pt_map = (int(m_mid[0]) + w_f, int(m_mid[1]))
                
                cv2.line(canvas, pt_frame, pt_map, (0, 255, 255), 1)
                cv2.circle(canvas, pt_frame, 3, (0, 255, 255), -1)
                cv2.circle(canvas, pt_map, 3, (0, 255, 255), -1)

        # Draw Bounding Box of Frame on Map (Red)
        scale = match["scale"]
        angle = match["angle"]
        x, y = match["x"], match["y"]
        scaled_w, scaled_h = int(w_f * scale), int(h_f * scale)
        rect = ((x + scaled_w/2 + w_f, y + scaled_h/2), (scaled_w, scaled_h), angle)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(canvas, [box], 0, (0, 0, 255), 3)

        # Calculate Footprint Size
        # rotated bounding box size
        # Or just use the scale*frame_size?
        # User says "frame's footprint on map". This is best approximated by the axis-aligned bounding box of the rotated/scaled frame
        # OR simply the area covered.
        # Let's show the scaled dimensions.
        footprint_text = f"Footprint: {scaled_w}x{scaled_h} px (Scale {scale:.2f})"
        cv2.putText(canvas, footprint_text, (w_f + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Resize for Display
        display_h = 900
        scale_disp = display_h / h_canvas
        display_w = int(w_canvas * scale_disp)
        vis_small = cv2.resize(canvas, (display_w, display_h))
        
        cv2.imshow("Side-by-Side Match", vis_small)

    def pixel_to_latlon(self, tile_info, tile_x, tile_y):
        """
        Convert tile-relative pixel coordinates to Global Lat/Lon.
        """
        # 1. Global Pixel
        # tile_info["x"], ["y"] are the top-left of the tile in global pixel coords if map was one big image?
        # Yes, from map_processor: x, y are window offsets.
        global_px_x = tile_info["x"] + tile_x
        global_px_y = tile_info["y"] + tile_y
        
        # 2. Apply GeoTransform
        # transform is [a, b, c, d, e, f]
        # x_geo = a * px + b * py + c
        # y_geo = d * px + e * py + f
        # Standard Affine: x_geo = a*col + b*row + c
        #                  y_geo = d*col + e*row + f
        # Check rasterio docs: src.xy(row, col) -> (x, y)
        # Manually:
        t = self.transform
        # t = [a, b, c, d, e, f]
        # Xgeo = t[0] * global_px_x + t[1] * global_px_y + t[2]
        # Ygeo = t[3] * global_px_x + t[4] * global_px_y + t[5]
        
        # Rasterio transform usually maps (col, row) -> (x, y)
        # col = x, row = y
        # However, checking map_processor output: | 0.00, 0.00, 32.72| ...
        # It seems like a valid affine.
        
        # Let's rely on manual affine multiplication if we stored the list
        a, b, c, d, e, f = t
        geo_x = a * global_px_x + b * global_px_y + c
        geo_y = d * global_px_x + e * global_px_y + f
        
        # 3. CRS Conversion (if needed)
        # User map says EPSG:4326, so geo_x, geo_y are likely Lon, Lat.
        # If not, would need pyproj. 
        # Checking log: "Map CRS: EPSG:4326".
        # So geo_x is Longitude, geo_y is Latitude.
        
        return geo_y, geo_x # Lat, Lon

if __name__ == "__main__":
    localizer = DroneLocalizer(MAP_DIR)
    localizer.run()