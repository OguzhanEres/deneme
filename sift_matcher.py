"""
SIFT-based Drone Geo-Localizer for XFeat GT Dataset Generation.

Architecture:
  Phase 1: Global descriptor retrieval → top-K candidate tiles
  Phase 2: SIFT matching + RANSAC homography → precise alignment
  Phase 3: Multi-metric verification (NCC, SSIM, inlier quality)
  Phase 4: Temporal smoothing → GT export

Usage:
    # Step 1: Process tiles first
    python sift_map_processor.py --map dikilitas_new.tif

    # Step 2: Run matcher
    python sift_matcher.py --video DJIG0022.mov [--map-dir processed_map] [--skip 35]
"""

import cv2
import numpy as np
import os
import json
import time
import argparse
import bisect
import pathlib
import re as _re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# DJI SRT Parser (reused from original)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_dji_srt(srt_path: str):
    p = pathlib.Path(srt_path)
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8", errors="ignore")
    entries = []
    blocks = _re.split(r"\n\s*\n", txt.strip())
    time_re = _re.compile(
        r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d\d\d)")
    gps_re1 = _re.compile(
        r"GPS\s*\(?\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*\)?",
        _re.IGNORECASE)
    gps_re2 = _re.compile(
        r"\bLat\b\s*[:=]\s*([-+]?\d+\.\d+).*?\bLon\b\s*[:=]\s*([-+]?\d+\.\d+)"
        r".*?\bAlt\b\s*[:=]\s*([-+]?\d+\.?\d*)",
        _re.IGNORECASE | _re.DOTALL)
    yaw_re = _re.compile(
        r"\b(?:Yaw|Heading|Hdg)\b\s*[:=]\s*([-+]?\d+\.?\d*)", _re.IGNORECASE)

    def to_sec(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    for b in blocks:
        m = time_re.search(b)
        if not m:
            continue
        t0 = to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
        lat = lon = alt = yaw = None
        g = gps_re1.search(b) or gps_re2.search(b)
        if g:
            try:
                lat, lon, alt = float(g.group(1)), float(g.group(2)), float(g.group(3))
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
    if not entries:
        return None
    ts = [e[0] for e in entries]
    i = bisect.bisect_left(ts, t_sec)
    if i <= 0:
        return entries[0]
    if i >= len(entries):
        return entries[-1]
    before, after = entries[i - 1], entries[i]
    return before if (t_sec - before[0]) <= (after[0] - t_sec) else after


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Paths
    map_dir: str = "processed_map"
    video_path: str = "DJIG0022.mov"
    skip_seconds: float = 35.0

    # Retrieval
    retrieval_top_k: int = 5          # candidate tiles from global descriptor

    # SIFT matching
    sift_max_keypoints: int = 4000    # for frame
    lowe_ratio: float = 0.80         # Lowe's ratio test (relaxed for cross-domain)
    min_inliers: int = 10            # min RANSAC inliers for valid match
    ransac_reproj_thresh: float = 8.0 # RANSAC reprojection threshold (px)

    # Verification thresholds
    min_ncc: float = 0.15            # grayscale NCC minimum (relaxed for cross-domain)
    min_ssim: float = 0.10           # structural similarity minimum
    min_inlier_ratio: float = 0.08   # inliers / total matches
    max_reproj_error: float = 12.0   # mean reprojection error (px)
    min_coverage: float = 0.15       # spatial coverage of inliers (0-1)

    # Temporal
    gt_lock_n: int = 2               # consecutive frames for GT write
    max_jump_m: float = 30.0         # max position jump (metres)
    max_angle_diff: float = 15.0     # max rotation change (degrees)
    max_scale_drift: float = 0.15    # max scale change ratio

    # Frame processing
    edge_density_min: float = 0.004  # skip featureless frames
    roi_px: int = 1100               # prior-based search radius (pixels)

    # Debug / visualization
    save_debug: bool = True
    debug_dir: str = "debug_sift"
    live_view: bool = False           # show cv2.imshow live windows
    live_wait_ms: int = 1             # waitKey delay (1=fast, 0=pause each frame)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Decompose Homography → scale, rotation, translation
# ─────────────────────────────────────────────────────────────────────────────
def decompose_homography(H, frame_h, frame_w):
    """
    Extract (scale, rotation_deg, center_x, center_y) from a homography
    that maps frame pixels to tile pixels.

    Uses the frame center as reference point and decomposes the
    affine part of H.
    """
    # Map frame center through H
    fc = np.array([frame_w / 2.0, frame_h / 2.0, 1.0])
    mapped = H @ fc
    mapped /= mapped[2]
    center_x, center_y = mapped[0], mapped[1]

    # Extract affine part (top-left 2x2 of H, normalized by H[2,2])
    h = H / H[2, 2]
    a, b = h[0, 0], h[0, 1]
    c, d = h[1, 0], h[1, 1]

    # Scale = sqrt(det of 2x2)
    det = a * d - b * c
    scale = np.sqrt(abs(det))

    # Rotation = atan2 of first column
    rotation_rad = np.arctan2(c, a)
    rotation_deg = np.degrees(rotation_rad)

    return scale, rotation_deg, center_x, center_y


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Convert serialized keypoints back to cv2.KeyPoint
# ─────────────────────────────────────────────────────────────────────────────
def array_to_keypoints(kp_array):
    """Convert Nx7 array back to list of cv2.KeyPoint."""
    kps = []
    for row in kp_array:
        kp = cv2.KeyPoint(x=float(row[0]), y=float(row[1]),
                          size=float(row[2]), angle=float(row[3]),
                          response=float(row[4]), octave=int(row[5]),
                          class_id=int(row[6]))
        kps.append(kp)
    return kps


# ─────────────────────────────────────────────────────────────────────────────
# Main Matcher Class
# ─────────────────────────────────────────────────────────────────────────────
class SIFTMatcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tiles = []
        self.transform = None
        self.tile_size = 0
        self.stride = 0

        # SIFT detector for frames (more octave layers for better cross-scale matching)
        self.sift = cv2.SIFT_create(
            nfeatures=cfg.sift_max_keypoints,
            nOctaveLayers=4,        # default=3, more layers = better scale matching
            contrastThreshold=0.03,  # slightly lower to get more features
            edgeThreshold=15,        # slightly higher to keep more features
        )

        # BFMatcher as primary (more reliable than FLANN for cross-domain)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2)

        # FLANN matcher as backup
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Caches
        self.tile_cache = {}          # tile_idx -> tile data dict
        self.global_descs = None      # Nx128 matrix of all tile global descs
        self.global_desc_indices = [] # which tile_idx each row corresponds to

        # SRT GPS prior
        self.srt_entries = []
        self._prior_px = None

        # State
        self.state = "LOST"
        self.last_global_pos = None
        self.last_scale = None
        self.last_angle = None

        # Load
        self._load_metadata()
        self._load_global_descriptors()
        self._load_srt()

    # ── Loading ──────────────────────────────────────────────────────────

    def _load_metadata(self):
        meta_path = os.path.join(self.cfg.map_dir, "metadata.json")
        with open(meta_path) as f:
            data = json.load(f)
        self.tiles = data["tiles"]
        self.transform = data.get("transform")
        self.tile_size = data["tile_size"]
        self.stride = data["stride"]
        print(f"[MAP] Loaded {len(self.tiles)} tiles")

        if self.transform:
            a, b, c, d, e, f = self.transform
            pix_w = abs(a) if abs(a) > 1e-12 else abs(b)
            pix_h = abs(e) if abs(e) > 1e-12 else abs(d)
            print(f"[GEO] pixel size ~ {pix_w:.10f} x {pix_h:.10f} deg/px")

    def _load_global_descriptors(self):
        """Load all tile global descriptors into a single matrix for fast retrieval."""
        descs = []
        indices = []
        for i, tile in enumerate(self.tiles):
            desc_path = os.path.join(self.cfg.map_dir, tile["id"], "global_desc.npy")
            if os.path.exists(desc_path):
                desc = np.load(desc_path)
                if np.linalg.norm(desc) > 1e-6:
                    descs.append(desc)
                    indices.append(i)

        if descs:
            self.global_descs = np.vstack(descs).astype(np.float32)
            self.global_desc_indices = indices
            print(f"[RETRIEVAL] Loaded global descriptors for {len(indices)} tiles")
        else:
            self.global_descs = None
            print("[RETRIEVAL] WARNING: No global descriptors found!")

    def _load_srt(self):
        srt_guess = str(pathlib.Path(self.cfg.video_path).with_suffix(".SRT"))
        if os.path.exists(srt_guess):
            self.srt_entries = _parse_dji_srt(srt_guess)
            if self.srt_entries:
                print(f"[SRT] Loaded {len(self.srt_entries)} GPS samples")

    # ── Geo transforms ───────────────────────────────────────────────────

    def pixel_to_geo(self, px_x, px_y):
        if not self.transform:
            return None
        a, b, c, d, e, f = self.transform
        lon = a * px_x + b * px_y + c
        lat = d * px_x + e * px_y + f
        return lon, lat

    def geo_to_pixel(self, lon, lat):
        if not self.transform:
            return None
        a, b, c, d, e, f = self.transform
        det = a * e - b * d
        if abs(det) < 1e-18:
            return None
        x = (e * (lon - c) - b * (lat - f)) / det
        y = (-d * (lon - c) + a * (lat - f)) / det
        return float(x), float(y)

    def pixel_to_latlon(self, tile_info, tile_x, tile_y):
        global_px_x = tile_info["x"] + tile_x
        global_px_y = tile_info["y"] + tile_y
        a, b, c, d, e, f = self.transform
        geo_x = a * global_px_x + b * global_px_y + c
        geo_y = d * global_px_x + e * global_px_y + f
        return geo_y, geo_x  # lat, lon

    def tiles_in_roi(self, center_px, roi_px):
        cx, cy = center_px
        x0, y0 = cx - roi_px, cy - roi_px
        x1, y1 = cx + roi_px, cy + roi_px
        idxs = []
        for i, t in enumerate(self.tiles):
            tx0, ty0 = t["x"], t["y"]
            tx1 = tx0 + self.tile_size
            ty1 = ty0 + self.tile_size
            if tx1 < x0 or tx0 > x1 or ty1 < y0 or ty0 > y1:
                continue
            idxs.append(i)
        return idxs

    # ── Tile data loading ────────────────────────────────────────────────

    def get_tile_data(self, tile_idx):
        if tile_idx in self.tile_cache:
            return self.tile_cache[tile_idx]

        tile = self.tiles[tile_idx]
        tile_dir = os.path.join(self.cfg.map_dir, tile["id"])

        gray = cv2.imread(os.path.join(tile_dir, "gray.png"), 0)
        if gray is None:
            gray = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

        # Load SIFT features
        sift_path = os.path.join(tile_dir, "sift_kp.npz")
        if os.path.exists(sift_path):
            data = np.load(sift_path)
            kp_array = data["keypoints"]
            descs = data["descriptors"]
            kps = array_to_keypoints(kp_array)
        else:
            kps, descs = [], np.zeros((0, 128), dtype=np.float32)

        result = {
            "info": tile,
            "gray": gray,
            "keypoints": kps,
            "descriptors": descs,
        }
        self.tile_cache[tile_idx] = result
        return result

    # ── Frame preprocessing ──────────────────────────────────────────────

    def _build_roi_mask(self, h, w):
        mask = np.ones((h, w), dtype=np.uint8) * 255
        body_top = int(h * 0.80)
        mask[body_top:, :] = 0
        cv2.ellipse(mask, (int(w * 0.08), int(h * 0.75)),
                    (int(w * 0.15), int(h * 0.15)), 0, 0, 360, 0, -1)
        cv2.ellipse(mask, (int(w * 0.92), int(h * 0.75)),
                    (int(w * 0.15), int(h * 0.15)), 0, 0, 360, 0, -1)
        cv2.ellipse(mask, (int(w * 0.08), int(h * 0.05)),
                    (int(w * 0.10), int(h * 0.08)), 0, 0, 360, 0, -1)
        cv2.ellipse(mask, (int(w * 0.92), int(h * 0.05)),
                    (int(w * 0.10), int(h * 0.08)), 0, 0, 360, 0, -1)
        return mask

    def preprocess_frame(self, frame):
        """
        Preprocess frame for SIFT extraction.
        Must match tile preprocessing (CLAHE only, NO sharpening)
        to avoid descriptor domain gap.
        Returns: (enhanced_gray, roi_mask, edge_count)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # NOTE: No sharpening here - must match tile preprocessing exactly
        # (sift_map_processor.py uses CLAHE only)

        # ROI mask for SIFT (exclude propellers/body)
        h, w = enhanced.shape
        roi_mask = self._build_roi_mask(h, w)

        # Edge count for density check
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 200)
        edges = cv2.bitwise_and(edges, roi_mask)
        edge_count = int(np.count_nonzero(edges))

        return enhanced, roi_mask, edge_count

    # ── Phase 1: Tile Retrieval ──────────────────────────────────────────

    def retrieve_candidate_tiles(self, frame_descs, prior_px=None):
        """
        Find best candidate tiles using global descriptor similarity.
        If prior_px is available, also filter by spatial proximity.
        """
        candidate_indices = []

        # Spatial filtering (if prior available)
        if prior_px is not None:
            spatial_tiles = self.tiles_in_roi(prior_px, self.cfg.roi_px)
            if spatial_tiles:
                candidate_indices = spatial_tiles
                print(f"  [RETRIEVAL] Prior-based: {len(spatial_tiles)} tiles in ROI")

        # Global descriptor retrieval (if no prior or as supplement)
        if self.global_descs is not None and len(frame_descs) > 0:
            frame_global = frame_descs.astype(np.float64).mean(axis=0)
            norm = np.linalg.norm(frame_global)
            if norm > 1e-6:
                frame_global /= norm
            frame_global = frame_global.astype(np.float32)

            # Cosine similarity against all tile global descriptors
            scores = self.global_descs @ frame_global  # Nx1
            top_k_idx = np.argsort(scores)[::-1][:self.cfg.retrieval_top_k]

            retrieval_tiles = [self.global_desc_indices[i] for i in top_k_idx]
            retrieval_scores = [float(scores[i]) for i in top_k_idx]

            if candidate_indices:
                # Merge: union of spatial + retrieval, prioritize spatial
                combined = list(set(candidate_indices) | set(retrieval_tiles))
                candidate_indices = combined
            else:
                candidate_indices = retrieval_tiles

            print(f"  [RETRIEVAL] Top descriptor scores: "
                  f"{[f'{s:.3f}' for s in retrieval_scores[:5]]}")

        if not candidate_indices:
            candidate_indices = list(range(len(self.tiles)))

        return candidate_indices

    # ── Phase 2: SIFT Matching + Homography ──────────────────────────────

    def match_tile(self, frame_kps, frame_descs, tile_idx, verbose=False):
        """
        Match frame SIFT features against a single tile.
        Returns: dict with H, inlier_mask, matches etc, or None
        """
        tile_data = self.get_tile_data(tile_idx)
        tile_descs = tile_data["descriptors"]
        tile_kps = tile_data["keypoints"]
        tile_id = self.tiles[tile_idx]["id"]

        if len(tile_descs) < 10 or len(frame_descs) < 10:
            if verbose:
                print(f"    [{tile_id}] Too few descriptors: tile={len(tile_descs)} frame={len(frame_descs)}")
            return None

        # BFMatcher knnMatch (more reliable than FLANN for cross-domain matching)
        try:
            raw_matches = self.bf_matcher.knnMatch(
                frame_descs.astype(np.float32),
                tile_descs.astype(np.float32), k=2)
        except cv2.error as e:
            if verbose:
                print(f"    [{tile_id}] BFMatcher error: {e}")
            return None

        # Lowe's ratio test
        good_matches = []
        n_single = 0  # pairs with only 1 match (no ratio test possible)
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.cfg.lowe_ratio * n.distance:
                    good_matches.append(m)
            elif len(pair) == 1:
                n_single += 1

        if verbose or len(good_matches) < self.cfg.min_inliers:
            print(f"    [{tile_id}] raw={len(raw_matches)} ratio_pass={len(good_matches)} "
                  f"single={n_single} tile_kps={len(tile_kps)}")

        if len(good_matches) < self.cfg.min_inliers:
            return None

        # Extract matched point coordinates
        src_pts = np.float32([frame_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([tile_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC homography
        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.cfg.ransac_reproj_thresh)

        if H is None:
            if verbose:
                print(f"    [{tile_id}] RANSAC failed (H is None) from {len(good_matches)} matches")
            return None

        inlier_mask = mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())

        if verbose:
            print(f"    [{tile_id}] inliers={n_inliers}/{len(good_matches)}")

        if n_inliers < self.cfg.min_inliers:
            return None

        return {
            "H": H,
            "inlier_mask": inlier_mask,
            "matches": good_matches,
            "n_inliers": n_inliers,
            "n_total_matches": len(good_matches),
            "src_pts": src_pts,
            "dst_pts": dst_pts,
            "tile_idx": tile_idx,
        }

    # ── Phase 3: Verification ────────────────────────────────────────────

    def compute_reprojection_error(self, match_result):
        """Mean reprojection error of inliers (px)."""
        H = match_result["H"]
        src = match_result["src_pts"]
        dst = match_result["dst_pts"]
        mask = match_result["inlier_mask"]

        src_inlier = src[mask]
        dst_inlier = dst[mask]

        # Project source points through H
        projected = cv2.perspectiveTransform(src_inlier, H)
        errors = np.sqrt(((projected - dst_inlier) ** 2).sum(axis=2))
        return float(errors.mean())

    def compute_inlier_coverage(self, match_result, tile_h, tile_w, grid_size=4):
        """
        Spatial coverage: what fraction of a grid_size x grid_size grid
        has at least one inlier?
        """
        dst = match_result["dst_pts"]
        mask = match_result["inlier_mask"]
        inlier_pts = dst[mask].reshape(-1, 2)

        if len(inlier_pts) == 0:
            return 0.0

        cell_h = tile_h / grid_size
        cell_w = tile_w / grid_size

        occupied = set()
        for pt in inlier_pts:
            gx = min(int(pt[0] / cell_w), grid_size - 1)
            gy = min(int(pt[1] / cell_h), grid_size - 1)
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                occupied.add((gx, gy))

        return len(occupied) / (grid_size * grid_size)

    def compute_ncc(self, frame_gray, tile_gray, H, frame_h, frame_w):
        """
        Warp frame onto tile space using H, compute block-wise NCC.
        """
        tile_h, tile_w = tile_gray.shape[:2]

        # Warp frame to tile coordinates
        warped = cv2.warpPerspective(frame_gray, H, (tile_w, tile_h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)

        # Create validity mask (non-zero after warp)
        warp_mask = (warped > 0).astype(np.uint8)

        # Equalize both for fair comparison
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        tile_eq = clahe.apply(tile_gray)

        # Block-wise NCC (3x3 grid)
        grid = 3
        scores = []
        for gy in range(grid):
            for gx in range(grid):
                y0 = int(tile_h * gy / grid)
                y1 = int(tile_h * (gy + 1) / grid)
                x0 = int(tile_w * gx / grid)
                x1 = int(tile_w * (gx + 1) / grid)

                warp_block = warped[y0:y1, x0:x1].astype(np.float64)
                tile_block = tile_eq[y0:y1, x0:x1].astype(np.float64)
                mask_block = warp_mask[y0:y1, x0:x1]

                valid = mask_block > 0
                if valid.sum() < 100:
                    continue

                a = warp_block[valid]
                b = tile_block[valid]
                a = a - a.mean()
                b = b - b.mean()
                denom = np.sqrt((a * a).sum() * (b * b).sum())
                if denom < 1e-6:
                    continue
                ncc = float((a * b).sum() / denom)
                scores.append(ncc)

        if len(scores) < 3:
            return 0.0

        # Trimmed mean (drop worst 2)
        scores.sort()
        return float(np.mean(scores[2:])) if len(scores) > 2 else float(np.mean(scores))

    def compute_ssim_score(self, frame_gray, tile_gray, H, frame_h, frame_w):
        """
        Simplified SSIM between warped frame and tile.
        Uses block-based luminance + contrast + structure.
        """
        tile_h, tile_w = tile_gray.shape[:2]

        warped = cv2.warpPerspective(frame_gray, H, (tile_w, tile_h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        warp_mask = warped > 0

        # Work on overlapping region only
        valid = warp_mask
        if valid.sum() < 500:
            return 0.0

        a = warped[valid].astype(np.float64)
        b = tile_gray[valid].astype(np.float64)

        mu_a, mu_b = a.mean(), b.mean()
        sig_a, sig_b = a.std(), b.std()
        sig_ab = ((a - mu_a) * (b - mu_b)).mean()

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        ssim = ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (sig_a ** 2 + sig_b ** 2 + C2))
        return float(ssim)

    def verify_match(self, match_result, frame_gray, tile_idx):
        """
        Multi-metric verification of a SIFT match.
        Returns dict with all metrics + pass/fail.
        """
        tile_data = self.get_tile_data(tile_idx)
        tile_gray = tile_data["gray"]
        tile_h, tile_w = tile_gray.shape[:2]
        frame_h, frame_w = frame_gray.shape[:2]
        H = match_result["H"]

        # Metrics
        reproj_error = self.compute_reprojection_error(match_result)
        coverage = self.compute_inlier_coverage(match_result, tile_h, tile_w)
        inlier_ratio = match_result["n_inliers"] / max(match_result["n_total_matches"], 1)
        ncc = self.compute_ncc(frame_gray, tile_gray, H, frame_h, frame_w)
        ssim = self.compute_ssim_score(frame_gray, tile_gray, H, frame_h, frame_w)

        # Decompose homography
        scale, rotation, cx, cy = decompose_homography(H, frame_h, frame_w)

        metrics = {
            "n_inliers": match_result["n_inliers"],
            "n_total_matches": match_result["n_total_matches"],
            "inlier_ratio": inlier_ratio,
            "reproj_error": reproj_error,
            "coverage": coverage,
            "ncc": ncc,
            "ssim": ssim,
            "scale": scale,
            "rotation": rotation,
            "center_x": cx,
            "center_y": cy,
            "H": H,
            "tile_idx": tile_idx,
        }

        # Pass/fail gates
        reject_reasons = []
        if match_result["n_inliers"] < self.cfg.min_inliers:
            reject_reasons.append("INLIERS")
        if inlier_ratio < self.cfg.min_inlier_ratio:
            reject_reasons.append("INLIER_RATIO")
        if reproj_error > self.cfg.max_reproj_error:
            reject_reasons.append("REPROJ")
        if ncc < self.cfg.min_ncc:
            reject_reasons.append("NCC")
        if ssim < self.cfg.min_ssim:
            reject_reasons.append("SSIM")
        if coverage < self.cfg.min_coverage:
            reject_reasons.append("COV")
        if scale < 0.05 or scale > 5.0:
            reject_reasons.append("SCALE_RANGE")

        metrics["passed"] = len(reject_reasons) == 0
        metrics["reject_reasons"] = reject_reasons

        return metrics

    # ── Phase 4: Temporal Consistency ────────────────────────────────────

    def check_temporal(self, current, history):
        """Check temporal consistency between current match and recent history."""
        if not history:
            return True, []

        prev = history[-1]
        reasons = []

        # Position jump check (in metres)
        d_lat = current["lat"] - prev["lat"]
        d_lon = current["lon"] - prev["lon"]
        cos_lat = np.cos(np.radians(current["lat"]))
        dist_m = np.sqrt((d_lat * 111320) ** 2 + (d_lon * 111320 * cos_lat) ** 2)
        if dist_m > self.cfg.max_jump_m:
            reasons.append(f"JUMP_{dist_m:.0f}m")

        # Angle continuity
        angle_diff = abs(current["rotation"] - prev["rotation"])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > self.cfg.max_angle_diff:
            reasons.append(f"ANGLE_{angle_diff:.1f}")

        # Scale drift
        if prev.get("scale", 0) > 0:
            drift = abs(current["scale"] - prev["scale"]) / prev["scale"]
            if drift > self.cfg.max_scale_drift:
                reasons.append(f"SCALE_{drift:.2f}")

        return len(reasons) == 0, reasons

    # ── Uniqueness Check ─────────────────────────────────────────────────

    def check_uniqueness(self, verified_results):
        """
        Check if best match is clearly better than second-best.
        Uses inlier count ratio and NCC difference.
        """
        if len(verified_results) < 2:
            return True, 999.0  # only one candidate = unique

        best = verified_results[0]
        second = verified_results[1]

        # Check if same physical location (overlapping tiles)
        b_tile = self.tiles[best["tile_idx"]]
        s_tile = self.tiles[second["tile_idx"]]
        b_gx = b_tile["x"] + best["center_x"]
        b_gy = b_tile["y"] + best["center_y"]
        s_gx = s_tile["x"] + second["center_x"]
        s_gy = s_tile["y"] + second["center_y"]

        if abs(b_gx - s_gx) < 100 and abs(b_gy - s_gy) < 100:
            return True, 999.0  # same location on overlapping tiles

        # Inlier ratio
        inlier_ratio = best["n_inliers"] / max(second["n_inliers"], 1)

        # NCC difference
        ncc_diff = best["ncc"] - second["ncc"]

        # Must be clearly better
        if inlier_ratio < 1.3 and ncc_diff < 0.05:
            return False, inlier_ratio

        return True, inlier_ratio

    # ── Debug Visualization ──────────────────────────────────────────────

    def save_debug_image(self, frame_idx, frame, frame_gray, match_result,
                         metrics, tile_idx, is_gt, reject_reasons):
        if not self.cfg.save_debug:
            return

        subdir = "gt" if is_gt else "rejected"
        out_dir = os.path.join(self.cfg.debug_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)

        tile_data = self.get_tile_data(tile_idx)
        tile_gray = tile_data["gray"]

        # Draw matches
        tile_kps = tile_data["keypoints"]
        frame_kps_list = match_result.get("_frame_kps", [])

        if match_result and "H" in match_result:
            H = match_result["H"]
            frame_h, frame_w = frame_gray.shape[:2]
            tile_h, tile_w = tile_gray.shape[:2]

            # Warp frame onto tile for overlay
            warped = cv2.warpPerspective(frame_gray, H, (tile_w, tile_h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
            warp_mask = warped > 0

            # Create overlay
            tile_color = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)
            overlay = tile_color.copy()
            overlay[warp_mask] = cv2.addWeighted(
                tile_color[warp_mask], 0.5,
                cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[warp_mask], 0.5, 0
            )

            # Draw inlier matches as green dots
            if "dst_pts" in match_result and "inlier_mask" in match_result:
                inlier_pts = match_result["dst_pts"][match_result["inlier_mask"]].reshape(-1, 2)
                for pt in inlier_pts[:200]:
                    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

            # Draw frame boundary
            corners = np.float32([
                [0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]
            ]).reshape(-1, 1, 2)
            mapped_corners = cv2.perspectiveTransform(corners, H)
            mapped_corners = mapped_corners.astype(np.int32)
            cv2.polylines(overlay, [mapped_corners], True, (0, 0, 255), 3)

            # Resize for display
            scale_disp = 900 / max(tile_h, 1)
            overlay_small = cv2.resize(overlay,
                                       (int(tile_w * scale_disp), int(tile_h * scale_disp)))

            # Add text
            status = "GT" if is_gt else f"REJECT: {';'.join(reject_reasons)}"
            cv2.putText(overlay_small, f"F{frame_idx} {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if is_gt else (0, 0, 255), 2)
            cv2.putText(overlay_small,
                        f"Inliers:{metrics['n_inliers']} NCC:{metrics['ncc']:.3f} "
                        f"SSIM:{metrics['ssim']:.3f} Reproj:{metrics['reproj_error']:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(overlay_small,
                        f"Scale:{metrics['scale']:.3f} Rot:{metrics['rotation']:.1f} "
                        f"Cov:{metrics['coverage']:.2f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}.jpg"), overlay_small)

    # ── Live Visualization ────────────────────────────────────────────────

    def show_live(self, frame_idx, frame_gray, best_metrics, all_verified,
                  frame_kps, is_gt, reject_reasons, n_frame_kps, elapsed):
        """
        Show real-time matching visualization with cv2.imshow.
        Two windows:
          - 'SIFT Matches': side-by-side frame & tile with match lines
          - 'Overlay': warped frame blended on tile
        Press 'q' to quit, SPACE to pause/resume, +/- for speed.
        Returns False if user pressed 'q'.
        """
        if not self.cfg.live_view:
            return True

        frame_h, frame_w = frame_gray.shape[:2]

        # === Window 1: Match lines (or "no match" view) ===
        if best_metrics is not None and best_metrics.get("match_result"):
            match_result = best_metrics["match_result"]
            tile_idx = best_metrics["tile_idx"]
            tile_data = self.get_tile_data(tile_idx)
            tile_gray = tile_data["gray"]
            tile_kps = tile_data["keypoints"]
            tile_id = self.tiles[tile_idx]["id"]
            H = match_result["H"]
            good_matches = match_result["matches"]
            inlier_mask = match_result["inlier_mask"]

            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            tile_bgr = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)

            # Separate inliers/outliers
            inlier_matches = [m for i, m in enumerate(good_matches) if inlier_mask[i]]
            outlier_matches = [m for i, m in enumerate(good_matches) if not inlier_mask[i]]

            # Draw outliers (red, thin)
            match_img = cv2.drawMatches(
                frame_bgr, frame_kps,
                tile_bgr, tile_kps,
                outlier_matches, None,
                matchColor=(0, 0, 150),
                singlePointColor=(80, 80, 80),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Draw inliers on top (green)
            match_img = cv2.drawMatches(
                frame_bgr, frame_kps,
                tile_bgr, tile_kps,
                inlier_matches, match_img,
                matchColor=(0, 255, 0),
                singlePointColor=(0, 200, 0),
                flags=(cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS |
                       cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG))

            # Status bar on top
            status_color = (0, 255, 0) if is_gt else (0, 0, 255) if reject_reasons else (0, 255, 255)
            status_text = "GT" if is_gt else f"REJECT: {';'.join(reject_reasons[:3])}"
            cv2.rectangle(match_img, (0, 0), (match_img.shape[1], 95), (0, 0, 0), -1)
            cv2.putText(match_img, f"F{frame_idx} {status_text} | {tile_id}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(match_img,
                        f"Inliers: {best_metrics['n_inliers']}/{best_metrics['n_total_matches']}  "
                        f"NCC: {best_metrics['ncc']:.3f}  SSIM: {best_metrics['ssim']:.3f}  "
                        f"Reproj: {best_metrics['reproj_error']:.1f}px",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.putText(match_img,
                        f"Scale: {best_metrics['scale']:.3f}  Rot: {best_metrics['rotation']:.1f}  "
                        f"Cov: {best_metrics['coverage']:.2f}  KPs: {n_frame_kps}  [{elapsed:.2f}s]",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Resize to fit screen (max width 1600)
            mh, mw = match_img.shape[:2]
            if mw > 1600:
                scale = 1600 / mw
                match_img = cv2.resize(match_img, (1600, int(mh * scale)))

            cv2.imshow("SIFT Matches", match_img)

            # === Window 2: Overlay ===
            tile_h, tile_w = tile_gray.shape[:2]
            warped = cv2.warpPerspective(frame_gray, H, (tile_w, tile_h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
            warp_mask = warped > 0

            overlay = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)
            warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            overlay[warp_mask] = cv2.addWeighted(
                overlay[warp_mask], 0.5, warped_bgr[warp_mask], 0.5, 0)

            # Frame boundary
            corners = np.float32([
                [0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]
            ]).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(corners, H).astype(np.int32)
            cv2.polylines(overlay, [mapped], True, (0, 0, 255), 3)

            # Inlier dots
            inlier_pts = match_result["dst_pts"][inlier_mask].reshape(-1, 2)
            for pt in inlier_pts[:200]:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

            # Resize overlay to ~800px height
            oh, ow = overlay.shape[:2]
            scale_o = 800 / max(oh, 1)
            overlay_small = cv2.resize(overlay, (int(ow * scale_o), 800))

            cv2.putText(overlay_small, f"F{frame_idx} | {tile_id} | {status_text}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow("Overlay", overlay_small)

        else:
            # No match - show frame only with "NO MATCH" label
            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            if frame_bgr.shape[0] > 800:
                scale = 800 / frame_bgr.shape[0]
                frame_bgr = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)

            cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(frame_bgr,
                        f"F{frame_idx} NO MATCH | kps={n_frame_kps} [{elapsed:.2f}s]",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("SIFT Matches", frame_bgr)

        # Key handling
        key = cv2.waitKey(self.cfg.live_wait_ms) & 0xFF
        if key == ord('q'):
            print("\n[LIVE] User pressed 'q' - stopping.")
            return False
        elif key == ord(' '):
            # Pause - wait until next key
            print("[LIVE] Paused. Press any key to continue, 'q' to quit.")
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord('q'):
                return False
        elif key == ord('+') or key == ord('='):
            self.cfg.live_wait_ms = max(1, self.cfg.live_wait_ms // 2)
            print(f"[LIVE] Speed up: wait={self.cfg.live_wait_ms}ms")
        elif key == ord('-'):
            self.cfg.live_wait_ms = min(2000, self.cfg.live_wait_ms * 2)
            print(f"[LIVE] Slow down: wait={self.cfg.live_wait_ms}ms")

        return True

    # ── Main Pipeline ────────────────────────────────────────────────────

    def process_frame(self, frame_gray, roi_mask, frame_kps=None, frame_descs=None):
        """
        Full pipeline for one frame:
          1. Extract SIFT from frame (if not provided)
          2. Retrieve candidate tiles
          3. Match against each candidate
          4. Verify best match

        Returns: (best_metrics, match_result) or (None, None)
        """
        # Extract SIFT from frame
        if frame_kps is None or frame_descs is None:
            frame_kps, frame_descs = self.sift.detectAndCompute(frame_gray, roi_mask)

        if frame_kps is None or len(frame_kps) < 20:
            print(f"  [SIFT] Only {len(frame_kps) if frame_kps else 0} keypoints in frame")
            return None, None

        # Phase 1: Retrieve candidate tiles
        candidate_tiles = self.retrieve_candidate_tiles(
            frame_descs, prior_px=self._prior_px)

        # Phase 2: Match against each candidate tile
        # Enable verbose for first 5 calls to diagnose matching issues
        if not hasattr(self, '_match_call_count'):
            self._match_call_count = 0
        self._match_call_count += 1
        verbose = self._match_call_count <= 3

        all_results = []
        for tile_idx in candidate_tiles:
            result = self.match_tile(frame_kps, frame_descs, tile_idx, verbose=verbose)
            if result is not None:
                result["_frame_kps"] = frame_kps
                all_results.append(result)

        if not all_results:
            print(f"  [MATCH] No valid homography found in {len(candidate_tiles)} tiles")
            return None, None

        # Sort by inlier count (primary) + total matches (secondary)
        all_results.sort(key=lambda r: (r["n_inliers"], r["n_total_matches"]), reverse=True)

        # Phase 3: Verify top candidates
        verified = []
        for result in all_results[:5]:  # verify top 5
            metrics = self.verify_match(result, frame_gray, result["tile_idx"])
            metrics["match_result"] = result
            verified.append(metrics)

        # Sort by composite score: inliers * ncc * coverage
        verified.sort(
            key=lambda m: m["n_inliers"] * max(m["ncc"], 0) * max(m["coverage"], 0.01),
            reverse=True)

        if not verified:
            return None, None

        best = verified[0]

        # Uniqueness check
        is_unique, uniqueness_score = self.check_uniqueness(verified)
        best["uniqueness"] = uniqueness_score
        if not is_unique:
            best["passed"] = False
            best["reject_reasons"].append("AMBIG")

        return best, verified

    # ── Run Loop ─────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.cfg.video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video {self.cfg.video_path}")
            return

        cap.set(cv2.CAP_PROP_POS_MSEC, self.cfg.skip_seconds * 1000)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[VIDEO] {self.cfg.video_path}: {total_frames} frames @ {fps:.1f} fps")
        print(f"[VIDEO] Skipping first {self.cfg.skip_seconds}s")

        # GT output
        gt_path = "gt_sift.csv"
        gt_file = open(gt_path, "w")
        gt_file.write("frame_idx,timestamp,lat,lon,rotation,scale,score,"
                      "n_inliers,inlier_ratio,reproj_error,ncc,ssim,"
                      "coverage,uniqueness,tile_id\n")

        # All matches log
        all_path = "all_matches_sift.csv"
        all_file = open(all_path, "w")
        all_file.write("frame_idx,timestamp,lat,lon,rotation,scale,score,"
                       "n_inliers,inlier_ratio,reproj_error,ncc,ssim,"
                       "coverage,uniqueness,tile_id,"
                       "is_gt,reject_reason,state,n_frame_kps,time_s\n")

        frame_idx = 0
        history = []
        pending_streak = []
        gt_count = 0

        if self.cfg.save_debug:
            os.makedirs(self.cfg.debug_dir, exist_ok=True)

        print("\nStarting SIFT-based GT processing...\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # SRT GPS prior
            if self.srt_entries:
                e = _srt_lookup(self.srt_entries, timestamp)
                if e is not None:
                    _, lat, lon, alt, yaw = e
                    px = self.geo_to_pixel(lon, lat)
                    self._prior_px = px
                else:
                    self._prior_px = None

            # Rotate (DJI convention)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_idx += 1

            # Downscale if too large
            if frame.shape[1] > 2000:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            t0 = time.time()

            # Preprocess
            enhanced, roi_mask, edge_count = self.preprocess_frame(frame)
            fh, fw = enhanced.shape[:2]

            # Edge density gate
            edge_density = edge_count / max(fh * fw, 1)
            if edge_density < self.cfg.edge_density_min:
                print(f"  [SKIP] F{frame_idx}: edge density {edge_density:.4f} (featureless)")
                pending_streak.clear()
                continue

            # Extract SIFT from frame
            frame_kps, frame_descs = self.sift.detectAndCompute(enhanced, roi_mask)
            n_frame_kps = len(frame_kps) if frame_kps is not None else 0

            if n_frame_kps < 20:
                print(f"  [SKIP] F{frame_idx}: only {n_frame_kps} SIFT keypoints")
                pending_streak.clear()
                continue

            # Process frame
            best_metrics, all_verified = self.process_frame(
                enhanced, roi_mask, frame_kps, frame_descs)

            elapsed = time.time() - t0

            # === Determine GT status ===
            is_gt = False
            reject_reasons = []

            if best_metrics is None:
                reject_reasons.append("NO_MATCH")
                pending_streak.clear()
            elif not best_metrics["passed"]:
                reject_reasons = best_metrics["reject_reasons"]
                pending_streak.clear()
            else:
                # Passed verification gates - now do temporal check
                tile_info = self.tiles[best_metrics["tile_idx"]]
                lat, lon = self.pixel_to_latlon(
                    tile_info, best_metrics["center_x"], best_metrics["center_y"])

                match_data = {
                    "lat": lat, "lon": lon,
                    "rotation": best_metrics["rotation"],
                    "scale": best_metrics["scale"],
                    "tile_idx": best_metrics["tile_idx"],
                    "center_x": best_metrics["center_x"],
                    "center_y": best_metrics["center_y"],
                    "n_inliers": best_metrics["n_inliers"],
                    "ncc": best_metrics["ncc"],
                }

                # Temporal consistency
                temporal_ok, temporal_reasons = self.check_temporal(match_data, history)

                if temporal_ok or not history:
                    pending_streak.append(match_data)
                    if len(pending_streak) > self.cfg.gt_lock_n * 2:
                        pending_streak.pop(0)

                    # Global position jump check
                    if self.last_global_pos is not None:
                        cur_gx = tile_info["x"] + best_metrics["center_x"]
                        cur_gy = tile_info["y"] + best_metrics["center_y"]
                        dx = cur_gx - self.last_global_pos[0]
                        dy = cur_gy - self.last_global_pos[1]
                        pos_jump = (dx ** 2 + dy ** 2) ** 0.5
                        if pos_jump > 200:
                            reject_reasons.append("POS_JUMP")

                    # Scale drift check
                    if self.last_scale is not None:
                        drift = abs(best_metrics["scale"] - self.last_scale) / max(self.last_scale, 0.01)
                        if drift > self.cfg.max_scale_drift:
                            reject_reasons.append("SCALE_DRIFT")

                    # GT lock: need N consecutive frames
                    if len(pending_streak) >= self.cfg.gt_lock_n and not reject_reasons:
                        is_gt = True
                    elif len(pending_streak) < self.cfg.gt_lock_n:
                        reject_reasons.append("TEMP_LOCK")
                else:
                    reject_reasons.extend(temporal_reasons)
                    pending_streak.clear()

            # === Logging ===
            _lat = _lon = _rot = _scale = _score = ""
            _ni = _ir = _re_str = _ncc = _ssim = _cov = _uq = _tile = ""

            if best_metrics is not None:
                tile_info = self.tiles[best_metrics["tile_idx"]]
                lat, lon = self.pixel_to_latlon(
                    tile_info, best_metrics["center_x"], best_metrics["center_y"])
                _lat = f"{lat:.8f}"
                _lon = f"{lon:.8f}"
                _rot = f"{best_metrics['rotation']:.2f}"
                _scale = f"{best_metrics['scale']:.4f}"
                _score = f"{best_metrics['n_inliers'] * max(best_metrics['ncc'], 0):.2f}"
                _ni = str(best_metrics["n_inliers"])
                _ir = f"{best_metrics['inlier_ratio']:.3f}"
                _re_str = f"{best_metrics['reproj_error']:.2f}"
                _ncc = f"{best_metrics['ncc']:.4f}"
                _ssim = f"{best_metrics['ssim']:.4f}"
                _cov = f"{best_metrics['coverage']:.3f}"
                _uq = f"{best_metrics.get('uniqueness', 0):.3f}"
                _tile = tile_info["id"]

            reason_str = ";".join(reject_reasons) if reject_reasons else ""

            all_file.write(
                f"{frame_idx},{timestamp:.3f},{_lat},{_lon},{_rot},{_scale},"
                f"{_score},{_ni},{_ir},{_re_str},{_ncc},{_ssim},{_cov},{_uq},{_tile},"
                f"{'1' if is_gt else '0'},{reason_str},{self.state},"
                f"{n_frame_kps},{elapsed:.4f}\n")
            all_file.flush()

            # === GT Write ===
            if is_gt and best_metrics is not None:
                tile_info = self.tiles[best_metrics["tile_idx"]]
                lat, lon = self.pixel_to_latlon(
                    tile_info, best_metrics["center_x"], best_metrics["center_y"])

                gt_file.write(
                    f"{frame_idx},{timestamp:.3f},{lat:.8f},{lon:.8f},"
                    f"{best_metrics['rotation']:.2f},{best_metrics['scale']:.4f},"
                    f"{best_metrics['n_inliers'] * max(best_metrics['ncc'], 0):.4f},"
                    f"{best_metrics['n_inliers']},{best_metrics['inlier_ratio']:.3f},"
                    f"{best_metrics['reproj_error']:.2f},{best_metrics['ncc']:.4f},"
                    f"{best_metrics['ssim']:.4f},{best_metrics['coverage']:.3f},"
                    f"{best_metrics.get('uniqueness', 0):.3f},{tile_info['id']}\n")
                gt_file.flush()

                # Update state
                self.state = "LOCKED"
                self.last_scale = best_metrics["scale"]
                self.last_angle = best_metrics["rotation"]
                self.last_global_pos = (
                    tile_info["x"] + best_metrics["center_x"],
                    tile_info["y"] + best_metrics["center_y"])
                self._prior_px = self.last_global_pos

                history.append({
                    "lat": lat, "lon": lon,
                    "rotation": best_metrics["rotation"],
                    "scale": best_metrics["scale"],
                })
                if len(history) > 30:
                    history.pop(0)

                gt_count += 1

                # Debug visualization
                if best_metrics.get("match_result"):
                    self.save_debug_image(
                        frame_idx, frame, enhanced,
                        best_metrics["match_result"], best_metrics,
                        best_metrics["tile_idx"], True, [])
            else:
                if self.state == "LOCKED":
                    self.state = "SEARCHING"

                # Debug visualization for rejects (sample every 10th)
                if best_metrics is not None and frame_idx % 10 == 0:
                    if best_metrics.get("match_result"):
                        self.save_debug_image(
                            frame_idx, frame, enhanced,
                            best_metrics["match_result"], best_metrics,
                            best_metrics["tile_idx"], False, reject_reasons)

            # Console log
            status = "GT" if is_gt else f"[{';'.join(reject_reasons[:3])}]" if reject_reasons else "?"
            metrics_str = ""
            if best_metrics:
                metrics_str = (
                    f"inl={best_metrics['n_inliers']} "
                    f"ncc={best_metrics['ncc']:.3f} "
                    f"ssim={best_metrics['ssim']:.3f} "
                    f"reproj={best_metrics['reproj_error']:.1f} "
                    f"cov={best_metrics['coverage']:.2f} "
                    f"s={best_metrics['scale']:.3f} "
                    f"r={best_metrics['rotation']:.1f}")
            print(f"  F{frame_idx} {status} kps={n_frame_kps} {metrics_str} "
                  f"[{elapsed:.2f}s] GT#{gt_count}")

            # === Live Visualization ===
            if self.cfg.live_view:
                keep_going = self.show_live(
                    frame_idx, enhanced, best_metrics, all_verified,
                    frame_kps, is_gt, reject_reasons, n_frame_kps, elapsed)
                if not keep_going:
                    break

        cap.release()
        if self.cfg.live_view:
            cv2.destroyAllWindows()
        gt_file.close()
        all_file.close()

        print(f"\n{'='*60}")
        print(f"Done! {gt_count} GT frames written to {gt_path}")
        print(f"All matches logged to {all_path}")
        if self.cfg.save_debug:
            print(f"Debug images in {self.cfg.debug_dir}/")
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SIFT-based Drone Geo-Localizer for XFeat GT Dataset")

    parser.add_argument("--video", default="DJIG0022.mov", help="Video file path")
    parser.add_argument("--map-dir", default="processed_map", help="Processed tile directory")
    parser.add_argument("--skip", type=float, default=35.0, help="Skip N seconds from start")
    parser.add_argument("--retrieval-k", type=int, default=5, help="Top-K tiles from retrieval")
    parser.add_argument("--min-inliers", type=int, default=15, help="Min RANSAC inliers")
    parser.add_argument("--lowe-ratio", type=float, default=0.75, help="Lowe's ratio test")
    parser.add_argument("--min-ncc", type=float, default=0.25, help="Min NCC for GT")
    parser.add_argument("--gt-lock-n", type=int, default=2, help="Consecutive frames for GT")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug images")
    parser.add_argument("--roi-px", type=int, default=1100, help="Prior ROI radius (pixels)")
    parser.add_argument("--live", action="store_true",
                        help="Enable live cv2.imshow visualization (q=quit, SPACE=pause, +/-=speed)")
    parser.add_argument("--live-wait", type=int, default=1,
                        help="Live view waitKey delay in ms (1=fast, 0=pause each frame)")

    args = parser.parse_args()

    cfg = Config(
        video_path=args.video,
        map_dir=args.map_dir,
        skip_seconds=args.skip,
        retrieval_top_k=args.retrieval_k,
        min_inliers=args.min_inliers,
        lowe_ratio=args.lowe_ratio,
        min_ncc=args.min_ncc,
        gt_lock_n=args.gt_lock_n,
        save_debug=not args.no_debug,
        roi_px=args.roi_px,
        live_view=args.live,
        live_wait_ms=args.live_wait,
    )

    matcher = SIFTMatcher(cfg)
    matcher.run()


if __name__ == "__main__":
    main()
