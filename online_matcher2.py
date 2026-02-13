import cv2
import numpy as np
import math
import time

# ===============================
# STRICT GT MODE (XFeat Ready)
# ===============================

GT_MODE = True

GT_MEDIAN_MAX = 6.0
GT_CORE_RATIO_MIN = 0.18
GT_COVERAGE_MIN = 0.45
GT_ORIENT_MIN = 0.15
GT_LIMIT_MIN = 1200

GT_LOCK_N = 3
TEMPORAL_MAX_JUMP_M = 20.0
TEMPORAL_MAX_ANGLE_DIFF = 5.0

TRACK_SCALE_DELTA = 0.08
TRACK_ANGLE_DELTA = 2.0

COARSE_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]
ROTATION_RANGE = 15
ROTATION_STEP = 5


class OnlineMatcher:

    def __init__(self, tiles, stride):
        self.tiles = tiles
        self.stride = stride

        self.state = "GLOBAL"
        self.last_good = None
        self.fail_count = 0
        self.pending_streak = []

    # -----------------------------------------------------

    def _get_neighbor_tiles(self, tile_idx):
        base = self.tiles[tile_idx]
        cx, cy = base["x"], base["y"]

        neighbors = []
        for i, t in enumerate(self.tiles):
            if abs(t["x"] - cx) <= self.stride * 2 and \
               abs(t["y"] - cy) <= self.stride * 2:
                neighbors.append(i)

        return neighbors

    # -----------------------------------------------------

    def check_temporal_consistency(
        self, current, history,
        max_dist_m, max_angle_diff, required_n
    ):
        history = history[-(required_n - 1):]

        for h in history:
            dist = abs(current["center_x"] - h["center_x"]) + \
                   abs(current["center_y"] - h["center_y"])

            angle_diff = abs(current["angle"] - h["angle"])

            if dist > max_dist_m:
                return False

            if angle_diff > max_angle_diff:
                return False

        return True

    # -----------------------------------------------------

    def coarse_search(self, frame):

        best = None
        best_score = -1

        if GT_MODE and self.state == "TRACK" and self.last_good:
            search_indices = self._get_neighbor_tiles(
                self.last_good["tile_idx"]
            )

            last_scale = self.last_good["scale"]
            scales = [
                s for s in COARSE_SCALES
                if abs(s - last_scale) <= last_scale * TRACK_SCALE_DELTA
            ]

            angle_min = self.last_good["angle"] - TRACK_ANGLE_DELTA
            angle_max = self.last_good["angle"] + TRACK_ANGLE_DELTA

        else:
            search_indices = range(len(self.tiles))
            scales = COARSE_SCALES
            angle_min = -ROTATION_RANGE
            angle_max = ROTATION_RANGE

        for idx in search_indices:

            tile = self.tiles[idx]["image"]

            for scale in scales:

                scaled = cv2.resize(
                    tile,
                    None,
                    fx=scale,
                    fy=scale
                )

                for angle in np.arange(
                    angle_min,
                    angle_max + 0.1,
                    ROTATION_STEP
                ):

                    M = cv2.getRotationMatrix2D(
                        (scaled.shape[1] // 2,
                         scaled.shape[0] // 2),
                        angle,
                        1.0
                    )

                    rotated = cv2.warpAffine(
                        scaled,
                        M,
                        (scaled.shape[1], scaled.shape[0])
                    )

                    if frame.shape[0] < rotated.shape[0] or \
                       frame.shape[1] < rotated.shape[1]:
                        continue

                    res = cv2.matchTemplate(
                        frame,
                        rotated,
                        cv2.TM_CCOEFF_NORMED
                    )

                    _, score, _, loc = cv2.minMaxLoc(res)

                    if score > best_score:
                        best_score = score
                        best = {
                            "tile_idx": idx,
                            "scale": scale,
                            "angle": angle,
                            "score": score,
                            "center_x": loc[0],
                            "center_y": loc[1]
                        }

        return best

    # -----------------------------------------------------

    def quality_gate(self, match):

        if match["score"] < 0.35:
            return False

        if match["score"] > 0.99:
            return False

        return True

    # -----------------------------------------------------

    def run(self, frame):

        best_match = self.coarse_search(frame)

        if best_match is None:
            return

        is_good = self.quality_gate(best_match)

        if not is_good:
            self.fail_count += 1
        else:
            self.pending_streak.append(best_match)

            if len(self.pending_streak) >= GT_LOCK_N:

                if self.check_temporal_consistency(
                    best_match,
                    self.pending_streak,
                    TEMPORAL_MAX_JUMP_M,
                    TEMPORAL_MAX_ANGLE_DIFF,
                    GT_LOCK_N
                ):

                    print("GT_SAVED")

                    self.state = "TRACK"
                    self.last_good = {
                        "tile_idx": best_match["tile_idx"],
                        "scale": best_match["scale"],
                        "angle": best_match["angle"]
                    }

                    self.fail_count = 0
                    self.pending_streak = []
                    return

        if self.fail_count >= 3:
            self.state = "GLOBAL"
            self.last_good = None
            self.fail_count = 0
            self.pending_streak = []

            print("RESET → GLOBAL")
