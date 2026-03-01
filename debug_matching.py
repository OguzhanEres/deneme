"""
Debug script: Analyze why SIFT ratio test fails so badly.
Checks distance distributions and descriptor statistics.
"""
import cv2
import numpy as np
import json
import os
import sys

# Load a frame
video_path = "DJIG0022.mov"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
skip_frames = int(35.0 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame")
    sys.exit(1)

print(f"Frame shape: {frame.shape}")

# Preprocess frame (same as sift_matcher.py)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# ROI mask
h, w = enhanced.shape
roi_mask = np.ones((h, w), dtype=np.uint8) * 255
body_top = int(h * 0.80)
roi_mask[body_top:, :] = 0

# Extract SIFT from frame
sift = cv2.SIFT_create(nfeatures=4000)
frame_kps, frame_descs = sift.detectAndCompute(enhanced, roi_mask)
print(f"Frame: {len(frame_kps)} keypoints, descriptors shape={frame_descs.shape}")
print(f"Frame desc: mean={frame_descs.mean():.3f} std={frame_descs.std():.3f} "
      f"min={frame_descs.min():.3f} max={frame_descs.max():.3f}")
print(f"Frame desc dtype={frame_descs.dtype}")

# Load tile descriptors
map_dir = "processed_map"
with open(os.path.join(map_dir, "metadata.json")) as f:
    meta = json.load(f)

# Check ALL tiles
for tile_info in meta["tiles"]:
    tile_id = tile_info["id"]
    tile_dir = os.path.join(map_dir, tile_id)
    sift_path = os.path.join(tile_dir, "sift_kp.npz")

    if not os.path.exists(sift_path):
        print(f"\n{tile_id}: no SIFT data")
        continue

    data = np.load(sift_path)
    tile_descs = data["descriptors"]

    print(f"\n--- {tile_id} ---")
    print(f"  Tile desc: mean={tile_descs.mean():.3f} std={tile_descs.std():.3f} "
          f"min={tile_descs.min():.3f} max={tile_descs.max():.3f}")
    print(f"  Tile desc dtype={tile_descs.dtype}")

    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(
        frame_descs.astype(np.float32),
        tile_descs.astype(np.float32), k=2)

    # Analyze distance distribution
    ratios = []
    d1_list = []
    d2_list = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            ratios.append(m.distance / n.distance)
            d1_list.append(m.distance)
            d2_list.append(n.distance)

    ratios = np.array(ratios)
    d1_list = np.array(d1_list)
    d2_list = np.array(d2_list)

    print(f"  Distance stats (best match d1):")
    print(f"    mean={d1_list.mean():.1f} std={d1_list.std():.1f} "
          f"min={d1_list.min():.1f} max={d1_list.max():.1f}")
    print(f"  Distance stats (2nd best d2):")
    print(f"    mean={d2_list.mean():.1f} std={d2_list.std():.1f} "
          f"min={d2_list.min():.1f} max={d2_list.max():.1f}")
    print(f"  Ratio stats (d1/d2):")
    print(f"    mean={ratios.mean():.4f} std={ratios.std():.4f} "
          f"min={ratios.min():.4f} max={ratios.max():.4f}")

    # Show how many pass at various thresholds
    for thresh in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]:
        n_pass = (ratios < thresh).sum()
        print(f"    ratio < {thresh}: {n_pass} matches ({100*n_pass/len(ratios):.1f}%)")

    # Check if descriptors are normalized differently
    frame_norms = np.linalg.norm(frame_descs.astype(np.float32), axis=1)
    tile_norms = np.linalg.norm(tile_descs.astype(np.float32), axis=1)
    print(f"  Frame descriptor L2 norms: mean={frame_norms.mean():.1f} std={frame_norms.std():.1f}")
    print(f"  Tile descriptor L2 norms: mean={tile_norms.mean():.1f} std={tile_norms.std():.1f}")

    # Just check first tile thoroughly
    if tile_id == meta["tiles"][0]["id"]:
        # Also try with crossCheck
        bf2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        cross_matches = bf2.match(
            frame_descs.astype(np.float32),
            tile_descs.astype(np.float32))
        print(f"\n  CrossCheck matches: {len(cross_matches)}")
        if cross_matches:
            cross_dists = [m.distance for m in cross_matches]
            print(f"  CrossCheck distance: mean={np.mean(cross_dists):.1f} "
                  f"min={np.min(cross_dists):.1f} max={np.max(cross_dists):.1f}")
