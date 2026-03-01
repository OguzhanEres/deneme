"""Quick diagnostic: check if tiles are actually valid."""
import os
import json
import cv2
import numpy as np

map_dir = "processed_map"

# Load metadata
with open(os.path.join(map_dir, "metadata.json")) as f:
    meta = json.load(f)

print(f"Found {len(meta['tiles'])} tiles")
print(f"Source map: {meta['map_path']}")
print(f"Tile size: {meta['tile_size']}, stride: {meta['stride']}")

# Check first tile
tile0 = meta['tiles'][0]
tile_dir = os.path.join(map_dir, tile0['id'])
print(f"\nChecking first tile: {tile0['id']}")

# Check gray.png
gray_path = os.path.join(tile_dir, "gray.png")
if os.path.exists(gray_path):
    gray = cv2.imread(gray_path, 0)
    print(f"  gray.png: {gray.shape} dtype={gray.dtype}")
    print(f"  gray mean={gray.mean():.1f} std={gray.std():.1f} nonzero={np.count_nonzero(gray)}/{gray.size}")
else:
    print(f"  ERROR: gray.png not found!")

# Check SIFT
sift_path = os.path.join(tile_dir, "sift_kp.npz")
if os.path.exists(sift_path):
    data = np.load(sift_path)
    kps = data['keypoints']
    descs = data['descriptors']
    print(f"  sift_kp.npz: {len(kps)} keypoints, descriptors shape={descs.shape}")
    print(f"  descriptor mean={descs.mean():.3f} std={descs.std():.3f}")
else:
    print(f"  ERROR: sift_kp.npz not found!")

# Check if tile is just blank/uniform
if os.path.exists(gray_path):
    gray = cv2.imread(gray_path, 0)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.count_nonzero(edges)
    print(f"  Canny edges: {edge_count} ({100*edge_count/edges.size:.2f}%)")

    if edge_count < 1000:
        print("  WARNING: Very few edges - tile might be blank/uniform!")
