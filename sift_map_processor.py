"""
SIFT-based Tile Pre-processor for XFeat GT Dataset Generation.

For each tile:
  1. Grayscale + CLAHE enhancement (same as original pipeline)
  2. SIFT keypoint & descriptor extraction
  3. Global descriptor (mean-of-SIFT for fast retrieval)
  4. Saves: gray.png, sift_kp.npz, global_desc.npy (alongside existing dt.npy etc.)

Usage:
    python sift_map_processor.py [--map MAP_PATH] [--output OUTPUT_DIR]
                                 [--tile-size 3072] [--stride 1536]
"""

import cv2
import numpy as np
import rasterio
import os
import json
import argparse
import time
from rasterio.windows import Window


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def preprocess_tile(img):
    """Convert to grayscale, apply CLAHE. Returns None if tile is mostly empty."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if np.count_nonzero(gray) / gray.size < 0.1:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def extract_edge_features(gray):
    """Extract Canny edges, Distance Transform, and LSD lines (same as original)."""
    gaussian = cv2.GaussianBlur(gray, (0, 0), 3.0)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 200)

    # Remove small blobs
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < 30:
            edges[labels == lbl] = 0

    dt_input = cv2.bitwise_not(edges)
    dt = cv2.distanceTransform(dt_input, cv2.DIST_L2, 5)

    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(gray)

    return edges, dt, lines


def extract_sift_features(gray, max_keypoints=8000):
    """
    Extract SIFT keypoints and descriptors from a grayscale tile.
    Returns (keypoints_array, descriptors) where keypoints_array is Nx7:
        [x, y, size, angle, response, octave, class_id]
    """
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    kps, descs = sift.detectAndCompute(gray, None)

    if kps is None or len(kps) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    # Serialize keypoints to numpy array (cv2.KeyPoint is not picklable)
    kp_array = np.zeros((len(kps), 7), dtype=np.float32)
    for i, kp in enumerate(kps):
        kp_array[i] = [kp.pt[0], kp.pt[1], kp.size, kp.angle,
                        kp.response, kp.octave, kp.class_id]

    if descs is None:
        descs = np.zeros((len(kps), 128), dtype=np.float32)

    return kp_array, descs


def compute_global_descriptor(descriptors):
    """
    Compute a simple global descriptor for fast tile retrieval.
    Uses mean-of-SIFT + L2 normalization (GeM-lite).
    Returns a 128-dim float32 vector.
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(128, dtype=np.float32)

    # Mean pooling of all SIFT descriptors
    global_desc = descriptors.astype(np.float64).mean(axis=0)

    # L2 normalize
    norm = np.linalg.norm(global_desc)
    if norm > 1e-6:
        global_desc /= norm

    return global_desc.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="SIFT-based tile pre-processor")
    parser.add_argument("--map", default="dikilitas_new.tif", help="Path to GeoTIFF")
    parser.add_argument("--output", default="processed_map", help="Output directory")
    parser.add_argument("--tile-size", type=int, default=3072, help="Tile size in pixels")
    parser.add_argument("--stride", type=int, default=1536, help="Stride between tiles")
    parser.add_argument("--max-keypoints", type=int, default=8000,
                        help="Max SIFT keypoints per tile")
    args = parser.parse_args()

    ensure_dir(args.output)

    print(f"Opening map: {args.map}")
    with rasterio.open(args.map) as src:
        width = src.width
        height = src.height
        print(f"Map size: {width}x{height}")

        transform_vals = [src.transform.a, src.transform.b, src.transform.c,
                          src.transform.d, src.transform.e, src.transform.f]
        crs_wkt = src.crs.to_wkt() if src.crs else None
        print(f"Map CRS: {src.crs}")
        print(f"Map Transform: {src.transform}")

        metadata = {
            "map_path": args.map,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "crs_wkt": crs_wkt,
            "transform": transform_vals,
            "tiles": []
        }

        total_kps = 0
        t_start = time.time()

        for y in range(0, height, args.stride):
            for x in range(0, width, args.stride):
                window = Window(x, y,
                                min(args.tile_size, width - x),
                                min(args.tile_size, height - y))

                if window.width < args.tile_size or window.height < args.tile_size:
                    print(f"  Skipping partial tile {x},{y}: {window.width}x{window.height}")
                    continue

                img_data = src.read(window=window)
                if img_data.shape[0] == 1:
                    img = img_data[0]
                else:
                    img = np.transpose(img_data[:3], (1, 2, 0))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                gray = preprocess_tile(img)
                if gray is None:
                    print(f"  Skipping empty tile {x},{y}")
                    continue

                tile_id = f"tile_{x}_{y}"
                tile_dir = os.path.join(args.output, tile_id)
                ensure_dir(tile_dir)

                # 1. Edge features (backward compatible with old pipeline)
                edges, dt, lines = extract_edge_features(gray)
                cv2.imwrite(os.path.join(tile_dir, "gray.png"), gray)
                cv2.imwrite(os.path.join(tile_dir, "edges.png"), edges)
                np.save(os.path.join(tile_dir, "dt.npy"), dt)
                np.save(os.path.join(tile_dir, "lines.npy"),
                        lines if lines is not None else np.array([]))

                # 2. SIFT features (new)
                kp_array, descs = extract_sift_features(gray, args.max_keypoints)
                np.savez_compressed(os.path.join(tile_dir, "sift_kp.npz"),
                                    keypoints=kp_array, descriptors=descs)

                # 3. Global descriptor (new)
                global_desc = compute_global_descriptor(descs)
                np.save(os.path.join(tile_dir, "global_desc.npy"), global_desc)

                n_kp = len(kp_array)
                total_kps += n_kp

                metadata["tiles"].append({
                    "id": tile_id,
                    "x": x,
                    "y": y,
                    "width": int(window.width),
                    "height": int(window.height),
                    "sift_count": n_kp
                })

                print(f"  {tile_id}: {n_kp} SIFT keypoints, "
                      f"{int(np.count_nonzero(edges))} edge pixels")

        with open(os.path.join(args.output, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - t_start
        print(f"\nDone! {len(metadata['tiles'])} tiles, "
              f"{total_kps} total keypoints in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
