
import cv2
import numpy as np
import rasterio
import os
import json
from rasterio.windows import Window

# Settings
MAP_PATH = "dikilitas_new.tif"
OUTPUT_DIR = "processed_map"

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def preprocess_tile(img):
    """
    Convert to grayscale if needed, apply CLAHE.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Check if tile has enough data (not mostly black/nodata)
    if np.count_nonzero(gray) / gray.size < 0.1:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def extract_features(gray):
    """
    Extract Canny edges, Distance Transform, and Lines.
    """
    # 1. Canny Edges (blur + high thresholds + erosion to reduce density)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 200)

    # 2. Distance Transform
    # Invert edges for DT (0=edge, 1=background)
    # dist_transform expects 0 at the feature, non-zero elsewhere
    # So we invert edges: 255 (edge) -> 0, 0 (bg) -> 255
    dt_input = cv2.bitwise_not(edges)
    dt = cv2.distanceTransform(dt_input, cv2.DIST_L2, 5)

    # 3. Lines (LSD)
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(gray)

    # Filter short lines? Maybe later.

    return edges, dt, lines

def main():
    ensure_dir(OUTPUT_DIR)

    print(f"Opening map: {MAP_PATH}")
    with rasterio.open(MAP_PATH) as src:
        width = src.width
        height = src.height

        print(f"Map size: {width}x{height}")

        # Single tile = entire map. No tiling needed for maps up to ~10k px.
        # This eliminates tile disambiguation entirely: no overlap, no tile jumping.
        TILE_SIZE = max(width, height)
        STRIDE = TILE_SIZE

        print(f"Using single-tile mode: {TILE_SIZE}x{TILE_SIZE}")

        # Get CRS and Transform
        transform_vals = [src.transform.a, src.transform.b, src.transform.c,
                          src.transform.d, src.transform.e, src.transform.f]

        crs_wkt = src.crs.to_wkt() if src.crs else None

        print(f"Map CRS: {src.crs}")
        print(f"Map Transform: {src.transform}")

        metadata = {
            "map_path": MAP_PATH,
            "tile_size": TILE_SIZE,
            "stride": STRIDE,
            "crs_wkt": crs_wkt,
            "transform": transform_vals,
            "tiles": []
        }

        # Read entire map as a single tile
        img_data = src.read()

        # Handle shapes
        if img_data.shape[0] == 1:
            img = img_data[0]
        else:
            img = np.transpose(img_data[:3], (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Preprocess
        gray = preprocess_tile(img)
        if gray is None:
            print("ERROR: Map image has insufficient data!")
            return

        # Extract features
        edges, dt, lines = extract_features(gray)

        # Save
        tile_id = "tile_0_0"
        tile_dir = os.path.join(OUTPUT_DIR, tile_id)
        ensure_dir(tile_dir)

        cv2.imwrite(os.path.join(tile_dir, "gray.png"), gray)
        cv2.imwrite(os.path.join(tile_dir, "edges.png"), edges)
        np.save(os.path.join(tile_dir, "dt.npy"), dt)
        if lines is not None:
            np.save(os.path.join(tile_dir, "lines.npy"), lines)
        else:
            np.save(os.path.join(tile_dir, "lines.npy"), np.array([]))

        metadata["tiles"].append({
            "id": tile_id,
            "x": 0,
            "y": 0,
            "width": width,
            "height": height
        })

        print(f"Processed {tile_id} ({width}x{height})")

        # Save metadata
        with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print("Map processing complete.")

if __name__ == "__main__":
    main()
