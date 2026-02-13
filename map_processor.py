
import cv2
import numpy as np
import rasterio
import os
import json
from rasterio.windows import Window

# Settings
MAP_PATH = "dikilitas_new.tif"
OUTPUT_DIR = "processed_map"
TILE_SIZE = 3072  # covers drone FOV (~2500px at scale 0.35) with margin
STRIDE = 1536     # 50% overlap instead of 87.5% → tiles more distinguishable

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
        
        # Get CRS and Transform
        # rasterio.transform is Affine object. Convert to list/tuple for JSON.
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
        
        # Sliding window
        for y in range(0, height, STRIDE):
            for x in range(0, width, STRIDE):
                # Check if window is within bounds (or handle partial tiles)
                # For simplicity, let's process partial tiles by padding or just clipping
                # But for registration, fixed size is better. Let's clip and if too small, skip or pad.
                # Actually, reading with Window automatically handles clipping if we request valid bounds.
                # But we want consistent 2048x2048 for the matcher.
                
                # Let's adjust window to be full size if possible, or pad.
                # If x + TILE_SIZE > width, we can either:
                # 1. Skip if overlap covers it
                # 2. Shift back to fit
                # 3. Pad
                
                # Let's simple clip for reading, then pad for processing
                window = Window(x, y, min(TILE_SIZE, width - x), min(TILE_SIZE, height - y))
                
                if window.width < TILE_SIZE or window.height < TILE_SIZE:
                    print(f"Skipping partial tile {x},{y}: {window.width}x{window.height}")
                    continue

                img_data = src.read(window=window)
                
                # Handle shapes
                if img_data.shape[0] == 1:
                    # Grayscale
                    img = img_data[0]
                else:
                    # RGB or RGBA. Take first 3 bands and assumes RGB
                    img = np.transpose(img_data[:3], (1, 2, 0))
                    # Convert RGB to BGR for OpenCV
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # No padding needed if we skip partials

                # Preprocess
                gray = preprocess_tile(img)
                if gray is None:
                    continue
                
                # Extract features
                edges, dt, lines = extract_features(gray)
                
                # Save
                tile_id = f"tile_{x}_{y}"
                tile_dir = os.path.join(OUTPUT_DIR, tile_id)
                ensure_dir(tile_dir)
                
                cv2.imwrite(os.path.join(tile_dir, "gray.png"), gray)
                cv2.imwrite(os.path.join(tile_dir, "edges.png"), edges)
                np.save(os.path.join(tile_dir, "dt.npy"), dt)
                # Lines might be None
                if lines is not None:
                    np.save(os.path.join(tile_dir, "lines.npy"), lines)
                else:
                    np.save(os.path.join(tile_dir, "lines.npy"), np.array([]))
                
                # Store metadata
                metadata["tiles"].append({
                    "id": tile_id,
                    "x": x,
                    "y": y,
                    "width": window.width, # Original valid width
                    "height": window.height # Original valid height
                })
                
                print(f"Processed {tile_id}")
        
        # Save metadata
        with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    # ======================================================================
    # PASS 2: Per-tile uniqueness mask
    # For each tile, find edges that do NOT appear at the same global position
    # in any other overlapping tile. These unique edges are the best features
    # for disambiguation (farmland edges repeat across tiles).
    # ======================================================================
    print("\n--- Computing per-tile uniqueness masks ---")
    tiles_meta = metadata["tiles"]
    n_tiles = len(tiles_meta)

    # Load all edges into memory indexed by global origin
    all_edges = {}
    for tm in tiles_meta:
        epath = os.path.join(OUTPUT_DIR, tm["id"], "edges.png")
        all_edges[tm["id"]] = cv2.imread(epath, 0)

    for i, ti in enumerate(tiles_meta):
        edges_i = all_edges[ti["id"]]
        # Start with all edge pixels marked as unique
        unique_mask = edges_i.copy()  # 255 where edge, 0 elsewhere

        for j, tj in enumerate(tiles_meta):
            if i == j:
                continue
            # Compute overlap region in global coords
            ox0 = max(ti["x"], tj["x"])
            oy0 = max(ti["y"], tj["y"])
            ox1 = min(ti["x"] + TILE_SIZE, tj["x"] + TILE_SIZE)
            oy1 = min(ti["y"] + TILE_SIZE, tj["y"] + TILE_SIZE)
            if ox1 <= ox0 or oy1 <= oy0:
                continue  # no overlap

            # Local coords in tile i
            li_x0 = ox0 - ti["x"]
            li_y0 = oy0 - ti["y"]
            li_x1 = ox1 - ti["x"]
            li_y1 = oy1 - ti["y"]

            # Local coords in tile j
            lj_x0 = ox0 - tj["x"]
            lj_y0 = oy0 - tj["y"]
            lj_x1 = ox1 - tj["x"]
            lj_y1 = oy1 - tj["y"]

            # Edges that exist in BOTH tiles at the same global position → not unique
            overlap_i = edges_i[li_y0:li_y1, li_x0:li_x1]
            overlap_j = all_edges[tj["id"]][lj_y0:lj_y1, lj_x0:lj_x1]

            # Dilate slightly to account for 1-2px alignment differences
            kernel = np.ones((3, 3), np.uint8)
            shared = cv2.bitwise_and(overlap_i, cv2.dilate(overlap_j, kernel, iterations=1))

            # Remove shared edges from unique mask
            unique_mask[li_y0:li_y1, li_x0:li_x1] = cv2.bitwise_and(
                unique_mask[li_y0:li_y1, li_x0:li_x1],
                cv2.bitwise_not(shared))

        # Save uniqueness mask
        upath = os.path.join(OUTPUT_DIR, ti["id"], "unique_edges.png")
        cv2.imwrite(upath, unique_mask)
        n_total = np.count_nonzero(edges_i)
        n_unique = np.count_nonzero(unique_mask)
        pct = 100 * n_unique / max(n_total, 1)
        print(f"  {ti['id']}: {n_unique}/{n_total} unique edges ({pct:.1f}%)")

    print("Map processing complete.")

if __name__ == "__main__":
    main()
