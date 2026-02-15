"""
SIFT Match Visualization Tool.

Produces visual outputs for inspecting SIFT matching quality:
  1. Side-by-side drawMatches (inliers green, outliers red)
  2. Warped overlay on tile (blended)
  3. Combined panel with metrics

Usage:
    # Visualize a single frame from video
    python visualize_matches.py --video DJIG0022.mov --frame 100 --map-dir processed_map

    # Visualize multiple frames (e.g. every 30th frame, first 10 matches)
    python visualize_matches.py --video DJIG0022.mov --every 30 --max-vis 10 --map-dir processed_map

    # Specify output directory
    python visualize_matches.py --video DJIG0022.mov --frame 100 --out vis_output
"""

import cv2
import numpy as np
import os
import argparse
import time

from sift_matcher import SIFTMatcher, Config, array_to_keypoints


def draw_match_panel(frame_gray, tile_gray, frame_kps, tile_kps,
                     good_matches, inlier_mask, H, metrics, tile_id,
                     frame_idx, max_height=800):
    """
    Create a comprehensive match visualization panel:
      Top: side-by-side drawMatches (inliers=green, outliers=red)
      Bottom-left: warped frame overlay on tile
      Bottom-right: metrics text
    """

    # ── 1. Side-by-side match drawing ──
    # Separate inlier and outlier matches
    inlier_matches = []
    outlier_matches = []
    for i, m in enumerate(good_matches):
        if inlier_mask[i]:
            inlier_matches.append(m)
        else:
            outlier_matches.append(m)

    # Convert grayscale to BGR for colored drawing
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    tile_bgr = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)

    # Draw outliers first (red, thin)
    match_img = cv2.drawMatches(
        frame_bgr, frame_kps,
        tile_bgr, tile_kps,
        outlier_matches, None,
        matchColor=(0, 0, 180),       # dark red for outliers
        singlePointColor=(100, 100, 100),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Draw inliers on top (green, thick)
    match_img = cv2.drawMatches(
        frame_bgr, frame_kps,
        tile_bgr, tile_kps,
        inlier_matches, match_img,
        matchColor=(0, 255, 0),        # green for inliers
        singlePointColor=(0, 200, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)

    # ── 2. Warped overlay ──
    tile_h, tile_w = tile_gray.shape[:2]
    frame_h, frame_w = frame_gray.shape[:2]

    warped = cv2.warpPerspective(frame_gray, H, (tile_w, tile_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    warp_mask = warped > 0

    # Blended overlay (tile=blue channel, warped frame=green channel)
    overlay = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)
    warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # Where warped is valid: blend 50/50
    overlay[warp_mask] = cv2.addWeighted(
        overlay[warp_mask], 0.5,
        warped_color[warp_mask], 0.5, 0)

    # Draw frame boundary polygon on overlay
    corners = np.float32([
        [0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]
    ]).reshape(-1, 1, 2)
    mapped_corners = cv2.perspectiveTransform(corners, H).astype(np.int32)
    cv2.polylines(overlay, [mapped_corners], True, (0, 0, 255), 3)

    # Draw inlier points on overlay
    for i, m in enumerate(good_matches):
        if inlier_mask[i]:
            pt = tile_kps[m.trainIdx].pt
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

    # ── 3. Combine into a single panel ──

    # Scale match_img to max_height
    mh, mw = match_img.shape[:2]
    scale_match = max_height / mh
    match_resized = cv2.resize(match_img, (int(mw * scale_match), max_height))

    # Scale overlay to same height
    oh, ow = overlay.shape[:2]
    scale_overlay = max_height / oh
    overlay_resized = cv2.resize(overlay, (int(ow * scale_overlay), max_height))

    # Create metrics panel
    metrics_w = 420
    metrics_panel = np.zeros((max_height, metrics_w, 3), dtype=np.uint8)
    metrics_panel[:] = (30, 30, 30)  # dark background

    # Draw metrics text
    y_pos = 35
    line_h = 32
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)

    def put(text, color=white, scale=font_scale):
        nonlocal y_pos
        cv2.putText(metrics_panel, text, (15, y_pos), font, scale, color, 1, cv2.LINE_AA)
        y_pos += line_h

    put(f"Frame #{frame_idx}", cyan, 0.8)
    put(f"Tile: {tile_id}", cyan, 0.8)
    y_pos += 10

    passed = metrics.get("passed", False)
    status_color = green if passed else red
    status_text = "PASSED" if passed else "REJECTED"
    put(f"Status: {status_text}", status_color, 0.8)

    if not passed and metrics.get("reject_reasons"):
        put(f"  Reasons: {', '.join(metrics['reject_reasons'])}", red, 0.5)
    y_pos += 10

    put("--- Match Stats ---", yellow)
    put(f"Inliers: {metrics.get('n_inliers', '?')} / {metrics.get('n_total_matches', '?')}")
    ratio = metrics.get('inlier_ratio', 0)
    put(f"Inlier Ratio: {ratio:.3f}", green if ratio >= 0.10 else red)

    reproj = metrics.get('reproj_error', 999)
    put(f"Reproj Error: {reproj:.2f} px", green if reproj <= 8.0 else red)

    cov = metrics.get('coverage', 0)
    put(f"Coverage: {cov:.3f}", green if cov >= 0.30 else red)
    y_pos += 10

    put("--- Similarity ---", yellow)
    ncc = metrics.get('ncc', 0)
    put(f"NCC: {ncc:.4f}", green if ncc >= 0.25 else red)

    ssim = metrics.get('ssim', 0)
    put(f"SSIM: {ssim:.4f}", green if ssim >= 0.20 else red)
    y_pos += 10

    put("--- Geometry ---", yellow)
    put(f"Scale: {metrics.get('scale', 0):.4f}")
    put(f"Rotation: {metrics.get('rotation', 0):.2f} deg")

    uniq = metrics.get('uniqueness', 0)
    if uniq < 999:
        put(f"Uniqueness: {uniq:.3f}")

    # ── Bottom row: overlay + metrics side by side ──
    bottom_row = np.hstack([overlay_resized, metrics_panel])

    # Match the widths for stacking
    top_w = match_resized.shape[1]
    bot_w = bottom_row.shape[1]

    if top_w < bot_w:
        pad = np.zeros((max_height, bot_w - top_w, 3), dtype=np.uint8)
        match_resized = np.hstack([match_resized, pad])
    elif bot_w < top_w:
        pad = np.zeros((max_height, top_w - bot_w, 3), dtype=np.uint8)
        bottom_row = np.hstack([bottom_row, pad])

    panel = np.vstack([match_resized, bottom_row])

    return panel


def visualize_single_frame(matcher, cap, frame_idx_target, out_dir, skip_seconds):
    """Process and visualize a specific frame."""
    cap.set(cv2.CAP_PROP_POS_MSEC, skip_seconds * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"  Could not read frame {frame_idx_target}")
            return False

        frame_count += 1
        if frame_count < frame_idx_target:
            continue

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # DJI rotation
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if frame.shape[1] > 2000:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        t0 = time.time()

        # Preprocess
        enhanced, roi_mask, edge_count = matcher.preprocess_frame(frame)
        fh, fw = enhanced.shape[:2]

        # Extract SIFT
        frame_kps, frame_descs = matcher.sift.detectAndCompute(enhanced, roi_mask)
        n_kps = len(frame_kps) if frame_kps else 0
        print(f"  Frame {frame_idx_target}: {n_kps} SIFT keypoints")

        if n_kps < 20:
            print(f"  Too few keypoints, skipping")
            return False

        # Run matching pipeline
        best_metrics, all_verified = matcher.process_frame(
            enhanced, roi_mask, frame_kps, frame_descs)

        elapsed = time.time() - t0

        if best_metrics is None:
            print(f"  No match found for frame {frame_idx_target}")
            return False

        match_result = best_metrics.get("match_result")
        if match_result is None:
            print(f"  No match result data")
            return False

        tile_idx = best_metrics["tile_idx"]
        tile_data = matcher.get_tile_data(tile_idx)
        tile_gray = tile_data["gray"]
        tile_kps = tile_data["keypoints"]
        tile_id = matcher.tiles[tile_idx]["id"]

        # Build inlier mask for good_matches
        good_matches = match_result["matches"]
        inlier_mask = match_result["inlier_mask"]
        H = match_result["H"]

        print(f"  Best tile: {tile_id}")
        print(f"  Inliers: {best_metrics['n_inliers']}/{best_metrics['n_total_matches']}")
        print(f"  NCC={best_metrics['ncc']:.4f}  SSIM={best_metrics['ssim']:.4f}  "
              f"Reproj={best_metrics['reproj_error']:.2f}")
        print(f"  Passed: {best_metrics['passed']}  [{elapsed:.2f}s]")

        # Generate visualization panel
        panel = draw_match_panel(
            enhanced, tile_gray,
            frame_kps, tile_kps,
            good_matches, inlier_mask, H,
            best_metrics, tile_id, frame_idx_target)

        # Save
        out_path = os.path.join(out_dir, f"match_frame_{frame_idx_target:05d}.jpg")
        cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Saved: {out_path} ({panel.shape[1]}x{panel.shape[0]})")

        # Also save individual components
        # Side-by-side matches only
        frame_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        tile_bgr = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)

        inlier_matches = [m for i, m in enumerate(good_matches) if inlier_mask[i]]
        matches_only = cv2.drawMatches(
            frame_bgr, frame_kps,
            tile_bgr, tile_kps,
            inlier_matches, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matches_path = os.path.join(out_dir, f"matches_frame_{frame_idx_target:05d}.jpg")
        cv2.imwrite(matches_path, matches_only, [cv2.IMWRITE_JPEG_QUALITY, 92])

        return True


def main():
    parser = argparse.ArgumentParser(description="SIFT Match Visualization")
    parser.add_argument("--video", default="DJIG0022.mov", help="Video file")
    parser.add_argument("--map-dir", default="processed_map", help="Processed tile directory")
    parser.add_argument("--frame", type=int, default=None, help="Specific frame to visualize")
    parser.add_argument("--every", type=int, default=30, help="Visualize every N frames")
    parser.add_argument("--max-vis", type=int, default=10, help="Max number of visualizations")
    parser.add_argument("--skip", type=float, default=35.0, help="Skip N seconds from start")
    parser.add_argument("--out", default="vis_matches", help="Output directory")
    parser.add_argument("--roi-px", type=int, default=1100, help="Prior ROI radius")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Config(
        video_path=args.video,
        map_dir=args.map_dir,
        skip_seconds=args.skip,
        save_debug=False,
        roi_px=args.roi_px,
    )

    print("Initializing SIFT matcher...")
    matcher = SIFTMatcher(cfg)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {cfg.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {fps:.1f} fps")

    if args.frame is not None:
        # Single frame mode
        print(f"\nVisualizing frame {args.frame}...")
        visualize_single_frame(matcher, cap, args.frame, args.out, cfg.skip_seconds)
    else:
        # Multi-frame mode
        print(f"\nVisualizing every {args.every} frames (max {args.max_vis})...")

        cap.set(cv2.CAP_PROP_POS_MSEC, cfg.skip_seconds * 1000)
        vis_count = 0
        frame_count = 0

        while vis_count < args.max_vis:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % args.every != 0:
                continue

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if frame.shape[1] > 2000:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            enhanced, roi_mask, edge_count = matcher.preprocess_frame(frame)
            fh, fw = enhanced.shape[:2]
            edge_density = edge_count / max(fh * fw, 1)

            if edge_density < cfg.edge_density_min:
                continue

            frame_kps, frame_descs = matcher.sift.detectAndCompute(enhanced, roi_mask)
            if frame_kps is None or len(frame_kps) < 20:
                continue

            best_metrics, all_verified = matcher.process_frame(
                enhanced, roi_mask, frame_kps, frame_descs)

            if best_metrics is None or best_metrics.get("match_result") is None:
                continue

            match_result = best_metrics["match_result"]
            tile_idx = best_metrics["tile_idx"]
            tile_data = matcher.get_tile_data(tile_idx)
            tile_gray = tile_data["gray"]
            tile_kps = tile_data["keypoints"]
            tile_id = matcher.tiles[tile_idx]["id"]

            good_matches = match_result["matches"]
            inlier_mask = match_result["inlier_mask"]
            H = match_result["H"]

            panel = draw_match_panel(
                enhanced, tile_gray,
                frame_kps, tile_kps,
                good_matches, inlier_mask, H,
                best_metrics, tile_id, frame_count)

            out_path = os.path.join(args.out, f"match_frame_{frame_count:05d}.jpg")
            cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

            status = "PASS" if best_metrics["passed"] else "REJECT"
            print(f"  [{vis_count+1}/{args.max_vis}] F{frame_count} {status} "
                  f"inl={best_metrics['n_inliers']} ncc={best_metrics['ncc']:.3f} "
                  f"-> {out_path}")

            vis_count += 1

        print(f"\nDone! {vis_count} visualizations saved to {args.out}/")

    cap.release()


if __name__ == "__main__":
    main()
