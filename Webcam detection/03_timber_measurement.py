"""
Step 3: Capture and measure timbers on a black background.

Pipeline:
    camera / test image
    -> capture timber (or load test image)
    -> save original image
    -> undistort lens using calibration
    -> detect 4 ArUco markers
    -> calculate live pixel -> mm homography from outer marker corners
    -> define workspace ROI from outer marker corners (masking out markers)
    -> detect and remove the black background / curtain
    -> extract timber as foreground on top of the black curtain
    -> clean timber mask (fill internal grain, remove noise)
    -> extract the exact yellow timber contour
    -> fit oriented bounding rectangle (minAreaRect) matching the yellow contour
    -> map corners to real-world mm
    -> compute length/width and side dimensions
    -> sample LAB color
    -> save masks, debug overlay images, and JSON measurements
    -> wait for next timber

This file is intentionally independent from 3_measure_board.py.
config.py is not modified by this script.

Controls:
    SPACE = capture and measure
    ENTER / ESC = exit
"""

import cv2
import numpy as np
import json
import os
import sys
import argparse
import config
import camera_utils


# ============================================================
# CAMERA UNDISTORTION
# ============================================================

def undistort(img):
    """Remove lens distortion using the saved intrinsic calibration (supports fisheye and standard)."""
    return camera_utils.undistort(img)


# ============================================================
# POINT / GEOMETRY HELPERS
# ============================================================

def polygon_mask(shape, points):
    """Create a filled polygon mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32)
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def order_quad(points):
    """
    Return four points in clockwise image order starting from top-left.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )
    order = np.argsort(angles)
    ordered = pts[order]

    # Rotate so the first point is top-left (minimum x+y sum)
    sums = ordered[:, 0] + ordered[:, 1]
    first = int(np.argmin(sums))
    ordered = np.roll(ordered, -first, axis=0)

    # Ensure clockwise orientation
    area = cv2.contourArea(ordered.reshape(-1, 1, 2))
    if area < 0:
        ordered = ordered[::-1]

    return ordered.astype(np.float32)


def get_contour_oriented_box(contour):
    """
    Fit an oriented bounding rectangle directly to the yellow timber contour.
    This guarantees that the measurement corners and green box tightly wrap
    the exact segmented timber shape.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_quad(box)


# ============================================================
# SUBPIXEL EDGE REFINEMENT
# ============================================================

def refine_corners_subpixel(img_gray, corners):
    """Refine corners using subpixel gradient optimization."""
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01
    )

    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    height, width = img_gray.shape[:2]

    pts[:, 0, 0] = np.clip(pts[:, 0, 0], 1, width - 2)
    pts[:, 0, 1] = np.clip(pts[:, 0, 1], 1, height - 2)

    refined = cv2.cornerSubPix(
        img_gray,
        pts,
        (5, 5),
        (-1, -1),
        criteria
    )

    return order_quad(refined.reshape(-1, 2))


# ============================================================
# BLACK BACKGROUND / TIMBER SEGMENTATION
# ============================================================

def segment_board(img, workspace_outer_px, markers_dict=None, verbose=True):
    """
    Segment timber placed on a black background / curtain inside the ArUco workspace.

    Strategy:
        1. Limit analysis to the full workspace defined by outer ArUco marker corners.
        2. Mask out the ArUco marker square regions so markers do not affect segmentation.
        3. Identify the black background curtain using Otsu and dark intensity filtering.
        4. Extract the curtain domain to isolate the workspace and exclude outer table borders.
        5. Extract timber as foreground inside the curtain domain.
        6. Perform morphological cleaning (fill wood grain, remove specks).
        7. Select the best timber candidate based on area, solidity, and aspect ratio.
        8. Fit oriented bounding rectangle matching the exact yellow timber contour.
    """
    ws_mask = polygon_mask(img.shape, workspace_outer_px)

    # Mask out the 4 ArUco markers with a safety dilation margin
    if markers_dict:
        markers_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for m_id, pts in markers_dict.items():
            pts_int = np.round(pts).astype(np.int32)
            cv2.fillPoly(markers_mask, [pts_int], 255)
        markers_mask = cv2.dilate(markers_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        usable_roi = cv2.bitwise_and(ws_mask, cv2.bitwise_not(markers_mask))
    else:
        usable_roi = ws_mask

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    roi_pixels = gray_blur[usable_roi > 0]
    if roi_pixels.size < 100:
        if verbose:
            print("  ERROR: Usable workspace is too small.")
        return None, np.zeros_like(gray), None

    # Compute dark/background threshold
    otsu_thresh, _ = cv2.threshold(
        roi_pixels.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    black_threshold = int(np.clip(otsu_thresh, 35, 120))

    if verbose:
        print(f"\nBlack-background segmentation:")
        print(f"  Otsu threshold: {otsu_thresh:.1f}")
        print(f"  Applied threshold: {black_threshold}")

    # Binary mask of dark black curtain pixels inside the ROI
    dark_mask = np.zeros_like(gray)
    dark_mask[(gray_blur <= black_threshold) & (usable_roi > 0)] = 255

    # Close small folds/seams in the black fabric
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark_mask_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel)

    # Enclosing domain of the black curtain to isolate the black background workspace
    curtain_cnts, _ = cv2.findContours(dark_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    curtain_domain = np.zeros_like(gray)
    for cnt in curtain_cnts:
        if cv2.contourArea(cnt) > 15000:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(curtain_domain, [hull], -1, 255, thickness=cv2.FILLED)

    # Fallback to usable ROI if curtain domain is small or fragmented
    if cv2.countNonZero(curtain_domain) < 0.25 * cv2.countNonZero(usable_roi):
        curtain_domain = usable_roi.copy()

    # Timber = Curtain Domain minus Dark Background Pixels
    timber_raw = cv2.bitwise_and(curtain_domain, cv2.bitwise_not(dark_mask_closed))
    timber_raw = cv2.bitwise_and(timber_raw, usable_roi)

    # Morphological cleaning: close internal grain / open noise
    t_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    timber_mask = cv2.morphologyEx(timber_raw, cv2.MORPH_CLOSE, t_close_kernel)
    t_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    timber_mask = cv2.morphologyEx(timber_mask, cv2.MORPH_OPEN, t_open_kernel)

    # Find timber candidates
    contours, _ = cv2.findContours(timber_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if verbose:
            print("  ERROR: No foreground contour found.")
        return None, timber_mask, None

    min_area = getattr(config, "MIN_BOARD_CONTOUR_AREA_PX", 3000)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidity = area / hull_area
        if solidity < 0.70:
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        aspect_ratio = max(rw, rh) / max(min(rw, rh), 1.0)
        if aspect_ratio < 1.8:
            continue

        score = area * min(aspect_ratio / 4.0, 2.0) * solidity
        candidates.append((score, contour, area, aspect_ratio, solidity))

    if not candidates:
        if verbose:
            print("  ERROR: No suitable timber contour found.")
        return None, timber_mask, None

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, best_contour, area, aspect_ratio, solidity = candidates[0]

    if verbose:
        print(f"\nTimber candidate:")
        print(f"  Area:         {area:.0f} px")
        print(f"  Aspect ratio: {aspect_ratio:.2f}")
        print(f"  Solidity:     {solidity:.3f}")
        print(f"  Score:        {score:.0f}")

    # Extract tightly fitted bounding box matching the exact yellow contour
    timber_corners_px = get_contour_oriented_box(best_contour)

    return best_contour, timber_mask, timber_corners_px


# ============================================================
# PIXEL -> REAL WORLD MM
# ============================================================

def pixel_to_mm(H, pts_px):
    """Transform image pixel coordinates into world coordinates."""
    pts_px = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    pts_mm = cv2.perspectiveTransform(pts_px, H)
    return pts_mm.reshape(-1, 2)


# ============================================================
# LIVE ARUCO HOMOGRAPHY
# ============================================================

def compute_live_homography(img_undist, verbose=True):
    """
    Detect all 4 ArUco markers and dynamically identify:
    - Outer corners: for high-accuracy pixel -> mm homography and outer workspace
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, config.ARUCO_DICT)
    )
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        if verbose:
            print(" WARNING: no ArUco markers detected.")
        return None, None, None, None, None

    ids = ids.flatten()
    detected_all = {}
    for marker_id, marker_corners in zip(ids, corners):
        detected_all[int(marker_id)] = marker_corners[0]

    expected_ids = set(config.MARKER_WORLD_POSITIONS_MM.keys())
    detected_ids = set(detected_all.keys())

    if not expected_ids.issubset(detected_ids):
        missing = expected_ids - detected_ids
        if verbose:
            print(f" WARNING: markers not all visible (missing {missing})")
        return None, None, None, None, None

    # Compute centroid of all 4 markers
    all_centers = [detected_all[m_id].mean(axis=0) for m_id in expected_ids]
    ws_centroid = np.mean(all_centers, axis=0)

    # Dynamically find outer corners based on distance to workspace centroid
    detected_outer = {}
    detected_inner = {}
    for m_id in expected_ids:
        pts = detected_all[m_id]
        dists = np.linalg.norm(pts - ws_centroid, axis=1)
        detected_inner[m_id] = pts[np.argmin(dists)]
        detected_outer[m_id] = pts[np.argmax(dists)]

    # Map each detected marker's outer corner directly to its configured real-world position in mm
    # e.g. Marker 1 (Bottom-Left Origin) -> (0, 0), Marker 0 (Bottom-Right) -> (776, 0), Marker 2 (Top-Left) -> (0, 440), Marker 3 (Top-Right) -> (776, 440)
    pixel_pts = []
    world_pts = []
    marker_order = [1, 0, 2, 3] if {1, 0, 2, 3}.issubset(expected_ids) else list(expected_ids)

    for marker_id in marker_order:
        pixel_pts.append(detected_outer[marker_id])
        world_pts.append(config.MARKER_WORLD_POSITIONS_MM[marker_id])

    pixel_pts = np.asarray(pixel_pts, dtype=np.float32)
    world_pts = np.asarray(world_pts, dtype=np.float32)

    H, _ = cv2.findHomography(pixel_pts, world_pts, method=0)
    if H is None:
        if verbose:
            print(" WARNING: could not calculate homography.")
        return None, None, None, None, None

    reprojected = cv2.perspectiveTransform(
        pixel_pts.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)
    errors_mm = np.linalg.norm(reprojected - world_pts, axis=1)

    if verbose:
        print("\nArUco marker positions in world coordinates:")
        for marker_id, point in zip(marker_order, reprojected):
            print(f"  Marker {marker_id}: X={point[0]:.2f} mm, Y={point[1]:.2f} mm")
        print(f"\n  Maximum marker reprojection error: {errors_mm.max():.4f} mm")

    return H, errors_mm, detected_outer, detected_inner, detected_all

# ============================================================
# COLOR SAMPLING
# ============================================================

def sample_timber_color(img_bgr, contour):
    """Calculate average LAB, RGB, and HEX color from inside the timber."""
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    if cv2.countNonZero(mask) > 400:
        eroded = cv2.erode(mask, np.ones((7, 7), np.uint8))
        if cv2.countNonZero(eroded) > 100:
            mask = eroded

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    mean_lab = cv2.mean(lab, mask=mask)[:3]

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mean_rgb = cv2.mean(rgb, mask=mask)[:3]
    r, g, b = int(round(mean_rgb[0])), int(round(mean_rgb[1])), int(round(mean_rgb[2]))

    return {
        "lab": {
            "L": round(float(mean_lab[0]), 2),
            "a": round(float(mean_lab[1]), 2),
            "b": round(float(mean_lab[2]), 2)
        },
        "rgb": {
            "R": r,
            "G": g,
            "B": b
        },
        "hex": f"#{r:02x}{g:02x}{b:02x}"
    }


# ============================================================
# MEASURE ONE TIMBER
# ============================================================

def measure_timber(img, image_path, timber_number, timber_thickness_mm=0.0):
    print("\n" + "=" * 60)
    print(f"MEASURING TIMBER {timber_number:02d}  (thickness: {timber_thickness_mm:.1f} mm)")
    print("=" * 60)

    img_undist = undistort(img)

    (
        H,
        errors_mm,
        detected_outer,
        detected_inner,
        detected_all
    ) = compute_live_homography(img_undist)

    if H is None:
        print("\nERROR: Could not detect all four ArUco markers.")
        return False

    max_err = float(errors_mm.max())
    print(f"\nLive homography OK. Maximum marker error: {max_err:.2f} mm")
    if max_err > 2.0:
        print(f"WARNING: marker error is high ({max_err:.2f} mm).")

    # Physical workspace order: TL (2) -> TR (3) -> BR (0) -> BL (1)
    workspace_outer_px = order_quad(
        np.array([detected_outer[m] for m in [0, 1, 2, 3]], dtype=np.float32)
    )

    contour, mask, timber_corners_px = segment_board(
        img_undist,
        workspace_outer_px,
        detected_all
    )

    base_name = f"timber_{timber_number:02d}"
    os.makedirs(config.CAPTURE_DIR, exist_ok=True)
    mask_path = os.path.join(config.CAPTURE_DIR, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, mask)

    if contour is None or timber_corners_px is None:
        print("\nERROR: No timber detected.")
        print(f"Mask saved to: {mask_path}")
        return False

    # Pixel -> mm mapping for corners
    corners_mm_table = pixel_to_mm(H, timber_corners_px)

    # Pixel -> mm mapping for contour with contour approximation (OpenCV approxPolyDP)
    # Epsilon = factor * arcLength (OpenCV tutorial standard: https://docs.opencv.org/4.13.0/dd/d49/tutorial_py_contour_features.html)
    peri = cv2.arcLength(contour, True)
    approx_factor = float(getattr(config, "CONTOUR_APPROX_FACTOR", 0.001))
    epsilon = approx_factor * peri
    approx_contour_px = cv2.approxPolyDP(contour, epsilon, True)
    approx_cnt = approx_contour_px.reshape(-1, 2)
    contour_mm_table = pixel_to_mm(H, approx_cnt)

    print(f"\nContour Approximation:")
    print(f"  Raw contour points:       {len(contour)}")
    print(f"  Approx factor (epsilon):  {approx_factor} ({epsilon:.2f} px)")
    print(f"  Approximated points:      {len(approx_contour_px)}")

    # Parallax compensation
    H_cam = float(getattr(config, "CAMERA_HEIGHT_MM", 0.0))
    h_timber = float(timber_thickness_mm)

    if H_cam > 0 and h_timber > 0 and H_cam > h_timber and os.path.exists(config.CAMERA_CALIB_FILE):
        scale_factor = (H_cam - h_timber) / H_cam
        calib = np.load(config.CAMERA_CALIB_FILE)
        K = calib["camera_matrix"]
        cx, cy = K[0, 2], K[1, 2]
        cam_center_table = pixel_to_mm(H, [[cx, cy]])[0]
        Xc, Yc = cam_center_table[0], cam_center_table[1]

        corners_mm = []
        for pt in corners_mm_table:
            X_corr = Xc + (pt[0] - Xc) * scale_factor
            Y_corr = Yc + (pt[1] - Yc) * scale_factor
            corners_mm.append([X_corr, Y_corr])
        corners_mm = np.asarray(corners_mm, dtype=np.float32)

        contour_mm = []
        for pt in contour_mm_table:
            X_corr = Xc + (pt[0] - Xc) * scale_factor
            Y_corr = Yc + (pt[1] - Yc) * scale_factor
            contour_mm.append([X_corr, Y_corr])
        contour_mm = np.asarray(contour_mm, dtype=np.float32)
    else:
        corners_mm = corners_mm_table
        contour_mm = contour_mm_table

    # Surface Area calculation in real-world mm²
    surface_area_mm2 = abs(cv2.contourArea(contour_mm.astype(np.float32).reshape(-1, 1, 2)))
    surface_area_cm2 = surface_area_mm2 / 100.0

    # Dimensions
    side_lengths = [
        float(np.linalg.norm(corners_mm[(i + 1) % 4] - corners_mm[i]))
        for i in range(4)
    ]
    dim_a = (side_lengths[0] + side_lengths[2]) / 2.0
    dim_b = (side_lengths[1] + side_lengths[3]) / 2.0
    length_mm = max(dim_a, dim_b)
    width_mm = min(dim_a, dim_b)

    # Sample color (LAB + RGB + HEX)
    color_info = sample_timber_color(img_undist, contour)

    # Calculate detected ArUco marker centers and corners in real-world mm
    marker_order = [1, 0, 2, 3] if {1, 0, 2, 3}.issubset(detected_all.keys()) else list(detected_all.keys())
    markers_info = {}
    for m_id in marker_order:
        m_corners_px = detected_all[m_id]
        m_corners_mm = pixel_to_mm(H, m_corners_px)
        m_center_px = m_corners_px.mean(axis=0)
        m_center_mm = pixel_to_mm(H, [m_center_px])[0]
        m_outer_mm = pixel_to_mm(H, [detected_outer[m_id]])[0]
        markers_info[str(m_id)] = {
            "outer_corner_mm": [round(float(m_outer_mm[0]), 2), round(float(m_outer_mm[1]), 2)],
            "center_mm": [round(float(m_center_mm[0]), 2), round(float(m_center_mm[1]), 2)],
            "corners_mm": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in m_corners_mm]
        }

    # Reference Frame definition based on Marker 1 as Origin (0,0,0) (CAD / Rhino standard)
    # +X points along table length towards Marker 0
    # +Y points across table width towards Marker 2
    # +Z points normal to table upwards
    reference_frame = {
        "origin_marker_id": 1,
        "origin_mm": [0.0, 0.0, 0.0],
        "x_axis": [1.0, 0.0, 0.0],
        "y_axis": [0.0, 1.0, 0.0],
        "z_axis": [0.0, 0.0, 1.0],
        "description": "Origin (0,0,0) at Marker 1 outer corner (Bottom-Left); +X towards Marker 0; +Y towards Marker 2"
    }

    # Build JSON result
    result = {
        "timber_id": timber_number,
        "source_image": os.path.abspath(image_path),
        "reference_frame": reference_frame,
        "markers_world_mm": markers_info,
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "surface_area_mm2": round(float(surface_area_mm2), 2),
        "surface_area_cm2": round(float(surface_area_cm2), 2),
        "side_lengths_mm": [round(s, 2) for s in side_lengths],
        "corners_mm": [
            [round(float(x), 2), round(float(y), 2)]
            for x, y in corners_mm
        ],
        "contour_mm": [
            [round(float(x), 2), round(float(y), 2)]
            for x, y in contour_mm
        ],
        "timber_thickness_mm": round(timber_thickness_mm, 1),
        "color_rgb": color_info["rgb"],
        "color_hex": color_info["hex"],
        "color_lab": color_info["lab"],
        "defects": []
    }

    out_path = os.path.join(config.CAPTURE_DIR, f"{base_name}_measurement.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Debug Overlay
    overlay = img_undist.copy()

    # Outer workspace (Red)
    cv2.polylines(overlay, [np.int32(workspace_outer_px)], True, (0, 0, 255), 3)

    # ArUco markers (Magenta)
    for m_id in [0, 1, 2, 3]:
        if m_id in detected_all:
            pts = detected_all[m_id]
            cv2.polylines(overlay, [np.int32(pts)], True, (255, 0, 255), 2)
            c = pts.mean(axis=0)
            cv2.putText(
                overlay, f"ID {m_id}", (int(c[0]), int(c[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
            )

    # Exact Timber contour (Yellow, smoothed approxPolyDP)
    cv2.drawContours(overlay, [approx_contour_px], -1, (0, 255, 255), 3)

    # Fitted Timber bounding box (Green) matching the yellow contour
    cv2.polylines(overlay, [np.int32(timber_corners_px)], True, (0, 255, 0), 3)

    for i, pt in enumerate(timber_corners_px):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(overlay, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(overlay, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Labels
    cv2.putText(overlay, "RED: workspace | YELLOW: timber contour | GREEN: measured box", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Dim: {length_mm:.1f} x {width_mm:.1f} mm  |  Area: {surface_area_cm2:.1f} cm2  (Thick: {timber_thickness_mm:.1f} mm)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    overlay_path = os.path.join(config.CAPTURE_DIR, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    print("\n" + "-" * 60)
    print(f"TIMBER {timber_number:02d} MEASUREMENT")
    print("-" * 60)
    print(f"Length:       {length_mm:.2f} mm")
    print(f"Width:        {width_mm:.2f} mm")
    print(f"Surface Area: {surface_area_mm2:.1f} mm² ({surface_area_cm2:.2f} cm²)")
    print(f"Thickness:    {timber_thickness_mm:.1f} mm (parallax input)")
    print(f"Sides:        {[round(s, 2) for s in side_lengths]} mm")
    print(f"Color RGB:    R={color_info['rgb']['R']}, G={color_info['rgb']['G']}, B={color_info['rgb']['B']} ({color_info['hex']})")
    print(f"Color LAB:    L={color_info['lab']['L']:.2f}, a={color_info['lab']['a']:.2f}, b={color_info['lab']['b']:.2f}")
    print(f"\nSaved:")
    print(f"  JSON:       {out_path}")
    print(f"  Mask:       {mask_path}")
    print(f"  Overlay:    {overlay_path}")

    return True


# ============================================================
# REAL-TIME LIVE VIEWPORT OVERLAY
# ============================================================

def render_live_viewport(img_undist, timber_number, current_thickness):
    """
    Run real-time detection on the live camera frame and return an annotated viewport display:
    - Red outline for detected ArUco workspace
    - Magenta boxes and IDs for ArUco markers
    - Smooth Yellow contour (approxPolyDP) for detected timber
    - Green bounding box and corner dots
    - Real-time dimension readout (Length x Width mm) and status overlay
    """
    display = img_undist.copy()

    H, errors_mm, detected_outer, detected_inner, detected_all = compute_live_homography(img_undist, verbose=False)

    timber_detected = False
    status_text = "Place timber inside black workspace"
    dim_text = ""

    if H is not None and detected_outer is not None:
        # Physical workspace order: BL (1) -> BR (0) -> TR (3) -> TL (2)
        workspace_outer_px = order_quad(
            np.array([detected_outer[m] for m in [1, 0, 2, 3] if m in detected_outer], dtype=np.float32)
        )

        # Draw outer workspace boundary (Red)
        cv2.polylines(display, [np.int32(workspace_outer_px)], True, (0, 0, 255), 2)

        # Draw ArUco markers (Magenta)
        for m_id in [1, 0, 2, 3]:
            if m_id in detected_all:
                pts = detected_all[m_id]
                cv2.polylines(display, [np.int32(pts)], True, (255, 0, 255), 2)
                c = pts.mean(axis=0)
                cv2.putText(
                    display, f"ID {m_id}", (int(c[0]) - 20, int(c[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
                )

        # Run fast live timber segmentation
        contour, mask, timber_corners_px = segment_board(
            img_undist, workspace_outer_px, detected_all, verbose=False
        )

        if contour is not None and timber_corners_px is not None:
            timber_detected = True

            # Contour approximation (Douglas-Peucker algorithm via cv2.approxPolyDP)
            peri = cv2.arcLength(contour, True)
            approx_factor = float(getattr(config, "CONTOUR_APPROX_FACTOR", 0.001))
            epsilon = approx_factor * peri
            approx_contour_px = cv2.approxPolyDP(contour, epsilon, True)

            # Draw Yellow smoothed contour
            cv2.drawContours(display, [approx_contour_px], -1, (0, 255, 255), 2)

            # Draw Green bounding box
            cv2.polylines(display, [np.int32(timber_corners_px)], True, (0, 255, 0), 2)

            for i, pt in enumerate(timber_corners_px):
                x, y = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)

            # Live dimensions with parallax compensation
            corners_mm_raw = pixel_to_mm(H, timber_corners_px)

            # Apply parallax correction (same formula as measure_timber)
            H_cam = float(getattr(config, "CAMERA_HEIGHT_MM", 0.0))
            h_timber = float(current_thickness)
            if H_cam > 0 and h_timber > 0 and H_cam > h_timber and os.path.exists(config.CAMERA_CALIB_FILE):
                scale_factor = (H_cam - h_timber) / H_cam
                calib = np.load(config.CAMERA_CALIB_FILE)
                K = calib["camera_matrix"]
                cx_cam, cy_cam = K[0, 2], K[1, 2]
                cam_center_table = pixel_to_mm(H, [[cx_cam, cy_cam]])[0]
                Xc, Yc = cam_center_table[0], cam_center_table[1]
                corners_mm = np.array([
                    [Xc + (pt[0] - Xc) * scale_factor, Yc + (pt[1] - Yc) * scale_factor]
                    for pt in corners_mm_raw
                ], dtype=np.float32)
            else:
                corners_mm = corners_mm_raw

            side_lengths = [
                float(np.linalg.norm(corners_mm[(i + 1) % 4] - corners_mm[i]))
                for i in range(4)
            ]
            dim_a = (side_lengths[0] + side_lengths[2]) / 2.0
            dim_b = (side_lengths[1] + side_lengths[3]) / 2.0
            length_mm = max(dim_a, dim_b)
            width_mm = min(dim_a, dim_b)
            thick_label = f"  T={current_thickness:.0f}mm" if current_thickness > 0 else ""
            dim_text = f"Live: {length_mm:.1f} x {width_mm:.1f} mm ({len(approx_contour_px)} pts){thick_label}"
            status_text = "Timber detected! Press SPACE to capture"

            # Label on timber centroid
            cx = int(timber_corners_px[:, 0].mean())
            cy = int(timber_corners_px[:, 1].mean())
            cv2.putText(display, f"{length_mm:.1f} x {width_mm:.1f} mm", (cx - 70, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        status_text = "Looking for 4 ArUco markers..."

    # Top HUD Bar
    h, w = display.shape[:2]
    header = np.zeros((85, w, 3), dtype=np.uint8)
    cv2.putText(header, f"TIMBER {timber_number:02d}  |  {dim_text if dim_text else status_text}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0) if timber_detected else (0, 200, 255), 2)
    cv2.putText(header, f"SPACE = Capture & Save JSON  |  T = Thickness ({current_thickness:.1f} mm)  |  ENTER/ESC = Exit", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    # Blend header onto display
    display[:85] = cv2.addWeighted(display[:85], 0.25, header, 0.75, 0)

    return display


# ============================================================
# FIND NEXT TIMBER NUMBER
# ============================================================

def get_next_timber_number():
    """Find the next available timber number."""
    os.makedirs(config.CAPTURE_DIR, exist_ok=True)
    number = 1
    while True:
        json_path = os.path.join(config.CAPTURE_DIR, f"timber_{number:02d}_measurement.json")
        if not os.path.exists(json_path):
            return number
        number += 1


# ============================================================
# MAIN CAMERA LOOP / IMAGE PROCESSING
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Measure timber on black background.")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image file to process directly.")
    parser.add_argument("--number", type=int, default=None, help="Timber ID number when processing a single image.")
    parser.add_argument("--thickness", type=float, default=None,
                        help="Timber thickness in mm for parallax compensation. "
                             "If not specified, uses TIMBER_THICKNESS_MM from config.py.")
    args = parser.parse_args()

    # Determine initial timber thickness
    if args.thickness is not None:
        current_thickness = args.thickness
    else:
        current_thickness = float(getattr(config, "TIMBER_THICKNESS_MM", 0.0))

    os.makedirs(config.CAPTURE_DIR, exist_ok=True)

    if not os.path.exists(config.CAMERA_CALIB_FILE):
        print(f"ERROR: Missing {config.CAMERA_CALIB_FILE}")
        print("Run 1_calibrate_camera.py first.")
        raise SystemExit(1)

    # Direct image processing mode
    if args.image is not None:
        if not os.path.exists(args.image):
            print(f"ERROR: Image file not found: {args.image}")
            raise SystemExit(1)

        img = cv2.imread(args.image)
        if img is None:
            print(f"ERROR: Could not read image: {args.image}")
            raise SystemExit(1)

        timber_num = args.number if args.number is not None else 1
        print(f"Processing static image: {args.image} as timber {timber_num:02d}...")
        print(f"Timber thickness for parallax: {current_thickness:.1f} mm")
        success = measure_timber(img, args.image, timber_num, current_thickness)
        if success:
            print("\nMeasurement finished successfully.")
        else:
            
            print("\nMeasurement failed.")
            raise SystemExit(1)
        return

    # Live Camera mode
    cap, controller = camera_utils.open_configured_camera()

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("\n" + "=" * 60)
    print("TIMBER MEASUREMENT CAMERA - BLACK BACKGROUND")
    print("=" * 60)
    print(f"\nCamera opened successfully at resolution: {actual_w:.0f}x{actual_h:.0f}")
    print("Place ONE timber inside the black workspace.")
    print("Live contour, detection box, and dimensions are displayed in real-time.")
    print("SPACE = capture and measure")
    print("T     = change timber thickness (for parallax correction)")
    print("ENTER / ESC = exit")
    print(f"\nCurrent timber thickness: {current_thickness:.1f} mm")
    print(f"Camera height: {getattr(config, 'CAMERA_HEIGHT_MM', 0.0):.0f} mm")
    print(f"Output folder: {config.CAPTURE_DIR}")

    timber_number = get_next_timber_number()

    window_name = "Timber Measurement - Live View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 850)

    zoom_factor = float(getattr(config, "CAMERA_ZOOM", getattr(config, "CAMERA_ZOOM_VALUE", 1.0)))

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("ERROR: Failed to read camera frame.")
                break

            frame = camera_utils.apply_digital_zoom(raw_frame, zoom_factor)
            img_undist = undistort(frame)
            display = render_live_viewport(img_undist, timber_number, current_thickness)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('t') or key == ord('T'):  # T = change thickness
                print(f"\nCurrent timber thickness: {current_thickness:.1f} mm")
                try:
                    new_val = input("Enter new timber thickness in mm (e.g. 20, 35, 50): ").strip()
                    current_thickness = float(new_val)
                    print(f"Timber thickness set to: {current_thickness:.1f} mm")
                except (ValueError, EOFError):
                    print("Invalid input. Thickness unchanged.")

            elif key == 32:  # SPACE
                print(f"\nCapturing timber {timber_number:02d} (thickness: {current_thickness:.1f} mm)...")
                image_path = os.path.join(config.CAPTURE_DIR, f"timber_{timber_number:02d}.jpg")
                cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                print(f"Captured: {image_path}")

                success = measure_timber(frame, image_path, timber_number, current_thickness)
                if success:
                    print("\n" + "=" * 60)
                    print(f"TIMBER {timber_number:02d} COMPLETE")
                    print("=" * 60)
                    timber_number += 1
                    print(f"\nPlace next timber. Press SPACE to measure timber {timber_number:02d}.")
                    print(f"Press T to change thickness if the next timber is different.")
                else:
                    print("\nMeasurement failed. Timber number not increased. Adjust and press SPACE.")

            elif key in (10, 13, 27):  # ENTER or ESC
                print("\nExiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass
        print("\nMeasurement session finished.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
