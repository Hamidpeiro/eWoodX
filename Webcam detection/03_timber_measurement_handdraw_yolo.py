"""
Step 3: Measure multiple timbers and detect hand-drawn shapes with trained YOLO.

Pipeline:
    camera / test image
    -> capture timber (or load test image)
    -> save original image
    -> undistort lens using calibration
    -> detect 4 ArUco markers
    -> calculate live pixel -> mm homography from outer marker corners
    -> define workspace ROI from outer marker corners (masking out markers)
    -> detect thickness reference plate (color-coded circle -> auto thickness)
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
    T     = set manual timber thickness
    [ / ] = decrease / increase YOLO confidence
    ENTER / ESC = exit

Thickness Detection:
    Place a colored circular plate (100mm diameter) anywhere visible
    in the workspace. The system auto-detects the plate color and
    sets thickness accordingly:
        Red    = 10mm    Blue   = 20mm    Green  = 30mm
        Yellow = 40mm    Orange = 50mm
    Remove the plate to revert to manual thickness (T key).
"""

import cv2
import numpy as np
import json
import os
import sys
import socket
import threading
import queue
import argparse
import config
import camera_utils
from handdraw_detector import (
    HanddrawDetector, DEFAULT_MODEL, project_detections, draw_detections,
)

HANDDRAW_DETECTOR = None


# ============================================================
# GRASSHOPPER REMOTE CAPTURE SERVER (TCP PORT 9998)
# ============================================================

class GrasshopperCaptureServer:
    """
    Lightweight background TCP listener allowing Grasshopper to remotely trigger
    timber capture & measurement on demand while the live camera loop is active.
    Reserved timber ID 0 is dedicated for Grasshopper updates.
    """
    def __init__(self, host="127.0.0.1", port=9998):
        self.host = host
        self.port = port
        self.running = False
        self.server_sock = None
        self.thread = None
        self.request_queue = queue.Queue()

    def start(self):
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind((self.host, self.port))
            self.server_sock.listen(2)
            self.server_sock.settimeout(0.5)
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            reserved_num = getattr(config, "GRASSHOPPER_RESERVED_TIMBER_NUMBER", 0)
            print(f"[Grasshopper] Remote capture listener active on {self.host}:{self.port}")
            print(f"[Grasshopper] Reserved timber #{reserved_num:02d} specifically for Grasshopper.")
        except Exception as e:
            print(f"[Grasshopper] Warning: Could not start remote capture listener on port {self.port}: {e}")

    def _listen_loop(self):
        while self.running:
            try:
                client, addr = self.server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            try:
                client.settimeout(15.0)
                data = client.recv(1024)
                if not data:
                    client.close()
                    continue
                text = data.decode("utf-8", errors="ignore").strip()
                print(f"\n[Grasshopper] Incoming remote capture request from {addr[0]}: '{text}'")
                requested_number = getattr(config, "GRASSHOPPER_RESERVED_TIMBER_NUMBER", 0)
                try:
                    payload = json.loads(text)
                    requested_number = int(payload.get("number", requested_number))
                except Exception:
                    parts = text.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        requested_number = int(parts[1])

                event = threading.Event()
                req = {
                    "number": requested_number,
                    "event": event,
                    "client": client,
                    "result": None
                }
                self.request_queue.put(req)

                # Wait for main camera loop to capture and measure the frame
                if event.wait(timeout=15.0):
                    res = req.get("result", {})
                    response_bytes = (json.dumps(res) + "\n").encode("utf-8")
                    try:
                        client.sendall(response_bytes)
                        print(f"[Grasshopper] Sent response back to {addr[0]}: status={res.get('status')}")
                    except Exception as se:
                        print(f"[Grasshopper] Error sending response: {se}")
                else:
                    err_res = {"status": "error", "message": "Capture timeout in camera loop"}
                    try:
                        client.sendall((json.dumps(err_res) + "\n").encode("utf-8"))
                    except Exception:
                        pass

            except Exception:
                pass
            finally:
                try:
                    client.close()
                except Exception:
                    pass

    def get_pending_request(self):
        try:
            return self.request_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass


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


def smooth_contour_spline(contour, num_points=120):
    """
    Fit a closed cubic B-spline through *contour* points and resample
    into *num_points* evenly-spaced points for a smooth organic curve.

    Parameters
    ----------
    contour : ndarray
        OpenCV contour (N, 1, 2) or (N, 2) int/float array.
    num_points : int
        Number of output sample points on the smoothed curve.

    Returns
    -------
    smooth_contour : ndarray
        (num_points, 1, 2) int32 array in OpenCV contour format.
    """
    from scipy.interpolate import splprep, splev

    pts = contour.reshape(-1, 2).astype(np.float64)
    
    if len(pts) < 4:
        # Not enough points to fit a cubic spline; return as-is
        return contour

    # Close the contour by repeating the first point
    pts_closed = np.vstack([pts, pts[0]])
    x = pts_closed[:, 0]
    y = pts_closed[:, 1]

    try:
        # s (smoothing factor): 0 = interpolate exactly, > 0 = allow deviation.
        # A small positive s absorbs pixel noise while staying faithful.
        tck, _ = splprep([x, y], s=len(pts) * 0.5, per=True, k=3)
        u_new = np.linspace(0, 1, num_points, endpoint=False)
        x_new, y_new = splev(u_new, tck)
        smooth = np.stack([x_new, y_new], axis=-1)
        return smooth.astype(np.int32).reshape(-1, 1, 2)
    except Exception:
        # Fallback: if spline fitting fails, return the original
        return contour


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

def segment_board(img, workspace_outer_px, markers_dict=None, verbose=True, plate_info=None):
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
        return [], np.zeros_like(gray)

    # Compute dark/background threshold
    otsu_thresh, _ = cv2.threshold(
        roi_pixels.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    black_threshold = int(np.clip(otsu_thresh * 0.65, 30, 75))

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

    # Timber = pixels inside usable ROI brighter than black background threshold
    timber_raw = np.zeros_like(gray)
    timber_raw[(gray_blur > black_threshold) & (usable_roi > 0)] = 255

    # Morphological cleaning: close internal grain / open noise
    t_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    timber_mask = cv2.morphologyEx(timber_raw, cv2.MORPH_CLOSE, t_close_kernel)
    t_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    timber_mask = cv2.morphologyEx(timber_mask, cv2.MORPH_OPEN, t_open_kernel)

    # Find timber candidates including inner cuts (holes) using RETR_CCOMP
    contours, hierarchy = cv2.findContours(timber_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        if verbose:
            print("  ERROR: No foreground contour found.")
        return [], timber_mask

    min_area = getattr(config, "MIN_BOARD_CONTOUR_AREA_PX", 3000)
    min_hole_area = getattr(config, "MIN_TIMBER_HOLE_AREA_PX", 500)
    candidates = []

    for i, contour in enumerate(contours):
        # In RETR_CCOMP, hierarchy is [Next, Previous, First_Child, Parent]
        # Only top-level outer contours have parent == -1
        if hierarchy[0][i][3] != -1:
            continue

        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidity = area / hull_area
        if solidity < 0.50:
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        aspect_ratio = max(rw, rh) / max(min(rw, rh), 1.0)
        if aspect_ratio < 1.0:
            continue

        # Ignore candidate if it matches the detected thickness plate sitting on the background
        if plate_info is not None:
            cx, cy = plate_info["center_px"]
            pr = plate_info["radius_px"]
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cnt_cx = M["m10"] / M["m00"]
                cnt_cy = M["m01"] / M["m00"]
                dist = np.hypot(cnt_cx - cx, cnt_cy - cy)
                plate_area = np.pi * (pr ** 2)
                if dist < pr * 0.85 and (0.25 * plate_area < area < 2.5 * plate_area) and aspect_ratio < 1.5:
                    if verbose:
                        print(f"  Ignoring candidate contour at ({cnt_cx:.0f}, {cnt_cy:.0f}): matches thickness plate ({plate_info['color_name']})")
                    continue

        # Extract all child contours (inner cuts / holes) for this timber
        holes = []
        child_idx = hierarchy[0][i][2]
        while child_idx != -1:
            h_cnt = contours[child_idx]
            h_area = cv2.contourArea(h_cnt)
            if h_area >= min_hole_area:
                # If hole overlaps with the thickness plate on top of timber, ignore it
                is_plate_hole = False
                if plate_info is not None:
                    cx, cy = plate_info["center_px"]
                    pr = plate_info["radius_px"]
                    h_M = cv2.moments(h_cnt)
                    if h_M["m00"] > 0:
                        h_cx = h_M["m10"] / h_M["m00"]
                        h_cy = h_M["m01"] / h_M["m00"]
                        if np.hypot(h_cx - cx, h_cy - cy) < pr:
                            is_plate_hole = True
                if not is_plate_hole:
                    holes.append(h_cnt)
            child_idx = hierarchy[0][child_idx][0]  # Next sibling hole

        # Create individual binary mask for this timber (timber wood = 255, holes & background = 0)
        timber_individual_mask = np.zeros(timber_mask.shape, dtype=np.uint8)
        cv2.drawContours(timber_individual_mask, [contour], -1, 255, thickness=cv2.FILLED)
        if holes:
            cv2.drawContours(timber_individual_mask, holes, -1, 0, thickness=cv2.FILLED)

        net_area = area - sum(cv2.contourArea(h) for h in holes)

        score = area * min(aspect_ratio / 4.0, 2.0) * solidity
        candidates.append((score, contour, area, aspect_ratio, solidity, holes, timber_individual_mask, net_area))

    if not candidates:
        if verbose:
            print("  ERROR: No suitable timber contour found.")
        return [], timber_mask

    candidates.sort(key=lambda item: item[0], reverse=True)
    detected_timbers = []

    for score, contour, area, aspect_ratio, solidity, holes, timber_individual_mask, net_area in candidates:

        timber_corners_px = get_contour_oriented_box(contour)

        detected_timbers.append({
            "contour": contour,
            "corners_px": timber_corners_px,
            "holes": holes,
            "mask": timber_individual_mask,
            "area": area,
            "net_area": net_area,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "score": score
        })

    return detected_timbers, timber_mask


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

def sample_timber_color(img_bgr, contour, holes=None, timber_mask=None, plate_info=None):
    """Calculate average LAB, RGB, and HEX color from inside the timber (excluding inner cuts/holes and plate)."""
    if timber_mask is not None:
        mask = timber_mask.copy()
    else:
        mask = np.zeros(img_bgr.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        if holes:
            cv2.drawContours(mask, holes, -1, 0, thickness=cv2.FILLED)

    # Exclude plate region from color sampling if plate is on the timber
    if plate_info is not None:
        cx, cy = plate_info["center_px"]
        r = int(round(plate_info["radius_px"] * 1.15))
        cv2.circle(mask, (cx, cy), r, 0, thickness=cv2.FILLED)

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
# THICKNESS REFERENCE PLATE DETECTION
# ============================================================

def detect_thickness_plate(img, H=None, workspace_mask=None, verbose=False):
    """
    Detect a colored circular reference plate in the image and return
    the corresponding thickness.

    The plate is a physical circle of known diameter (config.THICKNESS_PLATE_DIAMETER_MM)
    painted in a specific color.  Each color maps to a known timber thickness.

    Detection pipeline:
        1. Convert image to HSV.
        2. For each configured plate color, apply HSV threshold(s) to isolate
           candidate pixels.
        3. Apply workspace mask (only look inside the ArUco workspace).
        4. Find contours; filter by area and circularity.
        5. If a homography is available, validate that the detected circle's
           diameter in mm matches the expected plate diameter (within tolerance).
        6. Return the best (largest, most circular) detection.

    Parameters
    ----------
    img : ndarray
        Undistorted BGR image.
    H : ndarray or None
        3x3 pixel-to-mm homography.  When available, the function validates
        the physical size of detected circles.
    workspace_mask : ndarray or None
        Binary mask (255 inside workspace).  If None, the whole image is used.
    verbose : bool
        Print debug information.

    Returns
    -------
    dict or None
        {
            "color_name": str,          # e.g. "red"
            "thickness_mm": float,      # e.g. 10.0
            "center_px": (int, int),    # pixel center
            "radius_px": int,           # pixel radius
            "diameter_mm": float,       # measured real-world diameter
            "display_color_bgr": tuple, # BGR color for drawing
            "circularity": float,       # 0..1
        }
        or None if no valid plate is found.
    """
    plates_cfg = getattr(config, "THICKNESS_PLATES", {})
    if not plates_cfg:
        return None

    expected_diameter_mm = float(getattr(config, "THICKNESS_PLATE_DIAMETER_MM", 100.0))
    min_circularity = float(getattr(config, "THICKNESS_PLATE_MIN_CIRCULARITY", 0.65))
    diameter_tolerance = float(getattr(config, "THICKNESS_PLATE_DIAMETER_TOLERANCE", 0.35))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    best = None
    best_score = -1.0

    for color_name, plate_info in plates_cfg.items():
        thickness_mm = float(plate_info["thickness_mm"])
        hsv_ranges = plate_info.get("hsv_ranges", [])
        display_color = plate_info.get("display_color_bgr", (255, 255, 255))

        # Build combined mask for all HSV ranges of this color
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for hrange in hsv_ranges:
            lower = np.array(hrange["lower"], dtype=np.uint8)
            upper = np.array(hrange["upper"], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Restrict to workspace area
        if workspace_mask is not None:
            combined_mask = cv2.bitwise_and(combined_mask, workspace_mask)

        # Morphological cleaning to remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Minimum area filter: at least 500 px² to avoid noise
            if area < 500:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            # Circularity: 4π·area / perimeter²  (1.0 = perfect circle)
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < min_circularity:
                continue

            # Get enclosing circle
            (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)
            cx, cy, radius_px = int(round(cx)), int(round(cy)), int(round(radius_px))

            if radius_px < 5:
                continue

            # If homography is available, validate physical diameter
            diameter_mm = 0.0
            if H is not None:
                # Measure diameter by projecting two points across the circle
                pt_left = np.array([[cx - radius_px, cy]], dtype=np.float32).reshape(-1, 1, 2)
                pt_right = np.array([[cx + radius_px, cy]], dtype=np.float32).reshape(-1, 1, 2)
                mm_left = cv2.perspectiveTransform(pt_left, H).reshape(-1, 2)[0]
                mm_right = cv2.perspectiveTransform(pt_right, H).reshape(-1, 2)[0]
                diameter_mm = float(np.linalg.norm(mm_right - mm_left))

                # Validate diameter is within tolerance
                lower_limit = expected_diameter_mm * (1.0 - diameter_tolerance)
                upper_limit = expected_diameter_mm * (1.0 + diameter_tolerance)
                if not (lower_limit <= diameter_mm <= upper_limit):
                    if verbose:
                        print(f"  Plate rejected ({color_name}): diameter {diameter_mm:.1f}mm "
                              f"outside [{lower_limit:.1f}, {upper_limit:.1f}]mm")
                    continue

            score = area * circularity
            if score > best_score:
                best_score = score
                # Create dedicated binary mask for the plate
                plate_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(plate_mask, [cnt], -1, 255, thickness=cv2.FILLED)

                best = {
                    "color_name": color_name,
                    "thickness_mm": thickness_mm,
                    "center_px": (cx, cy),
                    "radius_px": radius_px,
                    "diameter_mm": round(diameter_mm, 1),
                    "display_color_bgr": display_color,
                    "circularity": round(circularity, 3),
                    "mask": plate_mask,
                }

    if best is not None and verbose:
        print(f"  Plate detected: {best['color_name']} → {best['thickness_mm']:.0f}mm "
              f"(⌀{best['diameter_mm']:.1f}mm, circularity {best['circularity']:.2f})")

    return best


def draw_plate_overlay(display, plate_info, active=True):
    """
    Draw a visual indicator on the display for a detected thickness plate.

    Parameters
    ----------
    display : ndarray
        Image to draw on (modified in-place).
    plate_info : dict
        Detection result from detect_thickness_plate().
    active : bool
        If True, draws with a solid indicator (plate thickness is in use).
    """
    if plate_info is None:
        return

    cx, cy = plate_info["center_px"]
    radius = plate_info["radius_px"]
    color = plate_info["display_color_bgr"]
    thickness_mm = plate_info["thickness_mm"]
    color_name = plate_info["color_name"]
    diameter_mm = plate_info["diameter_mm"]

    # Draw circle outline around the detected plate
    thickness_line = 4 if active else 2
    cv2.circle(display, (cx, cy), radius, color, thickness_line, cv2.LINE_AA)

    # Draw crosshair at center
    cross_size = max(8, radius // 4)
    cv2.line(display, (cx - cross_size, cy), (cx + cross_size, cy), color, 2, cv2.LINE_AA)
    cv2.line(display, (cx, cy - cross_size), (cx, cy + cross_size), color, 2, cv2.LINE_AA)

    # Label badge: color name + thickness
    label = f"{color_name.upper()} PLATE: {thickness_mm:.0f}mm"
    if diameter_mm > 0:
        label += f" (d={diameter_mm:.0f}mm)"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thick = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

    label_x = cx - tw // 2
    label_y = cy - radius - 15

    # Ensure label stays on screen
    label_x = max(5, min(label_x, display.shape[1] - tw - 5))
    label_y = max(th + 5, label_y)

    # Background rectangle for label
    cv2.rectangle(display, (label_x - 4, label_y - th - 4),
                  (label_x + tw + 4, label_y + 4), color, -1, cv2.LINE_AA)
    cv2.putText(display, label, (label_x, label_y), font, font_scale,
                (0, 0, 0), font_thick, cv2.LINE_AA)


# ============================================================
# ============================================================
# MEASURE MULTIPLE TIMBERS
# ============================================================

def measure_timber(img, image_path, timber_number, timber_thickness_mm=0.0):

    print("\n" + "=" * 60)
    print(
        f"MEASURING CAPTURE {timber_number:02d} "
        f"(thickness: {timber_thickness_mm:.1f} mm)"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # 1. UNDISTORT IMAGE
    # --------------------------------------------------------

    img_undist = undistort(img)


    # --------------------------------------------------------
    # 2. DETECT ARUCO + CALCULATE HOMOGRAPHY
    # --------------------------------------------------------

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

    print(
        f"\nLive homography OK. "
        f"Maximum marker error: {max_err:.2f} mm"
    )

    if max_err > 2.0:
        print(
            f"WARNING: marker error is high "
            f"({max_err:.2f} mm)."
        )


    # --------------------------------------------------------
    # 3. DEFINE WORKSPACE
    # --------------------------------------------------------

    workspace_outer_px = order_quad(
        np.array(
            [
                detected_outer[m]
                for m in [0, 1, 2, 3]
            ],
            dtype=np.float32
        )
    )


    # --------------------------------------------------------
    # 3b. DETECT THICKNESS REFERENCE PLATE
    # --------------------------------------------------------

    ws_mask_for_plate = polygon_mask(img_undist.shape, workspace_outer_px)
    plate_detection = detect_thickness_plate(
        img_undist, H=H, workspace_mask=ws_mask_for_plate, verbose=True
    )

    thickness_source = "manual"
    if plate_detection is not None:
        timber_thickness_mm = plate_detection["thickness_mm"]
        thickness_source = f"plate ({plate_detection['color_name']})"
        print(f"\n  Thickness plate detected: {plate_detection['color_name']} "
              f"→ {timber_thickness_mm:.0f} mm "
              f"(⌀{plate_detection['diameter_mm']:.1f}mm)")
    else:
        print(f"\n  No thickness plate detected. Using manual: {timber_thickness_mm:.1f} mm")


    # --------------------------------------------------------
    # 4. DETECT ALL TIMBERS
    # --------------------------------------------------------

    detected_timbers, mask = segment_board(
        img_undist,
        workspace_outer_px,
        detected_all,
        plate_info=plate_detection
    )


    # --------------------------------------------------------
    # 5. OUTPUT PATHS
    # --------------------------------------------------------

    base_name = f"timber_{timber_number:02d}"

    os.makedirs(
        config.CAPTURE_DIR,
        exist_ok=True
    )

    mask_path = os.path.join(
        config.CAPTURE_DIR,
        f"{base_name}_mask.png"
    )

    # Save combined segmentation mask (timbers + plate if detected)
    saved_mask = mask.copy()
    if plate_detection is not None:
        cx, cy = plate_detection["center_px"]
        r = plate_detection["radius_px"]
        cv2.circle(saved_mask, (cx, cy), r, 255, thickness=cv2.FILLED)

    cv2.imwrite(
        mask_path,
        saved_mask
    )

    plate_mask_path = None
    if plate_detection is not None and "mask" in plate_detection:
        plate_mask_path = os.path.join(
            config.CAPTURE_DIR,
            f"{base_name}_plate_mask.png"
        )
        cv2.imwrite(
            plate_mask_path,
            plate_detection["mask"]
        )


    # --------------------------------------------------------
    # 6. CHECK TIMBER DETECTION
    # --------------------------------------------------------

    if not detected_timbers:

        print("\nERROR: No timber detected.")
        print(f"Mask saved to: {mask_path}")

        return False


    print(
        f"\nDetected {len(detected_timbers)} timber(s)."
    )


    # ========================================================
    # 7. WORKSPACE INFORMATION
    #    THIS IS OUTSIDE THE TIMBER LOOP
    # ========================================================

    marker_order = (
        [1, 0, 2, 3]
        if {1, 0, 2, 3}.issubset(
            detected_all.keys()
        )
        else list(detected_all.keys())
    )

    markers_info = {}

    for m_id in marker_order:

        m_corners_px = detected_all[m_id]

        m_corners_mm = pixel_to_mm(
            H,
            m_corners_px
        )

        m_center_px = (
            m_corners_px.mean(axis=0)
        )

        m_center_mm = pixel_to_mm(
            H,
            [m_center_px]
        )[0]

        m_outer_mm = pixel_to_mm(
            H,
            [detected_outer[m_id]]
        )[0]

        markers_info[str(m_id)] = {

            "outer_corner_mm": [
                round(float(m_outer_mm[0]), 2),
                round(float(m_outer_mm[1]), 2)
            ],

            "center_mm": [
                round(float(m_center_mm[0]), 2),
                round(float(m_center_mm[1]), 2)
            ],

            "corners_mm": [
                [
                    round(float(pt[0]), 2),
                    round(float(pt[1]), 2)
                ]
                for pt in m_corners_mm
            ]
        }


    # --------------------------------------------------------
    # 8. REFERENCE FRAME
    #    ALSO OUTSIDE THE TIMBER LOOP
    # --------------------------------------------------------

    reference_frame = {

        "origin_marker_id": 1,

        "origin_mm": [
            0.0,
            0.0,
            0.0
        ],

        "x_axis": [
            1.0,
            0.0,
            0.0
        ],

        "y_axis": [
            0.0,
            1.0,
            0.0
        ],

        "z_axis": [
            0.0,
            0.0,
            1.0
        ],

        "description":
            "Origin (0,0,0) at Marker 1 outer corner "
            "(Bottom-Left); +X towards Marker 0; "
            "+Y towards Marker 2"
    }


    # ========================================================
    # 9. PREPARE PARALLAX VALUES
    #    CALCULATE ONCE, USE FOR ALL TIMBERS
    # ========================================================

    H_cam = float(
        getattr(
            config,
            "CAMERA_HEIGHT_MM",
            0.0
        )
    )

    h_timber = float(
        timber_thickness_mm
    )

    use_parallax = (
        H_cam > 0
        and h_timber > 0
        and H_cam > h_timber
        and os.path.exists(
            config.CAMERA_CALIB_FILE
        )
    )


    if use_parallax:

        scale_factor = (
            H_cam - h_timber
        ) / H_cam

        calib = np.load(
            config.CAMERA_CALIB_FILE
        )

        K = calib[
            "camera_matrix"
        ]

        cx_cam = K[0, 2]
        cy_cam = K[1, 2]

        cam_center_table = pixel_to_mm(
            H,
            [[cx_cam, cy_cam]]
        )[0]

        Xc = cam_center_table[0]
        Yc = cam_center_table[1]

    else:

        scale_factor = 1.0
        Xc = 0.0
        Yc = 0.0


    # ========================================================
    # 10. DEBUG OVERLAY
    # ========================================================

    overlay = img_undist.copy()


    # Draw workspace
    cv2.polylines(
        overlay,
        [np.int32(workspace_outer_px)],
        True,
        (0, 0, 255),
        3
    )


    # Draw ArUco markers
    for m_id in [0, 1, 2, 3]:

        if m_id in detected_all:

            pts = detected_all[m_id]

            cv2.polylines(
                overlay,
                [np.int32(pts)],
                True,
                (255, 0, 255),
                2
            )

            c = pts.mean(axis=0)

            cv2.putText(
                overlay,
                f"ID {m_id}",
                (
                    int(c[0]),
                    int(c[1])
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )


    # ========================================================
    # 11. CREATE LIST FOR ALL TIMBERS
    # ========================================================

    timbers_result = []
    global HANDDRAW_DETECTOR
    if HANDDRAW_DETECTOR is None:
        try:
            HANDDRAW_DETECTOR = HanddrawDetector()
        except Exception as exc:
            print(f"Warning: Could not initialize HanddrawDetector: {exc}")

    drawing_groups = (HANDDRAW_DETECTOR.detect(img_undist, detected_timbers)
                      if HANDDRAW_DETECTOR is not None
                      else [[] for _ in detected_timbers])


    # ========================================================
    # 12. LOOP THROUGH EVERY DETECTED TIMBER
    # ========================================================

    for i, timber in enumerate(
        detected_timbers
    ):

        print(
            "\n" +
            "-" * 50
        )

        print(
            f"Processing timber {i + 1}"
        )

        print(
            "-" * 50
        )


        # ----------------------------------------------------
        # GET THIS TIMBER'S CONTOUR + CORNERS
        # ----------------------------------------------------

        contour = timber[
            "contour"
        ]

        timber_corners_px = timber[
            "corners_px"
        ]

        holes = timber.get("holes", [])
        timber_mask_single = timber.get("mask", None)


        # ----------------------------------------------------
        # PIXEL -> MM : CORNERS
        # ----------------------------------------------------

        corners_mm_table = pixel_to_mm(
            H,
            timber_corners_px
        )


        # ----------------------------------------------------
        # CONTOUR APPROXIMATION
        # ----------------------------------------------------

        peri = cv2.arcLength(
            contour,
            True
        )

        approx_factor = float(
            getattr(
                config,
                "CONTOUR_APPROX_FACTOR",
                0.001
            )
        )

        epsilon = (
            approx_factor
            * peri
        )

        approx_contour_px = (
            cv2.approxPolyDP(
                contour,
                epsilon,
                True
            )
        )

        approx_cnt = (
            approx_contour_px
            .reshape(-1, 2)
        )

        contour_mm_table = pixel_to_mm(
            H,
            approx_cnt
        )

        # Approximate and convert all inner cuts (holes)
        inner_smooth_method = getattr(config, "INNER_CONTOUR_SMOOTH_METHOD", "approx").lower()
        inner_approx_factor = float(getattr(config, "INNER_CONTOUR_APPROX_FACTOR", approx_factor))
        inner_spline_pts = int(getattr(config, "INNER_CONTOUR_SPLINE_POINTS", 120))

        approx_holes_px = []
        holes_mm_table = []
        holes_mm = []
        hole_areas_mm2 = []

        for h_cnt in holes:
            if inner_smooth_method == "spline":
                approx_h_px = smooth_contour_spline(h_cnt, num_points=inner_spline_pts)
            else:
                h_peri = cv2.arcLength(h_cnt, True)
                h_eps = inner_approx_factor * h_peri
                approx_h_px = cv2.approxPolyDP(h_cnt, h_eps, True)
            approx_holes_px.append(approx_h_px)

            h_cnt_pts = approx_h_px.reshape(-1, 2)
            h_mm_table = pixel_to_mm(H, h_cnt_pts)
            holes_mm_table.append(h_mm_table)

            if use_parallax:
                h_mm = np.array([
                    [
                        Xc + (pt[0] - Xc) * scale_factor,
                        Yc + (pt[1] - Yc) * scale_factor
                    ]
                    for pt in h_mm_table
                ], dtype=np.float32)
            else:
                h_mm = h_mm_table

            holes_mm.append(h_mm)
            h_area_mm2 = abs(cv2.contourArea(h_mm.astype(np.float32).reshape(-1, 1, 2)))
            hole_areas_mm2.append(h_area_mm2)


        print(
            f"Raw contour points: "
            f"{len(contour)}"
        )

        print(
            f"Approximated points: "
            f"{len(approx_contour_px)}"
        )

        if holes:
            print(
                f"Detected inner cuts (holes): {len(holes)} "
                f"(smooth: {inner_smooth_method})"
            )


        # ----------------------------------------------------
        # PARALLAX COMPENSATION
        # ----------------------------------------------------

        if use_parallax:

            corners_mm = np.array(

                [
                    [
                        Xc
                        + (
                            pt[0] - Xc
                        )
                        * scale_factor,

                        Yc
                        + (
                            pt[1] - Yc
                        )
                        * scale_factor
                    ]

                    for pt
                    in corners_mm_table
                ],

                dtype=np.float32
            )


            contour_mm = np.array(

                [
                    [
                        Xc
                        + (
                            pt[0] - Xc
                        )
                        * scale_factor,

                        Yc
                        + (
                            pt[1] - Yc
                        )
                        * scale_factor
                    ]

                    for pt
                    in contour_mm_table
                ],

                dtype=np.float32
            )

        else:

            corners_mm = (
                corners_mm_table
            )

            contour_mm = (
                contour_mm_table
            )


        # ----------------------------------------------------
        # SURFACE AREA (Gross outer area minus all inner cuts/holes)
        # ----------------------------------------------------

        gross_surface_area_mm2 = abs(

            cv2.contourArea(

                contour_mm
                .astype(
                    np.float32
                )
                .reshape(
                    -1,
                    1,
                    2
                )
            )
        )

        total_hole_area_mm2 = float(sum(hole_areas_mm2))
        surface_area_mm2 = max(0.0, gross_surface_area_mm2 - total_hole_area_mm2)

        surface_area_cm2 = (
            surface_area_mm2
            / 100.0
        )


        # ----------------------------------------------------
        # DIMENSIONS
        # ----------------------------------------------------

        side_lengths = [

            float(

                np.linalg.norm(

                    corners_mm[
                        (j + 1) % 4
                    ]

                    -

                    corners_mm[j]
                )
            )

            for j in range(4)
        ]


        dim_a = (
            side_lengths[0]
            +
            side_lengths[2]
        ) / 2.0


        dim_b = (
            side_lengths[1]
            +
            side_lengths[3]
        ) / 2.0


        length_mm = max(
            dim_a,
            dim_b
        )

        width_mm = min(
            dim_a,
            dim_b
        )


        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        color_info = sample_timber_color(
            img_undist,
            contour,
            holes=holes,
            timber_mask=timber_mask_single,
            plate_info=plate_detection
        )


        # ----------------------------------------------------
        # CREATE THIS TIMBER'S JSON DATA
        # ----------------------------------------------------

        timber_result = {

            "timber_id":
                0 if timber_number == 0 else (i + 1),

            "length_mm":
                round(
                    length_mm,
                    2
                ),

            "width_mm":
                round(
                    width_mm,
                    2
                ),

            "gross_surface_area_mm2":
                round(
                    float(
                        gross_surface_area_mm2
                    ),
                    2
                ),

            "total_hole_area_mm2":
                round(
                    float(
                        total_hole_area_mm2
                    ),
                    2
                ),

            "surface_area_mm2":
                round(
                    float(
                        surface_area_mm2
                    ),
                    2
                ),

            "surface_area_cm2":
                round(
                    float(
                        surface_area_cm2
                    ),
                    2
                ),

            "side_lengths_mm": [
                round(
                    s,
                    2
                )
                for s
                in side_lengths
            ],

            "corners_mm": [

                [
                    round(
                        float(x),
                        2
                    ),

                    round(
                        float(y),
                        2
                    )
                ]

                for x, y
                in corners_mm
            ],

            "contour_mm": [

                [
                    round(
                        float(x),
                        2
                    ),

                    round(
                        float(y),
                        2
                    )
                ]

                for x, y
                in contour_mm
            ],

            "holes_mm": [

                [
                    [
                        round(
                            float(x),
                            2
                        ),

                        round(
                            float(y),
                            2
                        )
                    ]

                    for x, y
                    in h_mm
                ]

                for h_mm
                in holes_mm
            ],

            "inner_contours_mm": [

                [
                    [
                        round(
                            float(x),
                            2
                        ),

                        round(
                            float(y),
                            2
                        )
                    ]

                    for x, y
                    in h_mm
                ]

                for h_mm
                in holes_mm
            ],

            "hole_count":
                len(holes_mm),

            "timber_thickness_mm":
                round(
                    timber_thickness_mm,
                    1
                ),

            "thickness_source":
                thickness_source,

            "color_rgb":
                color_info[
                    "rgb"
                ],

            "color_hex":
                color_info[
                    "hex"
                ],

            "color_lab":
                color_info[
                    "lab"
                ],

            "defects": []
        }


        # Drawing locations use the same undistorted image and height correction.
        timber_result["hand_drawn_shapes"] = project_detections(
            drawing_groups[i], H,
            scale=scale_factor if use_parallax else 1.0,
            origin=(Xc, Yc) if use_parallax else (0.0, 0.0),
        )
        timber_result["hand_drawn_shape_count"] = len(drawing_groups[i])

        # ADD THIS TIMBER TO LIST
        timbers_result.append(
            timber_result
        )


        # ====================================================
        # DRAW THIS TIMBER ON DEBUG OVERLAY
        # ====================================================

        # Yellow outer contour
        cv2.drawContours(
            overlay,
            [approx_contour_px],
            -1,
            (0, 255, 255),
            3
        )

        # Yellow/Orange contour for inner cuts (holes)
        if approx_holes_px:
            cv2.drawContours(
                overlay,
                approx_holes_px,
                -1,
                (0, 215, 255),
                2
            )
            for h_idx, (h_px, h_a_mm2) in enumerate(zip(approx_holes_px, hole_areas_mm2)):
                h_mom = cv2.moments(h_px)
                if h_mom["m00"] > 0:
                    hcx = int(h_mom["m10"] / h_mom["m00"])
                    hcy = int(h_mom["m01"] / h_mom["m00"])
                    cv2.putText(
                        overlay,
                        f"Inner Cut #{h_idx + 1} ({h_a_mm2 / 100.0:.1f} cm2)",
                        (hcx - 70, hcy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 215, 255),
                        2
                    )


        # Green bounding box
        cv2.polylines(
            overlay,
            [
                np.int32(
                    timber_corners_px
                )
            ],
            True,
            (0, 255, 0),
            3
        )


        # Corner points
        for corner_index, pt in enumerate(
            timber_corners_px
        ):

            x = int(
                round(pt[0])
            )

            y = int(
                round(pt[1])
            )

            cv2.circle(
                overlay,
                (x, y),
                7,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # LABEL TIMBER
        # ----------------------------------------------------

        center_x = int(
            np.mean(
                timber_corners_px[:, 0]
            )
        )

        center_y = int(
            np.mean(
                timber_corners_px[:, 1]
            )
        )


        cv2.putText(
            overlay,

            (
                f"T{i + 1}: "
                f"{length_mm:.1f} x "
                f"{width_mm:.1f} mm"
            ),

            (
                center_x - 80,
                center_y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # PRINT THIS TIMBER'S RESULTS
        # ----------------------------------------------------

        print(
            f"Timber ID:     {i + 1}"
        )

        print(
            f"Length:        "
            f"{length_mm:.2f} mm"
        )

        print(
            f"Width:         "
            f"{width_mm:.2f} mm"
        )

        print(
            f"Gross Area:    "
            f"{gross_surface_area_mm2:.1f} mm² "
            f"({gross_surface_area_mm2 / 100.0:.2f} cm²)"
        )

        if holes_mm:
            print(
                f"Inner Cuts:    "
                f"{len(holes_mm)} hole(s), total cutout: {total_hole_area_mm2:.1f} mm² "
                f"({total_hole_area_mm2 / 100.0:.2f} cm²)"
            )

        print(
            f"Net Area:      "
            f"{surface_area_mm2:.1f} mm² "
            f"({surface_area_cm2:.2f} cm²)"
        )

        print(
            f"Sides:         "
            f"{[round(s, 2) for s in side_lengths]}"
        )

        print(
            f"Color:         "
            f"{color_info['hex']}"
        )


    # ========================================================
    # 13. LOOP FINISHED
    #     NOW BUILD ONE JSON FOR THE WHOLE CAPTURE
    # ========================================================

    result = {

        "capture_id":
            timber_number,

        "source_image":
            os.path.abspath(
                image_path
            ),

        "reference_frame":
            reference_frame,

        "markers_world_mm":
            markers_info,

        "timber_count":
            len(
                timbers_result
            ),

        "timbers":
            timbers_result
    }


    # ========================================================
    # 14. SAVE ONE JSON FILE
    # ========================================================

    result["handdraw_detection"] = (HANDDRAW_DETECTOR.metadata()
                                    if HANDDRAW_DETECTOR is not None else {"enabled": False})
    result["hand_drawn_shape_count"] = sum(len(group) for group in drawing_groups)

    # Add thickness plate detection info to JSON
    if plate_detection is not None:
        result["thickness_plate"] = {
            "detected": True,
            "color_name": plate_detection["color_name"],
            "thickness_mm": plate_detection["thickness_mm"],
            "diameter_mm": plate_detection["diameter_mm"],
            "circularity": plate_detection["circularity"],
            "center_px": list(plate_detection["center_px"]),
        }
    else:
        result["thickness_plate"] = {"detected": False}

    draw_detections(overlay, [item for group in drawing_groups for item in group])
    draw_plate_overlay(overlay, plate_detection)

    out_path = os.path.join(

        config.CAPTURE_DIR,

        f"{base_name}_measurement.json"
    )


    with open(
        out_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


    # ========================================================
    # 15. SAVE DEBUG OVERLAY
    # ========================================================

    cv2.putText(
        overlay,

        (
            f"Detected timbers: "
            f"{len(timbers_result)}"
        ),

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2
    )


    overlay_path = os.path.join(

        config.CAPTURE_DIR,

        f"{base_name}_overlay.png"
    )


    cv2.imwrite(
        overlay_path,
        overlay
    )


    # ========================================================
    # 16. FINAL REPORT
    # ========================================================

    print(
        "\n" +
        "=" * 60
    )

    print(
        f"CAPTURE {timber_number:02d} COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Detected timbers: "
        f"{len(timbers_result)}"
    )

    print(
        f"JSON:    {out_path}"
    )

    print(
        f"Mask:    {mask_path}"
    )

    if plate_mask_path:
        print(
            f"Plate:   {plate_mask_path}"
        )

    print(
        f"Overlay: {overlay_path}"
    )


    return True

# ============================================================
# REAL-TIME LIVE VIEWPORT OVERLAY - MULTI TIMBER
# ============================================================

def render_live_viewport(img_undist, timber_number, current_thickness):
    """
    Real-time multi-timber detection viewport.

    Shows:
    - Red workspace boundary
    - Magenta ArUco markers
    - Yellow contour for every detected timber
    - Green bounding box for every detected timber
    - Timber ID and live dimensions
    - Detected thickness reference plate (circle + label)

    Returns
    -------
    display : ndarray
        Rendered viewport image.
    effective_thickness : float
        The thickness to use: plate-detected if visible, otherwise manual.
    plate_info : dict or None
        Plate detection result, or None.
    """

    display = img_undist.copy()

    # --------------------------------------------------------
    # 1. DETECT ARUCO + HOMOGRAPHY
    # --------------------------------------------------------

    (
        H,
        errors_mm,
        detected_outer,
        detected_inner,
        detected_all
    ) = compute_live_homography(
        img_undist,
        verbose=False
    )

    detected_timbers = []
    timber_detected = False
    timber_count = 0
    plate_info = None
    effective_thickness = current_thickness

    status_text = "Place timber(s) inside black workspace"


    # --------------------------------------------------------
    # 2. IF WORKSPACE IS DETECTED
    # --------------------------------------------------------

    if H is not None and detected_outer is not None:

        # Physical workspace
        workspace_outer_px = order_quad(
            np.array(
                [
                    detected_outer[m]
                    for m in [1, 0, 2, 3]
                    if m in detected_outer
                ],
                dtype=np.float32
            )
        )


        # ----------------------------------------------------
        # DRAW WORKSPACE
        # ----------------------------------------------------

        cv2.polylines(
            display,
            [np.int32(workspace_outer_px)],
            True,
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # DETECT THICKNESS REFERENCE PLATE (LIVE)
        # ----------------------------------------------------

        ws_mask_live = polygon_mask(img_undist.shape, workspace_outer_px)
        plate_info = detect_thickness_plate(
            img_undist, H=H, workspace_mask=ws_mask_live, verbose=False
        )
        if plate_info is not None:
            effective_thickness = plate_info["thickness_mm"]
            draw_plate_overlay(display, plate_info, active=True)


        # ----------------------------------------------------
        # DRAW ARUCO MARKERS
        # ----------------------------------------------------

        for m_id in [1, 0, 2, 3]:

            if m_id in detected_all:

                pts = detected_all[m_id]

                cv2.polylines(
                    display,
                    [np.int32(pts)],
                    True,
                    (255, 0, 255),
                    2
                )

                c = pts.mean(axis=0)

                cv2.putText(
                    display,
                    f"ID {m_id}",
                    (
                        int(c[0]) - 20,
                        int(c[1])
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2
                )


        # ----------------------------------------------------
        # 3. DETECT ALL TIMBERS
        # ----------------------------------------------------

        detected_timbers, mask = segment_board(
            img_undist,
            workspace_outer_px,
            detected_all,
            verbose=False,
            plate_info=plate_info
        )


        # ----------------------------------------------------
        # 4. IF TIMBERS FOUND
        # ----------------------------------------------------

        if detected_timbers:

            timber_detected = True

            timber_count = len(
                detected_timbers
            )

            status_text = (
                f"{timber_count} timber(s) detected"
            )


            # =================================================
            # PREPARE PARALLAX VALUES ONCE
            # =================================================

            H_cam = float(
                getattr(
                    config,
                    "CAMERA_HEIGHT_MM",
                    0.0
                )
            )

            h_timber = float(
                effective_thickness
            )

            use_parallax = (
                H_cam > 0
                and h_timber > 0
                and H_cam > h_timber
                and os.path.exists(
                    config.CAMERA_CALIB_FILE
                )
            )


            if use_parallax:

                scale_factor = (
                    H_cam - h_timber
                ) / H_cam

                calib = np.load(
                    config.CAMERA_CALIB_FILE
                )

                K = calib[
                    "camera_matrix"
                ]

                cx_cam = K[0, 2]
                cy_cam = K[1, 2]

                cam_center_table = pixel_to_mm(
                    H,
                    [[cx_cam, cy_cam]]
                )[0]

                Xc = cam_center_table[0]
                Yc = cam_center_table[1]


            # =================================================
            # 5. LOOP THROUGH ALL TIMBERS
            # =================================================

            for timber_index, timber in enumerate(
                detected_timbers
            ):

                contour = timber[
                    "contour"
                ]

                timber_corners_px = timber[
                    "corners_px"
                ]


                # ---------------------------------------------
                # CONTOUR APPROXIMATION
                # ---------------------------------------------

                peri = cv2.arcLength(
                    contour,
                    True
                )

                approx_factor = float(
                    getattr(
                        config,
                        "CONTOUR_APPROX_FACTOR",
                        0.001
                    )
                )

                epsilon = (
                    approx_factor
                    * peri
                )

                approx_contour_px = (
                    cv2.approxPolyDP(
                        contour,
                        epsilon,
                        True
                    )
                )


                # ---------------------------------------------
                # DRAW YELLOW CONTOUR (OUTER) & INNER CUTS
                # ---------------------------------------------

                cv2.drawContours(
                    display,
                    [approx_contour_px],
                    -1,
                    (0, 255, 255),
                    2
                )

                # Draw inner cuts (holes) in live display
                holes = timber.get("holes", [])
                if holes:
                    inner_smooth_method = getattr(config, "INNER_CONTOUR_SMOOTH_METHOD", "approx").lower()
                    inner_approx_f = float(getattr(config, "INNER_CONTOUR_APPROX_FACTOR", approx_factor))
                    inner_spline_n = int(getattr(config, "INNER_CONTOUR_SPLINE_POINTS", 120))
                    approx_holes = []
                    for h in holes:
                        if inner_smooth_method == "spline":
                            approx_holes.append(smooth_contour_spline(h, num_points=inner_spline_n))
                        else:
                            approx_holes.append(
                                cv2.approxPolyDP(h, inner_approx_f * cv2.arcLength(h, True), True)
                            )
                    cv2.drawContours(
                        display,
                        approx_holes,
                        -1,
                        (0, 215, 255),
                        2
                    )


                # ---------------------------------------------
                # DRAW GREEN BOUNDING BOX
                # ---------------------------------------------

                cv2.polylines(
                    display,
                    [
                        np.int32(
                            timber_corners_px
                        )
                    ],
                    True,
                    (0, 255, 0),
                    2
                )


                # ---------------------------------------------
                # DRAW CORNER POINTS
                # ---------------------------------------------

                for pt in timber_corners_px:

                    x = int(
                        round(pt[0])
                    )

                    y = int(
                        round(pt[1])
                    )

                    cv2.circle(
                        display,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )


                # ---------------------------------------------
                # PIXEL -> MM
                # ---------------------------------------------

                corners_mm_raw = pixel_to_mm(
                    H,
                    timber_corners_px
                )


                # ---------------------------------------------
                # PARALLAX CORRECTION
                # ---------------------------------------------

                if use_parallax:

                    corners_mm = np.array(
                        [
                            [
                                Xc
                                + (
                                    pt[0] - Xc
                                )
                                * scale_factor,

                                Yc
                                + (
                                    pt[1] - Yc
                                )
                                * scale_factor
                            ]

                            for pt
                            in corners_mm_raw
                        ],

                        dtype=np.float32
                    )

                else:

                    corners_mm = (
                        corners_mm_raw
                    )


                # ---------------------------------------------
                # DIMENSIONS
                # ---------------------------------------------

                side_lengths = [

                    float(

                        np.linalg.norm(

                            corners_mm[
                                (j + 1) % 4
                            ]

                            -

                            corners_mm[j]
                        )
                    )

                    for j in range(4)
                ]


                dim_a = (
                    side_lengths[0]
                    +
                    side_lengths[2]
                ) / 2.0


                dim_b = (
                    side_lengths[1]
                    +
                    side_lengths[3]
                ) / 2.0


                length_mm = max(
                    dim_a,
                    dim_b
                )

                width_mm = min(
                    dim_a,
                    dim_b
                )


                # ---------------------------------------------
                # CENTER OF TIMBER
                # ---------------------------------------------

                center_x = int(
                    timber_corners_px[
                        :, 0
                    ].mean()
                )

                center_y = int(
                    timber_corners_px[
                        :, 1
                    ].mean()
                )


                # ---------------------------------------------
                # TIMBER LABEL
                # ---------------------------------------------

                cv2.putText(
                    display,

                    (
                        f"T{timber_index + 1}: "
                        f"{length_mm:.1f} x "
                        f"{width_mm:.1f} mm"
                    ),

                    (
                        center_x - 90,
                        center_y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 255, 0),

                    2
                )


        else:

            status_text = (
                "No timber detected inside workspace"
            )


    else:

        status_text = (
            "Looking for 4 ArUco markers..."
        )


    # ========================================================
    # 6. TOP HUD & HAND-DRAWN SHAPES
    # ========================================================

    drawings = []
    if HANDDRAW_DETECTOR is not None:
        if detected_timbers:
            drawing_groups = HANDDRAW_DETECTOR.detect(img_undist, detected_timbers)
            drawings = [item for group in drawing_groups for item in group]
            draw_detections(display, drawings)
        elif H is not None and detected_outer is not None:
            drawing_groups = HANDDRAW_DETECTOR.detect_workspace(img_undist, workspace_outer_px)
            drawings = [item for group in drawing_groups for item in group]
            draw_detections(display, drawings)

    h, w = display.shape[:2]

    header = np.zeros(
        (
            85,
            w,
            3
        ),
        dtype=np.uint8
    )

    drawing_info = f"  |  {len(drawings)} drawing(s)" if HANDDRAW_DETECTOR is not None else ""
    conf_info = f"  |  YOLO: {HANDDRAW_DETECTOR.confidence:.2f} ([/])" if HANDDRAW_DETECTOR is not None else ""

    # First line
    if plate_info is not None:
        thickness_label = f"T: {effective_thickness:.0f}mm ({plate_info['color_name'].upper()} PLATE)"
    else:
        thickness_label = f"T: {current_thickness:.1f}mm (manual)"

    if timber_detected:
        top_text = (
            f"CAPTURE {timber_number:02d}"
            f"  |  "
            f"{timber_count} timber(s)"
            f"{drawing_info}"
            f"{conf_info}"
            f"  |  "
            f"{thickness_label}"
        )
    else:
        top_text = (
            f"CAPTURE {timber_number:02d}"
            f"  |  "
            f"{status_text}"
            f"{drawing_info}"
            f"{conf_info}"
            f"  |  "
            f"{thickness_label}"
        )


    cv2.putText(
        header,
        top_text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (
            (0, 255, 0)
            if timber_detected
            else (0, 200, 255)
        ),
        2
    )


    # Second line
    cv2.putText(
        header,

        (
            "SPACE = Capture & Save JSON"
            "  |  "
            "T = Thickness"
            "  |  "
            "ENTER/ESC = Exit"
        ),

        (20, 68),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (220, 220, 220),

        2
    )


    # Blend HUD
    display[:85] = cv2.addWeighted(
        display[:85],
        0.25,
        header,
        0.75,
        0
    )


    return display, effective_thickness, plate_info
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
    parser.add_argument("--handdraw-model", default=str(DEFAULT_MODEL),
                        help="Trained ellipse/rectangle/triangle best.pt")
    parser.add_argument("--yolo-device", default="cpu", help="cpu (default), or GPU index such as 0")
    parser.add_argument("--yolo-confidence", type=float, default=0.30)
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--no-handdraw", action="store_true", help="Only measure timbers")
    args = parser.parse_args()

    global HANDDRAW_DETECTOR
    if not args.no_handdraw:
        try:
            HANDDRAW_DETECTOR = HanddrawDetector(
                args.handdraw_model, args.yolo_device, args.yolo_confidence,
                args.yolo_image_size, args.yolo_iou,
            )
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            parser.error(str(exc))
        print(f"Hand-drawing detection enabled: {HANDDRAW_DETECTOR.model_path}")
        print(f"YOLO device: {args.yolo_device}; confidence: {args.yolo_confidence}")

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
    print("Place timber(s) with hand-drawn shapes inside the black workspace.")
    print("Live contour, detection box, and dimensions are displayed in real-time.")
    print("SPACE       = capture and measure")
    print("[ / ]       = decrease / increase YOLO confidence threshold")
    print("T           = change timber thickness (for parallax correction)")
    print("              (color plates auto-detect thickness when visible)")
    print("ENTER / ESC = exit")
    print(f"\nCurrent timber thickness: {current_thickness:.1f} mm")
    print(f"Camera height: {getattr(config, 'CAMERA_HEIGHT_MM', 0.0):.0f} mm")
    
    # Print configured thickness plates
    plates_cfg = getattr(config, "THICKNESS_PLATES", {})
    if plates_cfg:
        print(f"\nThickness plates configured ({len(plates_cfg)} colors):")
        for c_name, c_info in plates_cfg.items():
            print(f"  {c_name.capitalize():8s} → {c_info['thickness_mm']:.0f} mm")
    
    print(f"Output folder: {config.CAPTURE_DIR}")

    timber_number = get_next_timber_number()

    window_name = "Timber Measurement - Live View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 850)

    zoom_factor = float(getattr(config, "CAMERA_ZOOM", getattr(config, "CAMERA_ZOOM_VALUE", 1.0)))

    live_thickness = current_thickness  # Effective thickness (plate or manual)

    # Start Grasshopper remote capture listener (TCP Port 9998)
    gh_server = GrasshopperCaptureServer(
        host=getattr(config, "GRASSHOPPER_TRIGGER_HOST", "127.0.0.1"),
        port=int(getattr(config, "GRASSHOPPER_TRIGGER_PORT", 9998))
    )
    gh_server.start()

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("ERROR: Failed to read camera frame.")
                break

            frame = camera_utils.apply_digital_zoom(raw_frame, zoom_factor)
            img_undist = undistort(frame)
            display, live_thickness, live_plate = render_live_viewport(
                img_undist, timber_number, current_thickness
            )

            cv2.imshow(window_name, display)
            target_fps = float(getattr(config, "CAMERA_FPS", 1))
            frame_delay = max(1, int(1000.0 / max(1.0, target_fps)))
            key = cv2.waitKey(frame_delay) & 0xFF

            # --------------------------------------------------------
            # CHECK FOR REMOTE CAPTURE TRIGGER (GRASSHOPPER BUTTON)
            # --------------------------------------------------------
            gh_req = gh_server.get_pending_request()
            file_trigger = False
            trigger_file_path = getattr(config, "CAPTURE_TRIGGER_FILE", os.path.join(config.CAPTURE_DIR, "capture_trigger.json"))
            if gh_req is None and os.path.exists(trigger_file_path):
                try:
                    with open(trigger_file_path, "r", encoding="utf-8") as tf:
                        tf_data = json.load(tf)
                    os.remove(trigger_file_path)
                    file_trigger = True
                    gh_req = {
                        "number": int(tf_data.get("number", getattr(config, "GRASSHOPPER_RESERVED_TIMBER_NUMBER", 0))),
                        "event": None,
                        "client": None,
                        "result": None
                    }
                except Exception:
                    pass

            if gh_req is not None:
                gh_num = int(gh_req.get("number", getattr(config, "GRASSHOPPER_RESERVED_TIMBER_NUMBER", 0)))
                capture_thickness = live_thickness
                source_label = "plate" if live_plate is not None else "manual"
                print("\n" + "=" * 60)
                print(f"[GRASSHOPPER REMOTE CAPTURE] Capturing timber {gh_num:02d} "
                      f"(thickness: {capture_thickness:.1f} mm [{source_label}])...")
                print("=" * 60)
                image_path = os.path.join(config.CAPTURE_DIR, f"timber_{gh_num:02d}.jpg")
                cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                print(f"Captured: {image_path}")

                success = measure_timber(frame, image_path, gh_num, capture_thickness)
                json_path = os.path.join(config.CAPTURE_DIR, f"timber_{gh_num:02d}_measurement.json")
                res_payload = {
                    "status": "success" if success else "error",
                    "timber_number": gh_num,
                    "json_path": json_path,
                    "image_path": image_path
                }
                gh_req["result"] = res_payload
                if gh_req.get("event") is not None:
                    gh_req["event"].set()

                if file_trigger:
                    try:
                        status_path = os.path.join(config.CAPTURE_DIR, "capture_status.json")
                        with open(status_path, "w", encoding="utf-8") as sf:
                            json.dump(res_payload, sf)
                    except Exception:
                        pass

                if success:
                    print(f"\n[GRASSHOPPER] Timber {gh_num:02d} captured and saved -> {json_path}")
                else:
                    print(f"\n[GRASSHOPPER] Timber {gh_num:02d} measurement failed.")
                # Manual timber_number is NOT incremented; timber 00 is reserved for Grasshopper.

            if key in (ord('['), ord('-')):  # Lower confidence
                if HANDDRAW_DETECTOR is not None:
                    new_conf = max(0.01, HANDDRAW_DETECTOR.confidence - 0.02)
                    HANDDRAW_DETECTOR.set_confidence(new_conf)
                    print(f"YOLO Confidence: {new_conf:.2f}")

            elif key in (ord(']'), ord('+'), ord('=')):  # Increase confidence
                if HANDDRAW_DETECTOR is not None:
                    new_conf = min(0.95, HANDDRAW_DETECTOR.confidence + 0.02)
                    HANDDRAW_DETECTOR.set_confidence(new_conf)
                    print(f"YOLO Confidence: {new_conf:.2f}")

            elif key == ord('t') or key == ord('T'):  # T = change manual thickness
                print(f"\nCurrent manual thickness: {current_thickness:.1f} mm")
                if live_plate is not None:
                    print(f"  (NOTE: Plate detected → auto thickness: {live_plate['thickness_mm']:.0f} mm. "
                          f"Remove plate to use manual thickness.)")
                try:
                    new_val = input("Enter new manual timber thickness in mm (e.g. 20, 35, 50): ").strip()
                    current_thickness = float(new_val)
                    print(f"Manual timber thickness set to: {current_thickness:.1f} mm")
                except (ValueError, EOFError):
                    print("Invalid input. Thickness unchanged.")

            elif key == 32:  # SPACE
                # Use live_thickness: plate-detected if plate is visible, otherwise manual
                capture_thickness = live_thickness
                source_label = "plate" if live_plate is not None else "manual"
                print(f"\nCapturing timber {timber_number:02d} "
                      f"(thickness: {capture_thickness:.1f} mm [{source_label}])...")
                image_path = os.path.join(config.CAPTURE_DIR, f"timber_{timber_number:02d}.jpg")
                cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                print(f"Captured: {image_path}")

                success = measure_timber(frame, image_path, timber_number, capture_thickness)
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
        gh_server.stop()
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
