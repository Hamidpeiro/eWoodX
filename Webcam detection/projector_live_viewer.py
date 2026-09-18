"""
eWoodX - Real-Time Projector Live Viewer (TCP Server & Interactive Display)

Features:
  - Runs at the Projector Resolution (from config.py, fullscreen at PROJECTOR_SCREEN_ORIGIN_X, Y)
  - Built-in High-Performance TCP Socket Server (Port 9999) receiving live geometry from Grasshopper
  - Real-time Multi-Color Rendering:
      * Base Timber Contour & Corners
      * Cutting Lines (Red / Customizable)
      * Milling / Pocket Toolpaths (Cyan / Blue / Customizable)
      * Drill / Reference Points (Green / Customizable)
      * Text Labels & Annotations (Amber / Customizable)
  - 3D Optical Parallax Compensation for Timber Thickness (live updated or adjusted via hotkeys)
  - Table-Level Interactive Offsets Nudging (W/A/S/D or Arrow Keys)
  - Fallback mode: Automatically loads latest scanned JSON when Grasshopper is disconnected
"""

import cv2
import numpy as np
import json
import socket
import threading
import time
import os
import glob
import sys
from pathlib import Path
import config

# ============================================================
# PROJECTOR RESOLUTION & POSITION
# ============================================================

PROJECTOR_WIDTH = config.PROJECTOR_WIDTH
PROJECTOR_HEIGHT = config.PROJECTOR_HEIGHT

# Screen origin (offset on desktop display arrangement)
PROJECTOR_SCREEN_ORIGIN_X = config.PROJECTOR_SCREEN_ORIGIN_X
PROJECTOR_SCREEN_ORIGIN_Y = config.PROJECTOR_SCREEN_ORIGIN_Y

WINDOW_NAME = "eWoodX Projector Live Viewer"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 9999

PROJECTOR_HOMOGRAPHY_PATH = Path(
    os.path.join(config.CALIB_DIR, "projector_homography.npz")
)

# 3D Projector Mounting & Parallax Parameters (mm)
projector_height_mm = float(getattr(config, "PROJECTOR_HEIGHT_MM", 2300.0))
projector_pos_x_mm = float(getattr(config, "PROJECTOR_POS_X_MM", 920.0))
projector_pos_y_mm = float(getattr(config, "PROJECTOR_POS_Y_MM", 50.0))

# Fine tuning offsets (mm)
offset_x_mm = float(getattr(config, "PROJECTOR_OFFSET_X_MM", 0.0))
offset_y_mm = float(getattr(config, "PROJECTOR_OFFSET_Y_MM", 0.0))
step_mm = 0.5
timber_thickness_mm = float(getattr(config, "TIMBER_THICKNESS_MM", 0.0))

SHOW_HUD = True
SHOW_GRID = False
DARK_BG = (15, 15, 15)

# ============================================================
# PROJECTION LINE THICKNESS & DISPLAY SIZES (Pixels)
# ============================================================
DEFAULT_TIMBER_THICKNESS_PX = 2      # Timber boundary line width
DEFAULT_CUT_THICKNESS_PX = 3         # Cut lines width
DEFAULT_MILL_THICKNESS_PX = 2        # Milling paths width
DEFAULT_DRILL_RADIUS_PX = 6          # Drill hole circle radius
DEFAULT_DRILL_THICKNESS_PX = 2       # Drill hole circle stroke width
DEFAULT_LABEL_SCALE = 0.45           # Text annotation size
DEFAULT_LABEL_THICKNESS = 1          # Text annotation stroke width


# ============================================================
# LOAD PROJECTOR HOMOGRAPHY
# ============================================================

if not PROJECTOR_HOMOGRAPHY_PATH.exists():
    print(f"Warning: Projector calibration not found at {PROJECTOR_HOMOGRAPHY_PATH}.")
    print("Using default identity homography for testing.")
    H_WORLD_TO_PROJECTOR = np.eye(3, dtype=np.float32)
else:
    projector_data = np.load(PROJECTOR_HOMOGRAPHY_PATH, allow_pickle=True)
    H_WORLD_TO_PROJECTOR = projector_data["H_world_to_projector"]


# ============================================================
# COLOR CONVERSION HELPERS
# ============================================================

def parse_color(color_val, default_bgr=(255, 255, 255)):
    """Convert Hex string, RGB tuple/list, or BGR tuple to OpenCV BGR tuple."""
    if color_val is None:
        return default_bgr
    try:
        if isinstance(color_val, str):
            c_str = color_val.strip().lstrip("#")
            if len(c_str) == 6:
                r = int(c_str[0:2], 16)
                g = int(c_str[2:4], 16)
                b = int(c_str[4:6], 16)
                return (b, g, r)  # BGR for OpenCV
            elif len(c_str) == 8:
                r = int(c_str[0:2], 16)
                g = int(c_str[2:4], 16)
                b = int(c_str[4:6], 16)
                return (b, g, r)
        elif isinstance(color_val, (list, tuple)):
            if len(color_val) >= 3:
                # Assume RGB if incoming from Grasshopper
                r, g, b = int(color_val[0]), int(color_val[1]), int(color_val[2])
                return (b, g, r)
        elif isinstance(color_val, dict):
            r = int(color_val.get("R", color_val.get("r", 255)))
            g = int(color_val.get("G", color_val.get("g", 255)))
            b = int(color_val.get("B", color_val.get("b", 255)))
            return (b, g, r)
    except Exception:
        pass
    return default_bgr


# ============================================================
# WORLD MM -> PROJECTOR PIXEL TRANSFORM WITH 3D PARALLAX
# ============================================================

def world_to_projector(points_mm, off_x=0.0, off_y=0.0, thickness_mm=0.0):
    """
    1. Apply (off_x, off_y) translation in real-world mm.
    2. Apply 3D Optical Parallax compensation for timber thickness.
    3. Apply World-to-Projector Homography.
    """
    if points_mm is None or len(points_mm) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.asarray(points_mm, dtype=np.float32).copy()
    if pts.ndim == 1 and len(pts) == 2:
        pts = pts.reshape(1, 2)

    pts[:, 0] += off_x
    pts[:, 1] += off_y

    # Parallax compensation for object height
    if thickness_mm > 0.0 and projector_height_mm > thickness_mm:
        scale_factor = projector_height_mm / (projector_height_mm - thickness_mm)
        pts[:, 0] = projector_pos_x_mm + (pts[:, 0] - projector_pos_x_mm) * scale_factor
        pts[:, 1] = projector_pos_y_mm + (pts[:, 1] - projector_pos_y_mm) * scale_factor

    pts_homo = pts.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts_homo, H_WORLD_TO_PROJECTOR)
    return projected.reshape(-1, 2)


# ============================================================
# FALLBACK LOCAL JSON LOADER
# ============================================================

def load_fallback_timber():
    """Load latest measurement JSON as initial fallback state."""
    pattern = os.path.join(config.CAPTURE_DIR, "timber_*_measurement.json")
    json_files = glob.glob(pattern)
    if json_files:
        json_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = json_files[0]
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "source": os.path.basename(latest_file),
                    "timber_id": data.get("timber_id", 1),
                    "length_mm": data.get("length_mm", 0.0),
                    "width_mm": data.get("width_mm", 0.0),
                    "thickness_mm": float(data.get("timber_thickness_mm", 0.0)),
                    "contour_mm": data.get("contour_mm", []),
                    "corners_mm": data.get("corners_mm", []),
                    "defects": data.get("defects", []),
                    "cut_geo": [],
                    "mill_geo": [],
                    "points_geo": [],
                    "labels": []
                }
        except Exception as e:
            print(f"Error reading fallback JSON: {e}")
    return {
        "source": "None",
        "timber_id": 1,
        "length_mm": 0.0,
        "width_mm": 0.0,
        "thickness_mm": 0.0,
        "contour_mm": [],
        "corners_mm": [],
        "cut_geo": [],
        "mill_geo": [],
        "points_geo": [],
        "labels": []
    }


# ============================================================
# LIVE STATE & THREAD-SAFE STORAGE
# ============================================================

class ProjectorState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_received_time = 0.0
        self.packet_count = 0
        self.client_address = None
        self.is_connected = False
        self.data = load_fallback_timber()
        if "thickness_mm" in self.data and self.data["thickness_mm"] > 0:
            global timber_thickness_mm
            timber_thickness_mm = self.data["thickness_mm"]

    def update_from_json(self, json_data, client_addr):
        with self.lock:
            self.last_received_time = time.time()
            self.packet_count += 1
            self.client_address = client_addr
            self.is_connected = True
            self.data = json_data
            if "thickness_mm" in json_data and json_data["thickness_mm"] is not None:
                global timber_thickness_mm
                # update thickness if provided
                try:
                    val = float(json_data["thickness_mm"])
                    if val >= 0.0:
                        timber_thickness_mm = val
                except (ValueError, TypeError):
                    pass

    def set_disconnected(self):
        with self.lock:
            self.is_connected = False

    def get_snapshot(self):
        with self.lock:
            return dict(self.data), self.is_connected, self.packet_count, self.last_received_time


state = ProjectorState()


# ============================================================
# TCP SOCKET SERVER THREAD
# ============================================================

def socket_server_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((SERVER_HOST, SERVER_PORT))
        server.listen(5)
        print(f"[TCP Server] Listening on port {SERVER_PORT} (Connect from Grasshopper using 127.0.0.1:{SERVER_PORT})...")
    except Exception as e:
        print(f"[TCP Server Error] Failed to bind {SERVER_HOST}:{SERVER_PORT}: {e}")
        return

    while True:
        try:
            client, addr = server.accept()
            print(f"[TCP Server] Connected from Grasshopper client {addr}")
            buffer = ""

            while True:
                chunk = client.recv(65536)
                if not chunk:
                    print(f"[TCP Server] Client {addr} disconnected.")
                    state.set_disconnected()
                    break

                buffer += chunk.decode("utf-8", errors="replace")

                # Handle newline-delimited JSON packets
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        state.update_from_json(payload, addr)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            state.set_disconnected()
            time.sleep(0.5)


# Start TCP server in background
srv_thread = threading.Thread(target=socket_server_thread, daemon=True)
srv_thread.start()


# ============================================================
# RENDER FRAME FUNCTION
# ============================================================

def render_frame(data, is_connected, packet_count, last_time):
    canvas = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # Thicknesses & Sizes (from Grasshopper payload or viewer defaults)
    timber_th_px = int(data.get("timber_thickness_px", data.get("timber_width", DEFAULT_TIMBER_THICKNESS_PX)))
    cut_th_px = int(data.get("cut_thickness_px", data.get("cut_thickness", data.get("cut_width", DEFAULT_CUT_THICKNESS_PX))))
    mill_th_px = int(data.get("mill_thickness_px", data.get("mill_thickness", data.get("mill_width", DEFAULT_MILL_THICKNESS_PX))))
    drill_rad_px = int(data.get("drill_radius_px", data.get("point_radius", data.get("point_size", DEFAULT_DRILL_RADIUS_PX))))
    drill_th_px = int(data.get("drill_thickness_px", data.get("point_thickness", DEFAULT_DRILL_THICKNESS_PX)))
    label_scale = float(data.get("label_scale", data.get("text_size", DEFAULT_LABEL_SCALE)))
    label_th_px = int(data.get("label_thickness", DEFAULT_LABEL_THICKNESS))

    # 1. Base Timber Contour
    contour_raw = data.get("contour_mm") or data.get("timber_contour") or data.get("main_border")
    timber_color_raw = data.get("timber_color", (255, 255, 255))
    timber_bgr = parse_color(timber_color_raw, default_bgr=(255, 255, 255))

    if contour_raw and len(contour_raw) >= 3:
        pts_px = world_to_projector(contour_raw, offset_x_mm, offset_y_mm, timber_thickness_mm)
        pts_int = np.round(pts_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_int], True, timber_bgr, max(1, timber_th_px), cv2.LINE_AA)

    # 2. Corners (if present)
    corners_raw = data.get("corners_mm") or data.get("timber_corners")
    if corners_raw and len(corners_raw) >= 3:
        c_px = world_to_projector(corners_raw, offset_x_mm, offset_y_mm, timber_thickness_mm)
        for i, pt in enumerate(c_px):
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(canvas, (px, py), 4, timber_bgr, -1, cv2.LINE_AA)
            cv2.putText(
                canvas, f"C{i+1}", (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, timber_bgr, 1, cv2.LINE_AA
            )

    # 3. Defect Marks (from scanner)
    defects_raw = data.get("defects", [])
    for d in defects_raw:
        dx = d.get("x_mm", d.get("x", 0))
        dy = d.get("y_mm", d.get("y", 0))
        d_px = world_to_projector([[dx, dy]], offset_x_mm, offset_y_mm, timber_thickness_mm)[0]
        px, py = int(round(d_px[0])), int(round(d_px[1]))
        cv2.drawMarker(canvas, (px, py), (0, 165, 255), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

    # 4. Milling Geometry (Cyan / Blue / Custom)
    mill_geo = data.get("mill_geo") or data.get("mill") or data.get("milling", [])
    mill_color_raw = data.get("mill_color", (0, 220, 255))
    mill_bgr = parse_color(mill_color_raw, default_bgr=(255, 220, 0))

    if isinstance(mill_geo, list):
        for poly in mill_geo:
            if poly and len(poly) >= 2:
                pts_px = world_to_projector(poly, offset_x_mm, offset_y_mm, timber_thickness_mm)
                pts_int = np.round(pts_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts_int], False, mill_bgr, max(1, mill_th_px), cv2.LINE_AA)

    # 5. Cutting Geometry (Red / Custom)
    cut_geo = data.get("cut_geo") or data.get("cut") or data.get("cutting", [])
    cut_color_raw = data.get("cut_color", (255, 30, 30))
    cut_bgr = parse_color(cut_color_raw, default_bgr=(30, 30, 255))

    if isinstance(cut_geo, list):
        for poly in cut_geo:
            if poly and len(poly) >= 2:
                pts_px = world_to_projector(poly, offset_x_mm, offset_y_mm, timber_thickness_mm)
                pts_int = np.round(pts_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts_int], False, cut_bgr, max(1, cut_th_px), cv2.LINE_AA)

    # 6. Drill / Reference Points (Green / Custom)
    points_geo = data.get("points_geo") or data.get("points", [])
    point_color_raw = data.get("point_color", (0, 255, 100))
    pt_bgr = parse_color(point_color_raw, default_bgr=(100, 255, 0))

    if isinstance(points_geo, list):
        for pt in points_geo:
            if pt and len(pt) >= 2:
                p_px = world_to_projector([pt[:2]], offset_x_mm, offset_y_mm, timber_thickness_mm)[0]
                px, py = int(round(p_px[0])), int(round(p_px[1]))
                cv2.circle(canvas, (px, py), max(1, drill_rad_px), pt_bgr, max(1, drill_th_px), cv2.LINE_AA)
                cv2.drawMarker(canvas, (px, py), pt_bgr, cv2.MARKER_CROSS, max(4, drill_rad_px * 2), 1, cv2.LINE_AA)

    # 7. Text Labels & Annotations
    labels_raw = data.get("labels", [])
    label_color_raw = data.get("label_color", (255, 180, 0))
    lbl_bgr = parse_color(label_color_raw, default_bgr=(0, 180, 255))

    if isinstance(labels_raw, list):
        for item in labels_raw:
            if isinstance(item, dict):
                txt = str(item.get("text", ""))
                pos = item.get("pos", [0, 0])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                txt = str(item[0])
                pos = item[1]
            else:
                txt = str(item)
                pos = [200, 200]

            if txt and len(pos) >= 2:
                p_px = world_to_projector([pos[:2]], offset_x_mm, offset_y_mm, timber_thickness_mm)[0]
                px, py = int(round(p_px[0])), int(round(p_px[1]))
                cv2.putText(
                    canvas, txt, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, lbl_bgr, max(1, label_th_px), cv2.LINE_AA
                )

    # 8. On-Screen HUD Bar
    if SHOW_HUD:
        hud_h = 36
        hud_y = PROJECTOR_HEIGHT - hud_h
        cv2.rectangle(canvas, (0, hud_y), (PROJECTOR_WIDTH, PROJECTOR_HEIGHT), DARK_BG, -1)
        cv2.line(canvas, (0, hud_y), (PROJECTOR_WIDTH, hud_y), (60, 60, 60), 1)

        conn_str = f"GH: CONNECTED ({packet_count} frames)" if is_connected else "GH: LISTENING (port 9999)"
        conn_color = (0, 255, 0) if is_connected else (0, 165, 255)

        cv2.putText(canvas, conn_str, (12, hud_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, conn_color, 1, cv2.LINE_AA)

        param_str = (
            f"Thick: {timber_thickness_mm:.1f}mm [T/G] | "
            f"Offset: X={offset_x_mm:+.1f}mm Y={offset_y_mm:+.1f}mm [W/S/A/D]"
        )
        cv2.putText(canvas, param_str, (290, hud_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, cv2.LINE_AA)

        ctrl_str = f"Step:{step_mm:.1f}mm [+/-] | [0] Zero-T | [H] HUD | [ESC] Exit"
        cv2.putText(canvas, ctrl_str, (PROJECTOR_WIDTH - 440, hud_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)

    return canvas


# ============================================================
# PROJECTOR WINDOW CREATION
# ============================================================

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, PROJECTOR_SCREEN_ORIGIN_X, PROJECTOR_SCREEN_ORIGIN_Y)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n" + "=" * 65)
print("eWoodX Real-Time Projector Live Viewer Running")
print(f"  Resolution:      {PROJECTOR_WIDTH} x {PROJECTOR_HEIGHT}")
print(f"  Window Position: X={PROJECTOR_SCREEN_ORIGIN_X}, Y={PROJECTOR_SCREEN_ORIGIN_Y}")
print(f"  TCP Server:      {SERVER_HOST}:{SERVER_PORT}")
print(f"  Base Thickness:  {timber_thickness_mm:.1f} mm")
print("=" * 65)
print("Interactive Controls:")
print("  T / G           : Increase / Decrease Thickness (+/- 1.0 mm)")
print("  Shift+T / G     : Increase / Decrease Thickness (+/- 5.0 mm)")
print("  0 (Zero)        : Reset Thickness to 0.0 mm (Table Plane)")
print("  W / S / A / D   : Nudge Real-World Offsets Y / X (+/- step)")
print("  + / -           : Change Nudge Step (0.1, 0.5, 1.0, 2.0, 5.0 mm)")
print("  R               : Reset Offsets to (0, 0)")
print("  H               : Toggle HUD Bar")
print("  ESC / Q         : Exit Viewer")
print("=" * 65 + "\n")


# ============================================================
# MAIN DISPLAY LOOP
# ============================================================

while True:
    cur_data, is_connected, packet_count, last_time = state.get_snapshot()

    # Check connection timeout (if no packet in 3 seconds, mark disconnected)
    if is_connected and (time.time() - last_time > 3.0):
        state.set_disconnected()
        is_connected = False

    frame = render_frame(cur_data, is_connected, packet_count, last_time)
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKeyEx(16)  # ~60 FPS update rate

    if key in (27, ord('q'), ord('Q')):
        break

    # Thickness adjustments
    elif key == ord('t'):
        timber_thickness_mm += 1.0
        print(f"Thickness: {timber_thickness_mm:.1f} mm")
    elif key == ord('T'):
        timber_thickness_mm += 5.0
        print(f"Thickness: {timber_thickness_mm:.1f} mm")
    elif key == ord('g'):
        timber_thickness_mm = max(0.0, timber_thickness_mm - 1.0)
        print(f"Thickness: {timber_thickness_mm:.1f} mm")
    elif key == ord('G'):
        timber_thickness_mm = max(0.0, timber_thickness_mm - 5.0)
        print(f"Thickness: {timber_thickness_mm:.1f} mm")
    elif key == ord('0'):
        timber_thickness_mm = 0.0
        print("Thickness reset to 0.0 mm (Table level)")

    # Offsets nudging
    elif key in (2490368, ord('w'), ord('W')):
        offset_y_mm += step_mm
    elif key in (2621440, ord('s'), ord('S')):
        offset_y_mm -= step_mm
    elif key in (2555904, ord('d'), ord('D')):
        offset_x_mm += step_mm
    elif key in (2424832, ord('a'), ord('A')):
        offset_x_mm -= step_mm

    # Step size
    elif key in (ord('+'), ord('='), ord(']')):
        steps = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        idx = min(len(steps) - 1, steps.index(step_mm) + 1 if step_mm in steps else 2)
        step_mm = steps[idx]
        print(f"Nudge step: {step_mm:.2f} mm")
    elif key in (ord('-'), ord('_'), ord('[')):
        steps = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        idx = max(0, steps.index(step_mm) - 1 if step_mm in steps else 2)
        step_mm = steps[idx]
        print(f"Nudge step: {step_mm:.2f} mm")

    # Reset
    elif key in (ord('r'), ord('R')):
        offset_x_mm = 0.0
        offset_y_mm = 0.0
        print("Offsets reset to X=0.0 mm, Y=0.0 mm")

    # Toggle HUD
    elif key in (ord('h'), ord('H')):
        SHOW_HUD = not SHOW_HUD

cv2.destroyAllWindows()
