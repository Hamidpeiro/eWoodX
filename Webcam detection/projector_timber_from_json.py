import cv2
import numpy as np
import json
import os
import glob
import sys
from pathlib import Path
import config


# ============================================================
# eWoodX - Project Detected Timber from JSON
#
# Features:
#   - Automatic Loading of the Latest Captured Timber JSON
#   - Real-time Timber Thickness Adjustment (T/G keys, 0 key to zero)
#   - Real-time 3D Projector Height & Parallax Angle Compensation
#   - Real-time Keyboard Nudging (W/A/S/D or Arrow Keys)
#   - Visual On-Screen HUD with Active Parameters
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

def get_latest_json():
    """Find the most recent timber measurement JSON in captures directory."""
    pattern = os.path.join(config.CAPTURE_DIR, "timber_*_measurement.json")
    json_files = glob.glob(pattern)
    if json_files:
        json_files.sort(key=os.path.getmtime, reverse=True)
        return Path(json_files[0])
    return Path(os.path.join(config.CAPTURE_DIR, "timber_21_measurement.json"))

# If a JSON path is passed as command line argument, use it; otherwise get latest
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    JSON_PATH = Path(sys.argv[1])
else:
    JSON_PATH = get_latest_json()

PROJECTOR_HOMOGRAPHY_PATH = Path(
    os.path.join(config.CALIB_DIR, "projector_homography.npz")
)


# ============================================================
# PROJECTOR RESOLUTION & POSITION
# ============================================================

PROJECTOR_WIDTH = config.PROJECTOR_WIDTH
PROJECTOR_HEIGHT = config.PROJECTOR_HEIGHT

# Screen origin (offset on desktop display arrangement)
PROJECTOR_SCREEN_ORIGIN_X = config.PROJECTOR_SCREEN_ORIGIN_X
PROJECTOR_SCREEN_ORIGIN_Y = config.PROJECTOR_SCREEN_ORIGIN_Y

WINDOW_NAME = "eWoodX Timber Projection"


# ============================================================
# 3D PROJECTOR MOUNTING & PARALLAX PARAMETERS
# ============================================================

# Height of projector optical lens above table (in mm) - 235 cm
projector_height_mm = float(getattr(config, "PROJECTOR_HEIGHT_MM", 2298.0))

# 3D Horizontal position of projector lens relative to table Origin (Marker 1 at 0,0)
projector_pos_x_mm = float(getattr(config, "PROJECTOR_POS_X_MM", 920.0))
projector_pos_y_mm = float(getattr(config, "PROJECTOR_POS_Y_MM", 50.0))


# ============================================================
# DISPLAY & OFFSET SETTINGS
# ============================================================

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
GRAY = (150, 150, 150)
DARK_BG = (20, 20, 20)

CONTOUR_THICKNESS = 2

SHOW_CORNERS = True
SHOW_INFO = True
SHOW_HUD = True

# Offsets loaded from config.py (in real-world mm)
offset_x_mm = float(getattr(config, "PROJECTOR_OFFSET_X_MM", 0.0))
offset_y_mm = float(getattr(config, "PROJECTOR_OFFSET_Y_MM", 0.0))
step_mm = 0.5  # adjustment step per keypress (in mm)


# ============================================================
# CHECK FILES
# ============================================================

if not JSON_PATH.exists():
    raise FileNotFoundError(
        f"JSON file not found:\n{JSON_PATH}\nPlease measure a timber first with 3_measure_board_black_Background.py"
    )

if not PROJECTOR_HOMOGRAPHY_PATH.exists():
    raise FileNotFoundError(
        f"Projector calibration not found:\n{PROJECTOR_HOMOGRAPHY_PATH}\n"
        "Please run calibrate_projector_corners.py first."
    )


# ============================================================
# LOAD TIMBER JSON
# ============================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    timber = json.load(f)

timber_id = timber.get("timber_id", 1)
length_mm = timber.get("length_mm", 0.0)
width_mm = timber.get("width_mm", 0.0)

# Initial thickness from JSON or config
timber_thickness_mm = float(timber.get("timber_thickness_mm", getattr(config, "TIMBER_THICKNESS_MM", 0.0)))

contour_mm_base = np.array(timber["contour_mm"], dtype=np.float32)
corners_mm_base = np.array(timber["corners_mm"], dtype=np.float32)
defects = timber.get("defects", [])

print()
print("============================================================")
print("eWoodX Timber Projection (with 3D Thickness Parallax)")
print("============================================================")
print(f"Loaded JSON:        {JSON_PATH.name}")
print(f"Timber ID:          {timber_id}")
print(f"Dimensions:         {length_mm:.1f} x {width_mm:.1f} mm")
print(f"Timber Thickness:   {timber_thickness_mm:.1f} mm")
print(f"Projector Height:   {projector_height_mm:.1f} mm")
print(f"Projector Mount:    X={projector_pos_x_mm:.1f} mm, Y={projector_pos_y_mm:.1f} mm")
print(f"Base Offset:        X={offset_x_mm:+.2f} mm, Y={offset_y_mm:+.2f} mm")
print("============================================================")


# ============================================================
# LOAD PROJECTOR HOMOGRAPHY
# ============================================================

projector_data = np.load(PROJECTOR_HOMOGRAPHY_PATH, allow_pickle=True)
H_WORLD_TO_PROJECTOR = projector_data["H_world_to_projector"]


# ============================================================
# WORLD MM -> PROJECTOR PIXEL TRANSFORM WITH 3D PARALLAX
# ============================================================

def world_to_projector(points_mm, off_x=0.0, off_y=0.0, thickness_mm=0.0):
    """
    1. Apply (off_x, off_y) translation in real-world mm.
    2. Apply 3D Optical Parallax compensation for timber thickness:
       Rays projected from (X_proj, Y_proj, H_proj) hitting the top surface (Z = thickness_mm)
       land on the table calibration plane (Z = 0) at:
       X_table = X_proj + (X - X_proj) * (H_proj / (H_proj - thickness))
       Y_table = Y_proj + (Y - Y_proj) * (H_proj / (H_proj - thickness))
    3. Apply World-to-Projector Homography.
    """
    pts = np.asarray(points_mm, dtype=np.float32).copy()
    pts[:, 0] += off_x
    pts[:, 1] += off_y

    # Parallax compensation for object height
    if thickness_mm > 0.0 and projector_height_mm > thickness_mm:
        scale_factor = projector_height_mm / (projector_height_mm - thickness_mm)
        pts[:, 0] = projector_pos_x_mm + (pts[:, 0] - projector_pos_x_mm) * scale_factor
        pts[:, 1] = projector_pos_y_mm + (pts[:, 1] - projector_pos_y_mm) * scale_factor

    pts = pts.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H_WORLD_TO_PROJECTOR)
    return projected.reshape(-1, 2)


# ============================================================
# RENDER CANVAS
# ============================================================

def render_projection():
    canvas = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # Transform contour and corners with current thickness & offsets
    contour_px = world_to_projector(contour_mm_base, offset_x_mm, offset_y_mm, timber_thickness_mm)
    corners_px = world_to_projector(corners_mm_base, offset_x_mm, offset_y_mm, timber_thickness_mm)

    contour_int = np.round(contour_px).astype(np.int32).reshape(-1, 1, 2)

    # 1. Draw contour line
    cv2.polylines(canvas, [contour_int], True, WHITE, CONTOUR_THICKNESS, cv2.LINE_AA)

    # 2. Draw corners
    if SHOW_CORNERS:
        for i, point in enumerate(corners_px):
            px, py = int(round(point[0])), int(round(point[1]))
            cv2.circle(canvas, (px, py), 4, WHITE, -1, cv2.LINE_AA)
            cv2.putText(
                canvas, f"C{i+1}", (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA
            )

    # 3. Draw center label & dimensions
    if SHOW_INFO:
        center_mm = np.mean(corners_mm_base, axis=0)
        center_px = world_to_projector([center_mm], offset_x_mm, offset_y_mm, timber_thickness_mm)[0]
        c_x, c_y = int(round(center_px[0])), int(round(center_px[1]))

        cv2.putText(
            canvas, f"TIMBER {timber_id}", (c_x - 60, c_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, f"{length_mm:.1f} x {width_mm:.1f} mm  (T: {timber_thickness_mm:.0f}mm)", (c_x - 85, c_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA
        )

    # 4. On-Screen HUD (Thickness, Offset & Controls Status)
    if SHOW_HUD:
        hud_bar_y = PROJECTOR_HEIGHT - 35
        # Draw background bar for clear readability
        cv2.rectangle(canvas, (0, hud_bar_y - 5), (PROJECTOR_WIDTH, PROJECTOR_HEIGHT), DARK_BG, -1)
        cv2.line(canvas, (0, hud_bar_y - 5), (PROJECTOR_WIDTH, hud_bar_y - 5), (50, 50, 50), 1)

        status_left = (
            f"Timber: {timber_id} | Thick: {timber_thickness_mm:.1f}mm [T/G] | "
            f"Offset: X={offset_x_mm:+.1f}mm Y={offset_y_mm:+.1f}mm [W/S/A/D]"
        )
        status_right = f"[+/-] Step:{step_mm:.1f}mm | [0] Zero-T | [P] Save | [ESC] Exit"

        cv2.putText(
            canvas, status_left, (15, hud_bar_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, status_right, (PROJECTOR_WIDTH - 480, hud_bar_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY, 1, cv2.LINE_AA
        )

    return canvas


# ============================================================
# PROJECTOR WINDOW
# ============================================================

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, PROJECTOR_SCREEN_ORIGIN_X, PROJECTOR_SCREEN_ORIGIN_Y)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ============================================================
# INTERACTIVE DISPLAY LOOP
# ============================================================

print()
print("Interactive Controls:")
print("  T / Shift+T     : Increase thickness (+1.0 mm / +5.0 mm)")
print("  G / Shift+G     : Decrease thickness (-1.0 mm / -5.0 mm)")
print("  0 (Zero)        : Reset thickness to 0.0 mm (table level)")
print("  W / Up Arrow    : Nudge +Y (+step mm)")
print("  S / Down Arrow  : Nudge -Y (-step mm)")
print("  D / Right Arrow : Nudge +X (+step mm)")
print("  A / Left Arrow  : Nudge -X (-step mm)")
print("  + / -           : Change nudge step size (0.1, 0.5, 1.0, 2.0, 5.0 mm)")
print("  R               : Reset offsets to (0, 0)")
print("  P               : Print current parameters to console")
print("  I / C / H       : Toggle Info / Corners / HUD")
print("  ESC / Q         : Exit")
print()

while True:
    frame = render_projection()
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKeyEx(30)

    if key in (27, ord('q'), ord('Q')):  # ESC or Q
        break

    # --- Thickness Adjustments ---
    elif key == ord('t'):
        timber_thickness_mm += 1.0
        print(f"Timber thickness: {timber_thickness_mm:.1f} mm")
    elif key == ord('T'):
        timber_thickness_mm += 5.0
        print(f"Timber thickness: {timber_thickness_mm:.1f} mm")
    elif key == ord('g'):
        timber_thickness_mm = max(0.0, timber_thickness_mm - 1.0)
        print(f"Timber thickness: {timber_thickness_mm:.1f} mm")
    elif key == ord('G'):
        timber_thickness_mm = max(0.0, timber_thickness_mm - 5.0)
        print(f"Timber thickness: {timber_thickness_mm:.1f} mm")
    elif key == ord('0'):
        timber_thickness_mm = 0.0
        print("Timber thickness set to 0.0 mm (Table Plane)")

    # --- Nudge Y-axis ---
    elif key in (2490368, ord('w'), ord('W')):
        offset_y_mm += step_mm
        print(f"Offset Y: {offset_y_mm:+.2f} mm (X: {offset_x_mm:+.2f} mm)")
    elif key in (2621440, ord('s'), ord('S')):
        offset_y_mm -= step_mm
        print(f"Offset Y: {offset_y_mm:+.2f} mm (X: {offset_x_mm:+.2f} mm)")

    # --- Nudge X-axis ---
    elif key in (2555904, ord('d'), ord('D')):
        offset_x_mm += step_mm
        print(f"Offset X: {offset_x_mm:+.2f} mm (Y: {offset_y_mm:+.2f} mm)")
    elif key in (2424832, ord('a'), ord('A')):
        offset_x_mm -= step_mm
        print(f"Offset X: {offset_x_mm:+.2f} mm (Y: {offset_y_mm:+.2f} mm)")

    # --- Step Size Controls ---
    elif key in (ord('+'), ord('='), ord(']')):
        steps = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        idx = min(len(steps) - 1, steps.index(step_mm) + 1 if step_mm in steps else 2)
        step_mm = steps[idx]
        print(f"Nudge step set to: {step_mm:.2f} mm")
    elif key in (ord('-'), ord('_'), ord('[')):
        steps = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        idx = max(0, steps.index(step_mm) - 1 if step_mm in steps else 2)
        step_mm = steps[idx]
        print(f"Nudge step set to: {step_mm:.2f} mm")

    # --- Reset ---
    elif key in (ord('r'), ord('R')):
        offset_x_mm = 0.0
        offset_y_mm = 0.0
        print("Offsets reset to X=0.0 mm, Y=0.0 mm")

    # --- Print Parameters ---
    elif key in (ord('p'), ord('P')):
        print("\n" + "=" * 50)
        print("CURRENT PROJECTION PARAMETERS:")
        print(f"timber_thickness_mm   = {timber_thickness_mm:.1f} mm")
        print(f"PROJECTOR_OFFSET_X_MM = {offset_x_mm:.2f}")
        print(f"PROJECTOR_OFFSET_Y_MM = {offset_y_mm:.2f}")
        print("=" * 50 + "\n")

    # --- Toggles ---
    elif key in (ord('i'), ord('I')):
        SHOW_INFO = not SHOW_INFO
    elif key in (ord('c'), ord('C')):
        SHOW_CORNERS = not SHOW_CORNERS
    elif key in (ord('h'), ord('H')):
        SHOW_HUD = not SHOW_HUD

cv2.destroyAllWindows()

print()
print("Final Parameters:")
print(f"  Thickness             = {timber_thickness_mm:.1f} mm")
print(f"  PROJECTOR_OFFSET_X_MM = {offset_x_mm:.2f}")
print(f"  PROJECTOR_OFFSET_Y_MM = {offset_y_mm:.2f}")