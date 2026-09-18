"""
Central config for the timber scanning rig.
Fill in the CAPS variables for your physical setup. Everything else
reads from here so you only edit numbers in one place.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "calibration_data")
CAMERA_CALIB_FILE = os.path.join(CALIB_DIR, "camera_intrinsics.npz")
HOMOGRAPHY_FILE = os.path.join(CALIB_DIR, "homography.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "sample_output")
# All captured timber images, debug images and measurement JSON files are stored here.
CAPTURE_DIR = os.path.join(OUTPUT_DIR, "captures")

# ---------------------------------------------------------------------------
# Camera Device & Video Stream (Angetube 4K / 120° Wide Angle Webcam)
# ---------------------------------------------------------------------------
CAMERA_INDEX = 1

# Capture resolution (e.g., 3840x2160 4K, 1920x1080 FHD)
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"

# ---------------------------------------------------------------------------
# Camera Hardware Parameters (via duvc_ctl / UVC DirectShow)
# Ranges for Angetube Webcam:
#   Focus: 0 - 1023 (manual focus value, e.g. 450)
#   Digital Zoom: 1.0 (1.0x full 120° FOV) to 4.0 (4.0x zoom)
#   Exposure: -13 to -1 (log2 scale; -5 is standard)
#   Brightness, Contrast, Saturation, Sharpness: 1 to 64 (default 32)
#   Gain: 0 to 15 (default 0)
#   White Balance: 1800 to 10000 K (default 5000)
# ---------------------------------------------------------------------------

# Focus control:
#   CAMERA_FOCUS_MODE: "manual" or "auto"
#   CAMERA_FOCUS_VALUE: Integer value (0 to 1023, e.g. 350, 450, 500)
CAMERA_FOCUS_MODE = "manual"
CAMERA_FOCUS_VALUE = 380

# Zoom control:
#   Float zoom multiplier: 1.0 = full wide-angle (no zoom), 1.5 = 1.5x zoom, 2.0 = 2.0x zoom
CAMERA_ZOOM = 2.00

# Exposure & Light controls:
#   CAMERA_EXPOSURE_MODE: "auto" or "manual"
#   CAMERA_EXPOSURE_VALUE: integer (-13 to -1, e.g. -5)
CAMERA_EXPOSURE_MODE = "auto"
CAMERA_EXPOSURE_VALUE = -5

# Image adjustments:
CAMERA_BRIGHTNESS = 11
CAMERA_CONTRAST = 26
CAMERA_SATURATION = 32      # 1 - 64 (default 32)
CAMERA_SHARPNESS = 32       # 1 - 64 (default 32)
CAMERA_GAIN = 0             # 0 - 15 (default 0)
CAMERA_BACKLIGHT_COMPENSATION = 0

# White Balance:
#   CAMERA_WHITE_BALANCE_MODE: "auto" or "manual"
#   CAMERA_WHITE_BALANCE_TEMPERATURE: Color temp in Kelvin (1800 to 10000, default 5000)
CAMERA_WHITE_BALANCE_MODE = "auto"
CAMERA_WHITE_BALANCE_TEMPERATURE = 5000

# ---------------------------------------------------------------------------
# Lens Calibration Model (Fisheye vs Standard Pinhole)
# ---------------------------------------------------------------------------
# For 120°+ wide-angle lenses, "fisheye" (cv2.fisheye) is strongly recommended.
CALIBRATION_MODEL = "fisheye"  # "fisheye" or "standard"

# Fisheye Undistortion Parameters:
#   FISHEYE_BALANCE: 0.0 retains only valid pixels (no black edges / cropped),
#                    1.0 retains all pixels including corners (with black edges).
#   FISHEYE_FOV_SCALE: scale factor for output FOV (1.0 = normal).
FISHEYE_BALANCE = 0.0
FISHEYE_FOV_SCALE = 1.0

# Fisheye Calibration Optimization Flags
FISHEYE_CHECK_COND = True
FISHEYE_RECOMPUTE_EXTRINSIC = True
FISHEYE_FIX_SKEW = True

# ---------------------------------------------------------------------------
# Checkerboard used for intrinsic (lens distortion) calibration
# ---------------------------------------------------------------------------
# INNER corners, not squares. A "14x9 squares" board has 13x8 inner corners.
CHECKERBOARD_INNER_CORNERS = (13, 8)
CHECKERBOARD_SQUARE_SIZE_MM = 20.0  # measure your printed squares with calipers

# ---------------------------------------------------------------------------
# Workspace / homography (ArUco markers at the 4 known table corners)
# ---------------------------------------------------------------------------
ARUCO_DICT = "DICT_4X4_50"
ARUCO_MARKER_SIZE_MM = 100.0  # physical printed size of each marker's black square

# Marker ID -> real-world (X, Y) position in millimetres on your table plane.
# Standard CAD / Rhino / Grasshopper coordinate system:
# - Origin (0,0) at Bottom-Left (Marker 1)
# - +X axis along table length towards Bottom-Right (Marker 0)
# - +Y axis along table width towards Top-Left (Marker 2)
# - (+X, +Y) at Top-Right (Marker 3)
# Table workspace physical dimensions in mm
TABLE_WIDTH_MM = 1780.0
TABLE_HEIGHT_MM = 1040.0

MARKER_WORLD_POSITIONS_MM = {
    1: (0.0, 0.0),                            # bottom-left (Origin)
    0: (TABLE_WIDTH_MM, 0.0),                 # bottom-right (+X)
    2: (0.0, TABLE_HEIGHT_MM),                # top-left (+Y)
    3: (TABLE_WIDTH_MM, TABLE_HEIGHT_MM),     # top-right (+X, +Y)
}

# ---------------------------------------------------------------------------
# Camera Setup & Parallax Compensation
# ---------------------------------------------------------------------------
# Height from camera lens to table surface in mm.
# Update this value if you adjust camera mounting height.
CAMERA_HEIGHT_MM = 2430.0



# ---------------------------------------------------------------------------
# Segmentation (board vs background)
# ---------------------------------------------------------------------------
# Simple approach: background is a known, roughly uniform color (e.g. a dark
# mat). We threshold in HSV. Tune these ranges to your actual mat color by
# running measure_board.py with --debug and looking at the mask.
BACKGROUND_HSV_LOWER = (0, 0, 0)
BACKGROUND_HSV_UPPER = (180, 15, 255)  # tune this: covers dark, low-saturation mat

MIN_BOARD_CONTOUR_AREA_PX = 5000  # ignore small noise blobs after thresholding

# ---------------------------------------------------------------------------
# Contour Approximation (Smoothing & Simplification)
# ---------------------------------------------------------------------------
# Epsilon = CONTOUR_APPROX_FACTOR * arcLength (OpenCV cv2.approxPolyDP)
# 0.001 smoothly simplifies contour and removes pixel noise while keeping organic curves.
CONTOUR_APPROX_FACTOR = 0.004

# ---------------------------------------------------------------------------
# Projector 3D Mounting & Height (Thickness / Parallax Compensation)
# ---------------------------------------------------------------------------
# Height from projector lens to table surface in mm (235 cm = 2350 mm)
PROJECTOR_HEIGHT_MM = 2300.0

# -------------------------------------------------
# Projector display settings (pixels) – can be overridden via CLI or environment
PROJECTOR_WIDTH = 1280
PROJECTOR_HEIGHT = 800
PROJECTOR_SCREEN_ORIGIN_X = 2560
PROJECTOR_SCREEN_ORIGIN_Y = 0
# Environment-specific overrides
import os
_EWOODX_ENV = os.getenv('EWOODX_ENV', 'development')
if _EWOODX_ENV == 'development':
    # Development defaults
    PROJECTOR_WIDTH = 1280
    PROJECTOR_HEIGHT = 800
elif _EWOODX_ENV == 'production':
    # Production defaults (example values)
    PROJECTOR_WIDTH = 1920
    PROJECTOR_HEIGHT = 1080
# CLI overrides
def _override_from_cli():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--proj_width', type=int)
    parser.add_argument('--proj_height', type=int)
    parser.add_argument('--proj_x', type=int)
    parser.add_argument('--proj_y', type=int)
    args, unknown = parser.parse_known_args()
    if args.proj_width is not None:
        globals()['PROJECTOR_WIDTH'] = args.proj_width
    if args.proj_height is not None:
        globals()['PROJECTOR_HEIGHT'] = args.proj_height
    if args.proj_x is not None:
        globals()['PROJECTOR_SCREEN_ORIGIN_X'] = args.proj_x
    if args.proj_y is not None:
        globals()['PROJECTOR_SCREEN_ORIGIN_Y'] = args.proj_y
_override_from_cli()
# Clean up helper
if '_override_from_cli' in globals():
    del _override_from_cli

# 3D Position of projector lens relative to table Origin (Marker 1 at 0,0) in mm
PROJECTOR_POS_X_MM = 920.0
PROJECTOR_POS_Y_MM = 50.0

# ---------------------------------------------------------------------------
# Projector Fine-Tuning Offsets (in real-world mm)
# ---------------------------------------------------------------------------
# Base fine adjustments at table level (Z=0)
PROJECTOR_OFFSET_X_MM = 0.0
PROJECTOR_OFFSET_Y_MM = 0.0

# Timber thickness in mm (default starting thickness for measurements and projection)
TIMBER_THICKNESS_MM = 0.0
