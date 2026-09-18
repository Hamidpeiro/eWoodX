import cv2
import numpy as np
import time
import os
from pathlib import Path
import config
import camera_utils
import cv2.aruco as aruco


# ============================================================
# eWoodX - Automatic Projector Calibration
#
# Pipeline:
#
# Projector pixel
#       ↓
# projected bright dot
#       ↓
# webcam detection (zoomed & undistorted to match workspace setup)
#       ↓
# camera pixel
#       ↓
# existing camera -> world homography
#       ↓
# world coordinate [mm]
#
# After enough correspondences:
#
# world [mm] -> projector [px]
#
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

CAMERA_CALIBRATION_PATH = Path(config.CAMERA_CALIB_FILE)
WORKSPACE_HOMOGRAPHY_PATH = Path(config.HOMOGRAPHY_FILE)

OUTPUT_PATH = Path(
    os.path.join(config.CALIB_DIR, "projector_homography.npz")
)

REPORT_PATH = Path(
    os.path.join(config.CALIB_DIR, "projector_homography.txt")
)


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 1)
CAMERA_WIDTH = getattr(config, "IMAGE_WIDTH", 3840)
CAMERA_HEIGHT = getattr(config, "IMAGE_HEIGHT", 2160)
CAMERA_FOURCC = getattr(config, "CAMERA_FOURCC", "MJPG")
CAMERA_ZOOM = float(getattr(config, "CAMERA_ZOOM", 1.0))


# ============================================================
# PROJECTOR SETTINGS
# ============================================================

PROJECTOR_WIDTH = config.PROJECTOR_WIDTH
PROJECTOR_HEIGHT = config.PROJECTOR_HEIGHT

# Screen origin (offset on desktop display arrangement)
PROJECTOR_SCREEN_ORIGIN_X = config.PROJECTOR_SCREEN_ORIGIN_X
PROJECTOR_SCREEN_ORIGIN_Y = config.PROJECTOR_SCREEN_ORIGIN_Y

PROJECTOR_WINDOW = "eWoodX Projector Calibration"


# ============================================================
# DEBUG WINDOWS
# ============================================================

DEBUG_CAMERA_WINDOW = "Camera Dot Detection"
DEBUG_THRESHOLD_WINDOW = "Difference Threshold"


# ============================================================
# PROJECTOR CALIBRATION GRID
# ============================================================
#
# 5 columns x 4 rows = 20 projector points
# Computed dynamically from PROJECTOR_WIDTH and PROJECTOR_HEIGHT (10% to 90% margin)
# ============================================================

PROJECTOR_X_POINTS = np.linspace(int(PROJECTOR_WIDTH * 0.10), int(PROJECTOR_WIDTH * 0.90), 5).astype(int).tolist()
PROJECTOR_Y_POINTS = np.linspace(int(PROJECTOR_HEIGHT * 0.15), int(PROJECTOR_HEIGHT * 0.85), 4).astype(int).tolist()

PROJECTOR_POINTS = np.array(
    [
        [x, y]
        for y in PROJECTOR_Y_POINTS
        for x in PROJECTOR_X_POINTS
    ],
    dtype=np.float32
)


# ============================================================
# PROJECTED DOT SETTINGS

# -----------------------------------------------------------
# ARUCO MARKER DETECTION SETTINGS
# -----------------------------------------------------------
_dict_id = getattr(cv2.aruco, getattr(config, "ARUCO_DICT", "DICT_4X4_50")) if isinstance(getattr(config, "ARUCO_DICT", None), str) else cv2.aruco.DICT_4X4_50
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(_dict_id)
ARUCO_PARAMS = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create()


# ============================================================

DOT_RADIUS = 25

# Time after changing projector image
SETTLE_TIME = 0.45

# Discard stale webcam frames
DISCARD_FRAMES = 5


# ============================================================
# DOT DETECTION SETTINGS
# ============================================================

BRIGHTNESS_THRESHOLD = 40

MIN_BLOB_AREA = 40
MAX_BLOB_AREA = 8000

MIN_CIRCULARITY = 0.25


# ============================================================
# WORKSPACE SIZE
# Defined by your existing ArUco calibration
# ============================================================

WORLD_WIDTH_MM = getattr(config, "TABLE_WIDTH_MM", 1780.0)
WORLD_HEIGHT_MM = getattr(config, "TABLE_HEIGHT_MM", 1040.0)

WORLD_MARGIN_MM = 100.0


# ============================================================
# LOAD CALIBRATION FILES
# ============================================================

print()
print("==========================================")
print("Loading existing eWoodX calibration")
print("==========================================")
print()


if not CAMERA_CALIBRATION_PATH.exists():
    raise FileNotFoundError(
        f"Camera calibration not found:\n"
        f"{CAMERA_CALIBRATION_PATH}"
    )


if not WORKSPACE_HOMOGRAPHY_PATH.exists():
    raise FileNotFoundError(
        f"Workspace homography not found:\n"
        f"{WORKSPACE_HOMOGRAPHY_PATH}"
    )


camera_data = np.load(
    CAMERA_CALIBRATION_PATH,
    allow_pickle=True
)

workspace_data = np.load(
    WORKSPACE_HOMOGRAPHY_PATH,
    allow_pickle=True
)


# ------------------------------------------------------------
# Camera calibration
# ------------------------------------------------------------

K = camera_data["camera_matrix"]
D = camera_data["dist_coeffs"]
CALIBRATION_IMAGE_SIZE = camera_data["image_size"]


# ------------------------------------------------------------
# Existing camera -> world homography
# ------------------------------------------------------------

H_CAMERA_TO_WORLD = workspace_data["H"]


print("Camera matrix:")
print(K)
print()

print("Camera distortion:")
print(D)
print()

print(
    "Calibration image size:",
    CALIBRATION_IMAGE_SIZE
)
print()

print("Camera -> World homography:")
print(H_CAMERA_TO_WORLD)
print()


# ============================================================
# OPEN CAMERA
# ============================================================

camera, controller = camera_utils.open_configured_camera()

actual_width = int(
    camera.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

actual_height = int(
    camera.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


print(
    f"Camera opened at: "
    f"{actual_width} x {actual_height} (Zoom: {CAMERA_ZOOM}x)"
)
print()


# ============================================================
# CREATE PROJECTOR WINDOW
# ============================================================

cv2.namedWindow(
    PROJECTOR_WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.moveWindow(
    PROJECTOR_WINDOW,
    PROJECTOR_SCREEN_ORIGIN_X,
    PROJECTOR_SCREEN_ORIGIN_Y
)

cv2.setWindowProperty(
    PROJECTOR_WINDOW,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# PROJECTOR DRAWING
# ============================================================

def create_black_canvas():

    return np.zeros(
        (
            PROJECTOR_HEIGHT,
            PROJECTOR_WIDTH,
            3
        ),
        dtype=np.uint8
    )


def show_black():

    canvas = create_black_canvas()

    cv2.imshow(
        PROJECTOR_WINDOW,
        canvas
    )

    cv2.waitKey(1)


def show_dot(x, y):

    canvas = create_black_canvas()

    cv2.circle(
        canvas,
        (
            int(x),
            int(y)
        ),
        DOT_RADIUS,
        (255, 255, 255),
        -1,
        cv2.LINE_AA
    )

    cv2.imshow(
        PROJECTOR_WINDOW,
        canvas
    )

    cv2.waitKey(1)


# ============================================================
# CAMERA CAPTURE (Zoomed & Undistorted to match workspace)
# ============================================================

def capture_frame():

    # Discard buffered frames
    for _ in range(
        DISCARD_FRAMES
    ):
        camera.read()

    success, raw_frame = camera.read()

    if not success:
        raise RuntimeError(
            "Could not capture webcam frame."
        )

    # 1. Apply digital zoom to match 2_setup_workspace and 3_measure_board
    frame_zoomed = camera_utils.apply_digital_zoom(
        raw_frame,
        CAMERA_ZOOM
    )

    # 2. Undistort using lens calibration
    frame_undist = camera_utils.undistort(
        frame_zoomed
    )

    return frame_undist


# ============================================================
# CAMERA PIXEL -> WORLD MM
# ============================================================

def camera_to_world(
    x_px,
    y_px
):

    point = np.array(
        [
            [
                [
                    x_px,
                    y_px
                ]
            ]
        ],
        dtype=np.float32
    )

    world = cv2.perspectiveTransform(
        point,
        H_CAMERA_TO_WORLD
    )

    x_mm = float(
        world[0, 0, 0]
    )

    y_mm = float(
        world[0, 0, 1]
    )

    return x_mm, y_mm


# ============================================================
# DETECT PROJECTED DOT
# ============================================================

def detect_projected_dot(
    background_frame,
    projected_frame
):

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    background_gray = cv2.cvtColor(
        background_frame,
        cv2.COLOR_BGR2GRAY
    )

    projected_gray = cv2.cvtColor(
        projected_frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # POSITIVE DIFFERENCE ONLY
    #
    # We only care about regions that became brighter.
    #
    # This avoids large false detections caused by the
    # camera exposure making other areas darker.
    # --------------------------------------------------------

    difference = cv2.subtract(
        projected_gray,
        background_gray
    )


    # --------------------------------------------------------
    # Smooth small sensor noise
    # --------------------------------------------------------

    difference_blurred = cv2.GaussianBlur(
        difference,
        (7, 7),
        0
    )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    _, threshold = cv2.threshold(
        difference_blurred,
        BRIGHTNESS_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )


    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------------
    # Find blobs
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        # Reject very small noise
        if area < MIN_BLOB_AREA:
            continue


        # Reject huge exposure-change regions
        if area > MAX_BLOB_AREA:
            continue


        perimeter = cv2.arcLength(
            contour,
            True
        )


        if perimeter <= 0:
            continue


        circularity = (
            4.0
            * np.pi
            * area
            / (
                perimeter
                * perimeter
            )
        )


        # Projected dot may distort slightly,
        # so circularity threshold is permissive.
        if circularity < MIN_CIRCULARITY:
            continue


        moments = cv2.moments(
            contour
        )


        if moments["m00"] == 0:
            continue


        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )


        # ----------------------------------------------------
        # Mean positive brightness increase in this blob
        # ----------------------------------------------------

        blob_mask = np.zeros_like(
            difference
        )

        cv2.drawContours(
            blob_mask,
            [contour],
            -1,
            255,
            -1
        )


        mean_brightness = cv2.mean(
            difference,
            mask=blob_mask
        )[0]


        candidates.append(
            {
                "brightness": mean_brightness,
                "circularity": circularity,
                "area": area,
                "x": cx,
                "y": cy
            }
        )


    # --------------------------------------------------------
    # Nothing valid found
    # --------------------------------------------------------

    if len(candidates) == 0:

        return (
            None,
            threshold,
            difference
        )


    # --------------------------------------------------------
    # Strongest positive brightness change wins
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item:
        item["brightness"],
        reverse=True
    )

    best = candidates[0]


    print(
        "   Candidate:"
        f" brightness={best['brightness']:.1f}"
        f" area={best['area']:.1f}"
        f" circularity={best['circularity']:.2f}"
    )


    return (
        (
            best["x"],
            best["y"]
        ),
        threshold,
        difference
    )


# ============================================================
# DEBUG VISUALIZATION
# ============================================================

def show_debug(
    projected_frame,
    threshold,
    dot_position,
    projector_point
):

    debug_frame = projected_frame.copy()


    if dot_position is not None:

        x = int(
            round(
                dot_position[0]
            )
        )

        y = int(
            round(
                dot_position[1]
            )
        )


        # Red circle around detected dot
        cv2.circle(
            debug_frame,
            (x, y),
            35,
            (0, 0, 255),
            4
        )


        cv2.putText(
            debug_frame,
            "DETECTED DOT",
            (
                x + 45,
                y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Show which projector pixel was active
    # --------------------------------------------------------

    label = (
        f"Projector: "
        f"{int(projector_point[0])}, "
        f"{int(projector_point[1])}"
    )


    cv2.putText(
        debug_frame,
        label,
        (60, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Downscale for normal monitor
    # --------------------------------------------------------

    debug_small = cv2.resize(
        debug_frame,
        (
            1280,
            720
        )
    )

    threshold_small = cv2.resize(
        threshold,
        (
            1280,
            720
        )
    )


    cv2.imshow(
        DEBUG_CAMERA_WINDOW,
        debug_small
    )

    cv2.imshow(
        DEBUG_THRESHOLD_WINDOW,
        threshold_small
    )

    cv2.waitKey(100)


# ============================================================
# STORAGE
# ============================================================

detected_world_points = []

valid_projector_points = []

detected_camera_points = []


# ============================================================
# START CALIBRATION
# ============================================================

print()
print("==========================================")
print("eWoodX Automatic Projector Calibration")
print("==========================================")
print()

print(
    f"Number of projector points: "
    f"{len(PROJECTOR_POINTS)}"
)

print()


# ============================================================
# BLACK INITIALIZATION
# ============================================================

show_black()

time.sleep(1.0)


# ============================================================
# LOOP THROUGH PROJECTOR POINTS
# ============================================================

for index, projector_point in enumerate(
    PROJECTOR_POINTS,
    start=1
):

    projector_x = int(
        projector_point[0]
    )

    projector_y = int(
        projector_point[1]
    )


    print()
    print(
        f"[{index:02d}/"
        f"{len(PROJECTOR_POINTS)}]"
    )

    print(
        f"   Projector pixel: "
        f"({projector_x}, "
        f"{projector_y})"
    )


    # ========================================================
    # STEP 1 - BLACK BACKGROUND
    # ========================================================

    show_black()

    time.sleep(
        SETTLE_TIME
    )

    background_frame = capture_frame()


    # ========================================================
    # STEP 2 - PROJECT WHITE DOT
    # ========================================================

    show_dot(
        projector_x,
        projector_y
    )

    time.sleep(
        SETTLE_TIME
    )

    projected_frame = capture_frame()


    # ========================================================
    # STEP 3 - FIND PROJECTED DOT
    # ========================================================

    (
        dot_position,
        threshold,
        difference
    ) = detect_projected_dot(
        background_frame,
        projected_frame
    )


    # ========================================================
    # DEBUG WINDOWS
    # ========================================================

    show_debug(
        projected_frame,
        threshold,
        dot_position,
        projector_point
    )


    # ========================================================
    # DOT NOT FOUND
    # ========================================================

    if dot_position is None:

        print(
            "   FAILED: "
            "projected dot not detected."
        )

        continue


    camera_x = float(
        dot_position[0]
    )

    camera_y = float(
        dot_position[1]
    )


    # ========================================================
    # CAMERA -> WORLD
    # ========================================================

    world_x, world_y = camera_to_world(
        camera_x,
        camera_y
    )


    print(
        f"   Camera pixel: "
        f"({camera_x:.1f}, "
        f"{camera_y:.1f})"
    )

    print(
        f"   World mm: "
        f"({world_x:.1f}, "
        f"{world_y:.1f})"
    )


    # ========================================================
    # WORLD BOUNDS VALIDATION
    # ========================================================

    if not (
        -WORLD_MARGIN_MM
        <= world_x
        <= WORLD_WIDTH_MM
        + WORLD_MARGIN_MM
    ):

        print(
            "   FAILED: "
            "X coordinate outside workspace."
        )

        continue


    if not (
        -WORLD_MARGIN_MM
        <= world_y
        <= WORLD_HEIGHT_MM
        + WORLD_MARGIN_MM
    ):

        print(
            "   FAILED: "
            "Y coordinate outside workspace."
        )

        continue


    # ========================================================
    # STORE VALID PAIR
    # ========================================================

    detected_camera_points.append(
        [
            camera_x,
            camera_y
        ]
    )


    detected_world_points.append(
        [
            world_x,
            world_y
        ]
    )


    valid_projector_points.append(
        [
            projector_x,
            projector_y
        ]
    )


# ============================================================
# PROJECT BLACK AT END
# ============================================================

show_black()


# ============================================================
# CONVERT TO NUMPY
# ============================================================

camera_points = np.array(
    detected_camera_points,
    dtype=np.float32
)

world_points = np.array(
    detected_world_points,
    dtype=np.float32
)

projector_points = np.array(
    valid_projector_points,
    dtype=np.float32
)


print()
print("==========================================")
print("Calibration collection finished")
print("==========================================")

print(
    f"Valid detections: "
    f"{len(world_points)} / "
    f"{len(PROJECTOR_POINTS)}"
)

print()


# ============================================================
# REQUIRE AT LEAST 4 POINTS
# ============================================================

if len(world_points) < 4:

    camera.release()

    cv2.destroyAllWindows()

    raise RuntimeError(
        "Not enough valid calibration points."
    )


# ============================================================
# WORLD -> PROJECTOR HOMOGRAPHY
# ============================================================

H_WORLD_TO_PROJECTOR, mask = cv2.findHomography(
    world_points,
    projector_points,
    cv2.RANSAC,
    4.0
)


if H_WORLD_TO_PROJECTOR is None:

    camera.release()

    cv2.destroyAllWindows()

    raise RuntimeError(
        "Could not calculate "
        "world-to-projector homography."
    )


# ============================================================
# RANSAC INLIER INFORMATION
# ============================================================

if mask is not None:

    inlier_mask = (
        mask.ravel() == 1
    )

    inlier_count = int(
        np.sum(
            inlier_mask
        )
    )

else:

    inlier_mask = np.ones(
        len(world_points),
        dtype=bool
    )

    inlier_count = len(
        world_points
    )


print(
    f"RANSAC inliers: "
    f"{inlier_count} / "
    f"{len(world_points)}"
)

print()


# ============================================================
# REPROJECTION ERROR
# ============================================================

predicted_projector_points = cv2.perspectiveTransform(
    world_points.reshape(
        -1,
        1,
        2
    ),
    H_WORLD_TO_PROJECTOR
).reshape(
    -1,
    2
)


errors = np.linalg.norm(
    predicted_projector_points
    - projector_points,
    axis=1
)


# ------------------------------------------------------------
# Error over all accepted detections
# ------------------------------------------------------------

mean_error_all = float(
    np.mean(
        errors
    )
)

max_error_all = float(
    np.max(
        errors
    )
)


# ------------------------------------------------------------
# Error using RANSAC inliers only
# ------------------------------------------------------------

inlier_errors = errors[
    inlier_mask
]


mean_error_inliers = float(
    np.mean(
        inlier_errors
    )
)

max_error_inliers = float(
    np.max(
        inlier_errors
    )
)


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("World -> Projector Homography:")
print()

print(
    H_WORLD_TO_PROJECTOR
)

print()

print(
    f"Mean error - all points: "
    f"{mean_error_all:.3f} px"
)

print(
    f"Maximum error - all points: "
    f"{max_error_all:.3f} px"
)

print()

print(
    f"Mean error - RANSAC inliers: "
    f"{mean_error_inliers:.3f} px"
)

print(
    f"Maximum error - RANSAC inliers: "
    f"{max_error_inliers:.3f} px"
)


# ============================================================
# SAVE NPZ
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


np.savez(
    OUTPUT_PATH,

    H_world_to_projector=
        H_WORLD_TO_PROJECTOR,

    camera_points_px=
        camera_points,

    world_points_mm=
        world_points,

    projector_points_px=
        projector_points,

    ransac_inlier_mask=
        inlier_mask,

    reprojection_errors_px=
        errors,

    mean_error_all_px=
        mean_error_all,

    max_error_all_px=
        max_error_all,

    mean_error_inlier_px=
        mean_error_inliers,

    max_error_inlier_px=
        max_error_inliers,

    projector_resolution=
        np.array(
            [
                PROJECTOR_WIDTH,
                PROJECTOR_HEIGHT
            ]
        )
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "eWoodX PROJECTOR CALIBRATION\n"
    )

    report.write(
        "=" * 55
        + "\n\n"
    )


    report.write(
        f"Valid detected points: "
        f"{len(world_points)}\n"
    )

    report.write(
        f"RANSAC inliers: "
        f"{inlier_count}\n\n"
    )


    report.write(
        f"Projector resolution: "
        f"{PROJECTOR_WIDTH} x "
        f"{PROJECTOR_HEIGHT}\n\n"
    )


    report.write(
        "World -> Projector Homography:\n"
    )

    report.write(
        np.array2string(
            H_WORLD_TO_PROJECTOR,
            precision=10
        )
    )

    report.write(
        "\n\n"
    )


    report.write(
        f"Mean error - all: "
        f"{mean_error_all:.4f} px\n"
    )

    report.write(
        f"Max error - all: "
        f"{max_error_all:.4f} px\n"
    )

    report.write(
        f"Mean error - inliers: "
        f"{mean_error_inliers:.4f} px\n"
    )

    report.write(
        f"Max error - inliers: "
        f"{max_error_inliers:.4f} px\n"
    )


    report.write(
        "\nCalibration Points:\n"
    )


    for i in range(
        len(world_points)
    ):

        world = world_points[i]

        projector = projector_points[i]

        camera_point = camera_points[i]

        error = errors[i]

        status = (
            "INLIER"
            if inlier_mask[i]
            else "OUTLIER"
        )


        report.write(
            f"P{i + 1:02d}: "
            f"Camera=("
            f"{camera_point[0]:.2f}, "
            f"{camera_point[1]:.2f}) px   "
            f"World=("
            f"{world[0]:.2f}, "
            f"{world[1]:.2f}) mm   "
            f"Projector=("
            f"{projector[0]:.0f}, "
            f"{projector[1]:.0f}) px   "
            f"Error="
            f"{error:.3f} px   "
            f"{status}\n"
        )


# ============================================================
# CLEANUP
# ============================================================

camera.release()

if controller is not None:
    try:
        controller.close()
    except Exception:
        pass

cv2.destroyAllWindows()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==========================================")
print("Calibration complete")
print("==========================================")

print()

print(
    "Saved NPZ:"
)

print(
    OUTPUT_PATH
)

print()

print(
    "Saved report:"
)

print(
    REPORT_PATH
)

print()

print(
    "IMPORTANT:"
)

print(
    "Watch the debug red circle during calibration."
)

print(
    "It must follow the projected white dot."
)