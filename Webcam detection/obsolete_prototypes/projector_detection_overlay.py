import cv2
import numpy as np
import json

# ============================================================
# eWoodX - Project Detection JSON
# ============================================================

PROJECTOR_WIDTH = 1280
PROJECTOR_HEIGHT = 800

WINDOW_NAME = "eWoodX Timber Projection"

# ------------------------------------------------------------
# WINDOWS DISPLAY POSITION
# Main monitor = 2560 x 1600
# Epson is assumed to be on the RIGHT
# ------------------------------------------------------------

PROJECTOR_X = 2560
PROJECTOR_Y = 0


# ============================================================
# PROJECTOR CALIBRATION
# ============================================================

# P02 projector pixel used as physical origin
ORIGIN_PX_X = 370
ORIGIN_PX_Y = 100

# Projected physical size / projected pixel size
MM_PER_PIXEL_X = 1750 / 1259   # X: width scale, mm per pixel
MM_PER_PIXEL_Y = 1010 / 779    # Y: height scale, mm per pixel


# ============================================================
# IMPORTANT: JSON ORIGIN ALIGNMENT
# ============================================================
#
# Your detection JSON uses:
#
# Marker 1 outer corner = (0, 0) mm
#
# Your projector currently uses:
#
# P02 = (0, 0) mm
#
# If P02 physically corresponds exactly to Marker 1 origin:
#
#     OFFSET_X_MM = 0
#     OFFSET_Y_MM = 0
#
# Otherwise measure the physical difference between:
#
#     Marker 1 origin
# and
#     P02 origin
#
# and enter it here.
# ============================================================

OFFSET_X_MM = 0
OFFSET_Y_MM = 0


# ============================================================
# JSON FILE
# ============================================================

JSON_PATH = r"C:\Users\hamid\Documents\GitHub\eEoodX\Webcam detection\sample_output\captures\timber_01_measurement.json"

# ============================================================
# COLORS - OpenCV BGR
# ============================================================

WHITE = (255, 255, 255)
GRAY = (160, 160, 160)


# ============================================================
# MM -> PROJECTOR PIXEL
# ============================================================

def mm_to_pixel(x_mm, y_mm):

    # Apply physical origin offset
    x_mm = x_mm + OFFSET_X_MM
    y_mm = y_mm + OFFSET_Y_MM

    px = ORIGIN_PX_X + (x_mm / MM_PER_PIXEL_X)
    py = ORIGIN_PX_Y + (y_mm / MM_PER_PIXEL_Y)

    return int(round(px)), int(round(py))


# ============================================================
# LOAD JSON
# ============================================================

with open(JSON_PATH, "r") as f:
    data = json.load(f)


timber_id = data["timber_id"]

length_mm = data["length_mm"]
width_mm = data["width_mm"]

contour_mm = data["contour_mm"]
corners_mm = data["corners_mm"]

defects = data.get("defects", [])


# ============================================================
# CREATE PROJECTOR CANVAS
# ============================================================

canvas = np.zeros(
    (
        PROJECTOR_HEIGHT,
        PROJECTOR_WIDTH,
        3
    ),
    dtype=np.uint8
)


# ============================================================
# CONVERT TIMBER CONTOUR
# ============================================================

contour_px = []

for x_mm, y_mm in contour_mm:

    px, py = mm_to_pixel(
        x_mm,
        y_mm
    )

    contour_px.append(
        [px, py]
    )


contour_px = np.array(
    contour_px,
    dtype=np.int32
).reshape((-1, 1, 2))


# ============================================================
# DRAW ACTUAL TIMBER CONTOUR
# ============================================================

cv2.polylines(
    canvas,
    [contour_px],
    True,
    WHITE,
    2,
    cv2.LINE_AA
)


# ============================================================
# DRAW DETECTED CORNERS
# ============================================================

for i, corner in enumerate(corners_mm):

    x_mm, y_mm = corner

    px, py = mm_to_pixel(
        x_mm,
        y_mm
    )

    cv2.circle(
        canvas,
        (px, py),
        5,
        WHITE,
        -1
    )

    cv2.putText(
        canvas,
        f"C{i + 1}",
        (px + 8, py - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        WHITE,
        1,
        cv2.LINE_AA
    )


# ============================================================
# TIMBER INFORMATION
# ============================================================

# Find visual center of detected timber
corners_array = np.array(corners_mm)

center_x_mm = np.mean(corners_array[:, 0])
center_y_mm = np.mean(corners_array[:, 1])

center_px = mm_to_pixel(
    center_x_mm,
    center_y_mm
)


info_1 = f"TIMBER {timber_id}"

info_2 = (
    f"L: {length_mm:.1f} mm  "
    f"W: {width_mm:.1f} mm"
)


cv2.putText(
    canvas,
    info_1,
    (
        center_px[0] - 80,
        center_px[1] - 25
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    WHITE,
    1,
    cv2.LINE_AA
)


cv2.putText(
    canvas,
    info_2,
    (
        center_px[0] - 120,
        center_px[1]
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    WHITE,
    1,
    cv2.LINE_AA
)


# ============================================================
# DRAW ORIGIN
# ============================================================

origin_px = mm_to_pixel(
    0,
    0
)

cv2.circle(
    canvas,
    origin_px,
    6,
    WHITE,
    -1
)

cv2.putText(
    canvas,
    "JSON ORIGIN (0,0)",
    (
        origin_px[0] + 10,
        origin_px[1] - 10
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.4,
    WHITE,
    1,
    cv2.LINE_AA
)


# ============================================================
# DEFECTS
# ============================================================

# Your current JSON has no defects yet.
# Later YOLO cracks / knots can be drawn here.

if len(defects) == 0:

    print("No defects stored in this JSON.")


# ============================================================
# TERMINAL INFORMATION
# ============================================================

print()
print("------------------------------------------")
print("eWoodX projection")
print("------------------------------------------")

print(f"Timber ID: {timber_id}")

print(
    f"Length: {length_mm:.2f} mm"
)

print(
    f"Width: {width_mm:.2f} mm"
)

print(
    f"Contour points: {len(contour_mm)}"
)

print(
    f"Defects: {len(defects)}"
)

print("------------------------------------------")


# ============================================================
# PROJECTOR WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.moveWindow(
    WINDOW_NAME,
    PROJECTOR_X,
    PROJECTOR_Y
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# DISPLAY
# ============================================================

while True:

    cv2.imshow(
        WINDOW_NAME,
        canvas
    )

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break


cv2.destroyAllWindows()