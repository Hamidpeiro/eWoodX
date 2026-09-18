import cv2
import numpy as np
import math

# ============================================================
# eWoodX - Physical Coordinate Projector Viewer
# Units: millimetres
# ============================================================

PROJECTOR_WIDTH = 1280
PROJECTOR_HEIGHT = 800

WINDOW_NAME = "eWoodX Projection"

# Projector position in Windows extended desktop
PROJECTOR_X = 2560
PROJECTOR_Y = 0


# ============================================================
# PHYSICAL CALIBRATION
# ============================================================

# P02 is our physical origin
ORIGIN_PX_X = 370
ORIGIN_PX_Y = 100

# Projected physical size / projected pixel size
MM_PER_PIXEL_X = 1750 / 1259   # X: width scale, mm per pixel
MM_PER_PIXEL_Y = 1010 / 779    # Y: height scale, mm per pixel


# ============================================================
# COLORS
# OpenCV uses BGR
# ============================================================

WHITE = (255, 255, 255)
GRAY = (160, 160, 160)


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def mm_to_pixel(x_mm, y_mm):
    """
    Convert physical table coordinates in mm
    to projector pixel coordinates.
    """

    px = ORIGIN_PX_X + (x_mm / MM_PER_PIXEL_X)
    py = ORIGIN_PX_Y + (y_mm / MM_PER_PIXEL_Y)

    return int(round(px)), int(round(py))


# ============================================================
# TEXT FUNCTION
# ============================================================

def draw_text(
    image,
    text,
    x_px,
    y_px,
    scale=0.45,
    thickness=1
):
    cv2.putText(
        image,
        text,
        (x_px, y_px),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        WHITE,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# RECTANGLE
# ============================================================

def draw_rectangle_mm(
    image,
    rect_id,
    x_mm,
    y_mm,
    width_mm,
    height_mm,
    thickness=2
):

    p1 = mm_to_pixel(
        x_mm,
        y_mm
    )

    p2 = mm_to_pixel(
        x_mm + width_mm,
        y_mm + height_mm
    )

    # Draw rectangle
    cv2.rectangle(
        image,
        p1,
        p2,
        WHITE,
        thickness
    )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    draw_text(
        image,
        rect_id,
        p1[0] + 8,
        p1[1] + 20,
        scale=0.5,
        thickness=1
    )

    # --------------------------------------------------------
    # WIDTH
    # --------------------------------------------------------

    width_text = f"W: {width_mm:.0f} mm"

    width_center_x = int(
        (p1[0] + p2[0]) / 2
    )

    draw_text(
        image,
        width_text,
        width_center_x - 35,
        p1[1] - 10
    )

    # --------------------------------------------------------
    # HEIGHT
    # --------------------------------------------------------

    height_text = f"H: {height_mm:.0f} mm"

    draw_text(
        image,
        height_text,
        p2[0] + 10,
        int((p1[1] + p2[1]) / 2)
    )

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    position_text = (
        f"X:{x_mm:.0f} "
        f"Y:{y_mm:.0f}"
    )

    draw_text(
        image,
        position_text,
        p1[0] + 8,
        p2[1] - 10,
        scale=0.4
    )


# ============================================================
# LINE / CUTTING LINE
# ============================================================

def draw_line_mm(
    image,
    line_id,
    x1_mm,
    y1_mm,
    x2_mm,
    y2_mm,
    thickness=2
):

    p1 = mm_to_pixel(
        x1_mm,
        y1_mm
    )

    p2 = mm_to_pixel(
        x2_mm,
        y2_mm
    )

    # Draw line
    cv2.line(
        image,
        p1,
        p2,
        WHITE,
        thickness
    )

    # --------------------------------------------------------
    # CALCULATE LENGTH
    # --------------------------------------------------------

    dx = x2_mm - x1_mm
    dy = y2_mm - y1_mm

    length_mm = math.sqrt(
        dx * dx +
        dy * dy
    )

    # --------------------------------------------------------
    # LABEL POSITION
    # --------------------------------------------------------

    center_x = int(
        (p1[0] + p2[0]) / 2
    )

    center_y = int(
        (p1[1] + p2[1]) / 2
    )

    label = (
        f"{line_id} | "
        f"{length_mm:.0f} mm"
    )

    draw_text(
        image,
        label,
        center_x + 10,
        center_y - 10,
        scale=0.45
    )


# ============================================================
# POINT / MARKER
# ============================================================

def draw_point_mm(
    image,
    point_id,
    x_mm,
    y_mm
):

    p = mm_to_pixel(
        x_mm,
        y_mm
    )

    cv2.circle(
        image,
        p,
        5,
        WHITE,
        -1
    )

    label = (
        f"{point_id} "
        f"({x_mm:.0f},{y_mm:.0f})"
    )

    draw_text(
        image,
        label,
        p[0] + 10,
        p[1] - 10,
        scale=0.4
    )


# ============================================================
# CREATE CANVAS
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
# ORIGIN
# ============================================================

draw_point_mm(
    canvas,
    "ORIGIN",
    0,
    0
)


# ============================================================
# TEST CUT RECTANGLES
# ============================================================

draw_rectangle_mm(
    canvas,
    rect_id="CUT-01",
    x_mm=0,
    y_mm=0,
    width_mm=100,
    height_mm=100
)

draw_rectangle_mm(
    canvas,
    rect_id="CUT-02",
    x_mm=300,
    y_mm=200,
    width_mm=200,
    height_mm=150
)

draw_rectangle_mm(
    canvas,
    rect_id="CUT-03",
    x_mm=600,
    y_mm=400,
    width_mm=250,
    height_mm=120
)


# ============================================================
# TEST CUTTING LINES
# ============================================================

draw_line_mm(
    canvas,
    line_id="L01",
    x1_mm=0,
    y1_mm=400,
    x2_mm=500,
    y2_mm=400
)

draw_line_mm(
    canvas,
    line_id="L02",
    x1_mm=400,
    y1_mm=50,
    x2_mm=700,
    y2_mm=250
)


# ============================================================
# GENERAL PROJECT INFO
# ============================================================

draw_text(
    canvas,
    "eWoodX - Cutting Projection",
    30,
    30,
    scale=0.6,
    thickness=1
)

draw_text(
    canvas,
    "Units: mm",
    30,
    55,
    scale=0.45
)


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