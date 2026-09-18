import cv2
import numpy as np

# ============================================================
# eWoodX - 100 x 100 mm Rectangle Test
# Start point: P02
# Projector: 1280 x 800
# ============================================================

WIDTH = 1280
HEIGHT = 800

WINDOW_NAME = "eWoodX 100x100mm Test"

PROJECTOR_X = 2560   
PROJECTOR_Y = 0

# ------------------------------------------------------------
# Current approximate projector scale
# ------------------------------------------------------------

# Projected physical size / projected pixel size
MM_PER_PIXEL_X = 1750 / 1259   # X: width scale, mm per pixel
MM_PER_PIXEL_Y = 1010 / 779    # Y: height scale, mm per pixel

# Desired physical rectangle
RECT_WIDTH_MM = 100
RECT_HEIGHT_MM = 100

# ------------------------------------------------------------
# P02 projector position
# ------------------------------------------------------------

P02_X = 1000
P02_Y = 600

'''
P02_X = 370
P02_Y = 100

# Top-left
P02_X = 150
P02_Y = 120

# Top-right
P02_X = 1000
P02_Y = 120

# Center
P02_X = 600
P02_Y = 350

# Bottom-left
P02_X = 150
P02_Y = 600

# Bottom-right
P02_X = 1000
P02_Y = 600
'''

# ------------------------------------------------------------
# Convert desired mm -> approximate projector pixels
# ------------------------------------------------------------

rect_width_px = round(RECT_WIDTH_MM / MM_PER_PIXEL_X)
rect_height_px = round(RECT_HEIGHT_MM / MM_PER_PIXEL_Y)

print("Rectangle size:")
print(f"Physical = {RECT_WIDTH_MM} x {RECT_HEIGHT_MM} mm")
print(f"Pixels   = {rect_width_px} x {rect_height_px} px")

# ------------------------------------------------------------
# Canvas
# ------------------------------------------------------------

canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

WHITE = (255, 255, 255)

# ------------------------------------------------------------
# Rectangle coordinates
#
# P02 is TOP-LEFT corner
# ------------------------------------------------------------

x1 = P02_X
y1 = P02_Y

x2 = P02_X + rect_width_px
y2 = P02_Y + rect_height_px

# ------------------------------------------------------------
# Draw rectangle
# ------------------------------------------------------------

cv2.rectangle(
    canvas,
    (x1, y1),
    (x2, y2),
    WHITE,
    2
)

# ------------------------------------------------------------
# Mark P02
# ------------------------------------------------------------

cv2.circle(
    canvas,
    (P02_X, P02_Y),
    6,
    WHITE,
    -1
)

cv2.putText(
    canvas,
    "P02",
    (P02_X + 10, P02_Y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    WHITE,
    1,
    cv2.LINE_AA
)

# ------------------------------------------------------------
# Create window
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

while True:

    cv2.imshow(
        WINDOW_NAME,
        canvas
    )

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break

cv2.destroyAllWindows()