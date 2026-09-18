import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 800

WINDOW_NAME = "eWoodX Projector Test"

canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

WHITE = (255, 255, 255)

# --------------------------------------------------
# CROSS FUNCTION
# --------------------------------------------------

def draw_cross(image, x, y, size=15, thickness=2):

    cv2.line(
        image,
        (x - size, y),
        (x + size, y),
        WHITE,
        thickness
    )

    cv2.line(
        image,
        (x, y - size),
        (x, y + size),
        WHITE,
        thickness
    )


# --------------------------------------------------
# OUTER BORDER
# 10 px inside the projector image
# --------------------------------------------------

cv2.rectangle(
    canvas,
    (10, 10),
    (WIDTH - 11, HEIGHT - 11),
    WHITE,
    2
)


# --------------------------------------------------
# 50 px BORDER
# --------------------------------------------------

cv2.rectangle(
    canvas,
    (50, 50),
    (WIDTH - 51, HEIGHT - 51),
    WHITE,
    1
)


# --------------------------------------------------
# 100 px BORDER
# --------------------------------------------------

cv2.rectangle(
    canvas,
    (100, 100),
    (WIDTH - 101, HEIGHT - 101),
    WHITE,
    1
)


# --------------------------------------------------
# CORNER MARKERS
# --------------------------------------------------

draw_cross(canvas, 50, 50)
draw_cross(canvas, WIDTH - 50, 50)

draw_cross(canvas, 50, HEIGHT - 50)
draw_cross(canvas, WIDTH - 50, HEIGHT - 50)


# --------------------------------------------------
# CENTER
# --------------------------------------------------

draw_cross(
    canvas,
    WIDTH // 2,
    HEIGHT // 2,
    size=30,
    thickness=2
)


# --------------------------------------------------
# CENTER LINES
# --------------------------------------------------

cv2.line(
    canvas,
    (WIDTH // 2, 0),
    (WIDTH // 2, HEIGHT),
    WHITE,
    1
)

cv2.line(
    canvas,
    (0, HEIGHT // 2),
    (WIDTH, HEIGHT // 2),
    WHITE,
    1
)


# --------------------------------------------------
# REFERENCE RECTANGLE
# 800 × 400 pixels
# --------------------------------------------------

cv2.rectangle(
    canvas,
    (240, 200),
    (1040, 600),
    WHITE,
    2
)


# --------------------------------------------------
# WINDOW
# --------------------------------------------------

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

PROJECTOR_X = 1920
PROJECTOR_Y = 0

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


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

while True:

    cv2.imshow(
        WINDOW_NAME,
        canvas
    )

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break


cv2.destroyAllWindows()