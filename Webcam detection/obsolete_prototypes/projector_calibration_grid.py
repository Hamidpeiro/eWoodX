import cv2
import numpy as np

# ============================================================
# eWoodX - Projector Calibration Grid
# Projector resolution: 1280 x 800
# ============================================================

WIDTH = 1280
HEIGHT = 800

WINDOW_NAME = "eWoodX Projector Calibration"

# Change this if your projector starts at another X position
PROJECTOR_X = 2560
PROJECTOR_Y = 0


# ============================================================
# CREATE BLACK CANVAS
# ============================================================

canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

WHITE = (255, 255, 255)


# ============================================================
# DRAW CROSS
# ============================================================

def draw_cross(image, x, y, size=12, thickness=2):

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


# ============================================================
# CALIBRATION GRID
#
# 5 columns
# 4 rows
#
# Keep points away from the extreme projector edges.
# ============================================================

x_positions = [
    100,
    370,
    640,
    910,
    1180
]

y_positions = [
    100,
    300,
    500,
    700
]


# ============================================================
# DRAW OUTER REFERENCE BORDER
# ============================================================

cv2.rectangle(
    canvas,
    (50, 50),
    (WIDTH - 50, HEIGHT - 50),
    WHITE,
    1
)


# ============================================================
# DRAW CENTER LINES
# ============================================================

cv2.line(
    canvas,
    (WIDTH // 2, 50),
    (WIDTH // 2, HEIGHT - 50),
    WHITE,
    1
)

cv2.line(
    canvas,
    (50, HEIGHT // 2),
    (WIDTH - 50, HEIGHT // 2),
    WHITE,
    1
)


# ============================================================
# DRAW CALIBRATION POINTS
# ============================================================

font = cv2.FONT_HERSHEY_SIMPLEX

point_id = 1

projector_points = []

for y in y_positions:

    for x in x_positions:

        # Save projector pixel coordinate
        projector_points.append((x, y))

        # Draw cross
        draw_cross(
            canvas,
            x,
            y,
            size=12,
            thickness=2
        )

        # Point label
        label = f"P{point_id:02d}"

        cv2.putText(
            canvas,
            label,
            (x + 15, y - 15),
            font,
            0.45,
            WHITE,
            1,
            cv2.LINE_AA
        )

        point_id += 1


# ============================================================
# CENTER MARKER
# ============================================================

draw_cross(
    canvas,
    WIDTH // 2,
    HEIGHT // 2,
    size=25,
    thickness=2
)

cv2.putText(
    canvas,
    "CENTER",
    (WIDTH // 2 + 30, HEIGHT // 2 - 10),
    font,
    0.5,
    WHITE,
    1,
    cv2.LINE_AA
)


# ============================================================
# INFO TEXT
# ============================================================

cv2.putText(
    canvas,
    "eWoodX Projector Calibration - 1280 x 800",
    (60, HEIGHT - 20),
    font,
    0.5,
    WHITE,
    1,
    cv2.LINE_AA
)


# ============================================================
# PRINT PROJECTOR POINTS IN TERMINAL
# ============================================================

print("\nProjector calibration points:")
print("--------------------------------")

for i, point in enumerate(projector_points, start=1):

    print(
        f"P{i:02d}: "
        f"x = {point[0]:4d}, "
        f"y = {point[1]:4d}"
    )

print("--------------------------------")
print("Press ESC to close.\n")


# ============================================================
# CREATE PROJECTOR WINDOW
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
# DISPLAY LOOP
# ============================================================

while True:

    cv2.imshow(
        WINDOW_NAME,
        canvas
    )

    key = cv2.waitKey(10) & 0xFF

    if key == 27:  # ESC
        break


cv2.destroyAllWindows()