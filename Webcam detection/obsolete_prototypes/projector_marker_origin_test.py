import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 800

PROJECTOR_X = 2560
PROJECTOR_Y = 0

# Change these values until the cross lands exactly
# on Marker 1 outer corner.
MARKER1_PX_X = 150
MARKER1_PX_Y = 150

canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

WHITE = (255, 255, 255)

size = 20

cv2.line(
    canvas,
    (MARKER1_PX_X - size, MARKER1_PX_Y),
    (MARKER1_PX_X + size, MARKER1_PX_Y),
    WHITE,
    2
)

cv2.line(
    canvas,
    (MARKER1_PX_X, MARKER1_PX_Y - size),
    (MARKER1_PX_X, MARKER1_PX_Y + size),
    WHITE,
    2
)

cv2.putText(
    canvas,
    "Marker 1 Origin",
    (MARKER1_PX_X + 25, MARKER1_PX_Y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    WHITE,
    1,
    cv2.LINE_AA
)

cv2.namedWindow(
    "Marker 1 Alignment",
    cv2.WINDOW_NORMAL
)

cv2.moveWindow(
    "Marker 1 Alignment",
    PROJECTOR_X,
    PROJECTOR_Y
)

cv2.setWindowProperty(
    "Marker 1 Alignment",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:

    cv2.imshow(
        "Marker 1 Alignment",
        canvas
    )

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break

cv2.destroyAllWindows()