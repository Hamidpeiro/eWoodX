"""Offline checks; no camera access or writes to the user's source project."""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from handdraw_detector import HanddrawDetector, draw_detections, project_detections

root = Path(__file__).parent
for name in ("handdraw_detector.py", "03_timber_measurement_handdraw_yolo.py"):
    ast.parse((root / name).read_text(encoding="utf-8"))
detector = HanddrawDetector()
assert detector.detect(np.full((64, 64, 3), 245, np.uint8), []) == []
data_root = Path(r"C:\Users\hamid\Documents\GitHub\handdraw_yolo")
image_path = next((data_root / "images" / "val").glob("ellipse_*.png"))
image = cv2.imread(str(image_path))
h, w = image.shape[:2]
contour = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], np.int32)
groups = detector.detect(image, [{"contour": contour}])
assert groups[0] and any(d["class_name"] == "ellipse" for d in groups[0]), groups
for detection in groups[0]:
    x0, y0, x1, y1 = detection["bbox_xyxy_px"]
    assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
output = root / "detection_verification"
output.mkdir(exist_ok=True)
cv2.imwrite(str(output / "synthetic_detection.png"), draw_detections(image.copy(), groups[0]))
item = {"bbox_xyxy_px": [10.0, 20.0, 30.0, 40.0], "center_px": [20.0, 30.0]}
projected = project_detections([item], np.eye(3), 0.5, (2.0, 4.0))[0]
assert np.allclose(projected["center_mm"], [11, 17])
assert np.allclose(projected["bbox_corners_mm"][0], [6, 12])
assert item.get("center_mm") is None

rig = Path(r"C:\Users\hamid\Documents\GitHub\eWoodX\Webcam detection")
sys.path.insert(0, str(rig))
spec = importlib.util.spec_from_file_location("integrated_measurement", root / "03_timber_measurement_handdraw_yolo.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
empty = np.zeros((640, 640, 3), np.uint8)
detected, mask = module.segment_board(empty, contour, verbose=False)
assert detected == [] and mask.shape == empty.shape[:2]
module.HANDDRAW_DETECTOR = detector
module.config.CAPTURE_DIR = str(output)
capture = rig / "sample_output" / "captures" / "timber_01.jpg"
if not capture.is_file():
    print("PASS: syntax, CPU inference, synthetic ellipse, empty timber mask and coordinate projection.")
    print("SKIP: saved-capture integration check; the original timber_01.jpg is no longer available.")
    raise SystemExit(0)
frame = cv2.imread(str(capture))
assert frame is not None
frame_undist = module.undistort(frame)
overlay = module.render_live_viewport(frame_undist, 1, 0)
assert overlay.shape == frame.shape
cv2.imwrite(str(output / "saved_capture_live_overlay.png"), overlay)
success = module.measure_timber(frame, str(capture), 1, 0)
if not success:
    raise RuntimeError("Saved capture could not be measured; inspect the offline overlay")
report = json.loads((output / "timber_01_measurement.json").read_text())
assert report["handdraw_detection"]["device"] == "cpu"
assert all("hand_drawn_shapes" in t for t in report["timbers"])
assert report["hand_drawn_shape_count"] == sum(t["hand_drawn_shape_count"] for t in report["timbers"])
print("PASS: syntax, CPU inference, synthetic ellipse, empty timber mask,")
print("      coordinate projection, saved-capture live overlay and measurement JSON.")
print("Saved-capture timber count:", report["timber_count"])
print("Saved-capture drawing count:", report["hand_drawn_shape_count"])
