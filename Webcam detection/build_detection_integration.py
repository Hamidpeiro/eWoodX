"""Stage an updated copy of the user's multi-timber camera script."""
import ast
from pathlib import Path

source = Path(r"C:\Users\hamid\Documents\GitHub\eWoodX\Webcam detection\03_timber_measurement -hand_draw_YOLO.py")
target = Path(__file__).parent / "03_timber_measurement_handdraw_yolo.py"
text = source.read_text(encoding="utf-8-sig")


def replace(old, new):
    global text
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one integration anchor: {old[:100]!r}")
    text = text.replace(old, new, 1)


replace("import camera_utils\n", """import camera_utils
from handdraw_detector import (
    HanddrawDetector, DEFAULT_MODEL, project_detections, draw_detections,
)

HANDDRAW_DETECTOR = None
""")
replace("    timbers_result = []\n", """    timbers_result = []
    drawing_groups = (HANDDRAW_DETECTOR.detect(img_undist, detected_timbers)
                      if HANDDRAW_DETECTOR is not None
                      else [[] for _ in detected_timbers])
""")
replace("        # ADD THIS TIMBER TO LIST\n", """        # Drawing locations use the same undistorted image and height correction.
        timber_result["hand_drawn_shapes"] = project_detections(
            drawing_groups[i], H,
            scale=scale_factor if use_parallax else 1.0,
            origin=(Xc, Yc) if use_parallax else (0.0, 0.0),
        )
        timber_result["hand_drawn_shape_count"] = len(drawing_groups[i])

        # ADD THIS TIMBER TO LIST
""")
replace("    # 14. SAVE ONE JSON FILE\n", "    # 14. SAVE ONE JSON FILE\n")
replace("    out_path = os.path.join(\n\n        config.CAPTURE_DIR,", """    result["handdraw_detection"] = (HANDDRAW_DETECTOR.metadata()
                                    if HANDDRAW_DETECTOR is not None else {"enabled": False})
    result["hand_drawn_shape_count"] = sum(len(group) for group in drawing_groups)
    draw_detections(overlay, [item for group in drawing_groups for item in group])

    out_path = os.path.join(

        config.CAPTURE_DIR,""")
replace("    timber_detected = False\n", "    detected_timbers = []\n    timber_detected = False\n")
replace("    h, w = display.shape[:2]\n", """    if HANDDRAW_DETECTOR is not None and detected_timbers:
        drawing_groups = HANDDRAW_DETECTOR.detect(img_undist, detected_timbers)
        drawings = [item for group in drawing_groups for item in group]
        draw_detections(display, drawings)
        status_text += f" | {len(drawings)} drawing(s)"

    h, w = display.shape[:2]
""")
replace('    args = parser.parse_args()\n', """    parser.add_argument("--handdraw-model", default=str(DEFAULT_MODEL),
                        help="Trained ellipse/rectangle/triangle best.pt")
    parser.add_argument("--yolo-device", default="cpu", help="cpu (default), or GPU index such as 0")
    parser.add_argument("--yolo-confidence", type=float, default=0.5)
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--no-handdraw", action="store_true", help="Only measure timbers")
    args = parser.parse_args()

    global HANDDRAW_DETECTOR
    if not args.no_handdraw:
        try:
            HANDDRAW_DETECTOR = HanddrawDetector(
                args.handdraw_model, args.yolo_device, args.yolo_confidence,
                args.yolo_image_size, args.yolo_iou,
            )
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            parser.error(str(exc))
        print(f"Hand-drawing detection enabled: {HANDDRAW_DETECTOR.model_path}")
        print(f"YOLO device: {args.yolo_device}; confidence: {args.yolo_confidence}")
""")
replace('    print("Place ONE timber inside the black workspace.")',
        '    print("Place timber(s) with hand-drawn shapes inside the black workspace.")')
# Existing no-timber paths returned three values, while both callers unpack two.
start = text.index("def segment_board(")
end = text.index("def pixel_to_mm(", start)
section = text[start:end]
section = section.replace("return None, np.zeros_like(gray), None", "return [], np.zeros_like(gray)")
section = section.replace("return None, timber_mask, None", "return [], timber_mask")
text = text[:start] + section + text[end:]
text = text.replace("Step 3: Capture and measure timbers on a black background.",
                    "Step 3: Measure multiple timbers and detect hand-drawn shapes with trained YOLO.", 1)
ast.parse(text)
target.write_text(text, encoding="utf-8")
print(f"Staged and syntax checked: {target}")
