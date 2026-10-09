"""YOLO drawing detection and geometric shape reconstruction for eWoodX.

Detects hand drawings on multi-timber setups and generates class-specific shapes:
  - 'ellipse'   -> Generates smooth Ellipse curves (inscribed in detection box)
  - 'rectangle' -> Generates 4-corner Oriented Rectangles
  - 'triangle'  -> Generates 3-vertex Triangles (pointing upwards / aligned)
"""

from pathlib import Path
import cv2
import numpy as np

DEFAULT_MODEL = Path(r"C:\Users\hamid\Documents\GitHub\eWoodX\Webcam detection\train\weights\best.pt")
DEFAULT_CLASS_NAMES = {0: "ellipse", 1: "rectangle", 2: "triangle"}
DEFAULT_COLORS = {
    0: (0, 165, 255),   # Ellipse: Vibrant Orange (BGR)
    1: (255, 0, 180),   # Rectangle: Bright Magenta (BGR)
    2: (255, 180, 0),   # Triangle: Cyan / Sky Blue (BGR)
}


def generate_shape_polygon(class_name, bbox_xyxy, num_ellipse_points=4, image=None):
    """
    Generate the exact geometric shape boundary points (in px or mm).
      - ellipse: 4 cardinal/quad points (top, right, bottom, left) or parametric curve
      - rectangle: 4 ordered corner points. If `image` is provided, fits a rotated rectangle using contour detection.
      - triangle: 3 apex/base vertex points (top-center, bottom-right, bottom-left)
    """
    import cv2
    import numpy as np

    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    c_lower = str(class_name).lower()

    if "ellipse" in c_lower or "circle" in c_lower:
        pts = None
        if image is not None:
            pad = 10
            px0 = max(0, int(x0) - pad)
            py0 = max(0, int(y0) - pad)
            px1 = min(image.shape[1], int(x1) + pad)
            py1 = min(image.shape[0], int(y1) + pad)

            if px1 > px0 and py1 > py0:
                roi = image[py0:py1, px0:px1].copy()
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
                contours, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    if len(largest_cnt) >= 5:
                        (ecx, ecy), (MA, ma), angle = cv2.fitEllipse(largest_cnt)
                        # Generate 36 points for smooth ellipse
                        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                        a_rad = np.deg2rad(angle)
                        cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)
                        rx, ry = MA / 2.0, ma / 2.0
                        pts_local = []
                        for ang in angles:
                            xx = ecx + rx * np.cos(ang) * cos_a - ry * np.sin(ang) * sin_a
                            yy = ecy + rx * np.cos(ang) * sin_a + ry * np.sin(ang) * cos_a
                            pts_local.append([float(xx + px0), float(yy + py0)])
                        pts = pts_local

        if pts is None:
            # Fallback 36-point parametric ellipse based on bounding box
            rx = w / 2.0
            ry = h / 2.0
            angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            pts = [[cx + rx * np.cos(a), cy + ry * np.sin(a)] for a in angles]
            
        return pts, "ellipse"

    elif "tri" in c_lower:
        pts = None
        if image is not None:
            pad = 10
            px0 = max(0, int(x0) - pad)
            py0 = max(0, int(y0) - pad)
            px1 = min(image.shape[1], int(x1) + pad)
            py1 = min(image.shape[0], int(y1) + pad)

            if px1 > px0 and py1 > py0:
                roi = image[py0:py1, px0:px1].copy()
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
                contours, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    epsilon = 0.05 * cv2.arcLength(largest_cnt, True)
                    approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
                    
                    if len(approx) == 3:
                        pts = [[float(p[0][0] + px0), float(p[0][1] + py0)] for p in approx]
                    else:
                        retval, triangle = cv2.minEnclosingTriangle(largest_cnt)
                        if retval:
                            pts = [[float(p[0][0] + px0), float(p[0][1] + py0)] for p in triangle]

        if pts is None:
            # Fallback Triangle: 3 points (top apex, bottom-right, bottom-left)
            pts = [
                [cx, y0],
                [x1, y1],
                [x0, y1]
            ]
        return pts, "triangle"

    else:
        # Rectangle: Try to fit a rotated rectangle if image is provided
        pts = None
        if image is not None:
            pad = 10
            px0 = max(0, int(x0) - pad)
            py0 = max(0, int(y0) - pad)
            px1 = min(image.shape[1], int(x1) + pad)
            py1 = min(image.shape[0], int(y1) + pad)

            if px1 > px0 and py1 > py0:
                roi = image[py0:py1, px0:px1].copy()
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
                contours, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    
                    # Try approxPolyDP to find exact 4 corners (avoids pencil line overshoot skewing)
                    epsilon = 0.05 * cv2.arcLength(largest_cnt, True)
                    approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
                    
                    box = None
                    if len(approx) == 4:
                        box = approx.reshape(-1, 2)
                    else:
                        # Fallback to minAreaRect if approx fails to find exactly 4 corners
                        rect = cv2.minAreaRect(largest_cnt)
                        box = cv2.boxPoints(rect)
                    
                    if box is not None:
                        # Order points: top-left, top-right, bottom-right, bottom-left
                        # Sort by x coordinate
                        x_sorted = box[np.argsort(box[:, 0]), :]
                        left_pts = x_sorted[:2, :]
                        right_pts = x_sorted[2:, :]
                        
                        # Sort left points by y coordinate
                        left_pts = left_pts[np.argsort(left_pts[:, 1]), :]
                        tl, bl = left_pts
                        
                        # Sort right points by y coordinate
                        right_pts = right_pts[np.argsort(right_pts[:, 1]), :]
                        tr, br = right_pts
                        
                        ordered_box = np.array([tl, tr, br, bl])
                        global_box = ordered_box + np.array([px0, py0])
                        pts = [[float(p[0]), float(p[1])] for p in global_box]

        if pts is None:
            # Fallback: Rectangle: 4 corner points (top-left, top-right, bottom-right, bottom-left)
            pts = [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1]
            ]
        return pts, "rectangle"


class HanddrawDetector:
    def __init__(self, model_path=DEFAULT_MODEL, device="cpu", confidence=0.15,
                 image_size=640, iou=0.45):
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Trained hand-drawing model not found: {path}")
        if not 0 < confidence <= 1 or not 0 < iou <= 1 or image_size < 32:
            raise ValueError("Invalid YOLO confidence, IoU, or image size")
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics and PyTorch in the Python environment running this script") from exc
        if str(device) != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use your GPU-enabled Python environment or --yolo-device cpu")

        self.model_path = str(path)
        self.model = YOLO(str(path), task="detect")
        names = self.model.names
        actual = dict(enumerate(names)) if isinstance(names, list) else (names if isinstance(names, dict) else DEFAULT_CLASS_NAMES)
        self.class_names = {int(k): str(v) for k, v in actual.items()}
        self.device = device
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.iou = float(iou)

        # Warm up YOLO model
        self.model.predict(np.full((image_size, image_size, 3), 245, np.uint8),
                           device=device, imgsz=image_size, verbose=False)

    def set_confidence(self, conf):
        self.confidence = max(0.01, min(0.99, float(conf)))

    @staticmethod
    def _is_edge_detection(bx0, by0, bx1, by1, crop_w, crop_h,
                           edge_margin_frac=0.08):
        """
        Return True if the detection bbox touches the crop boundary.
        Detections at the crop edges are almost always false positives caused
        by timber corner cutoffs, edges, or dark shadows against the curtain.
        """
        mx = crop_w * edge_margin_frac
        my = crop_h * edge_margin_frac
        return (bx0 < mx or bx1 > crop_w - mx or
                by0 < my or by1 > crop_h - my)

    @staticmethod
    def _preprocess_adaptive_threshold(masked_crop, crop_mask):
        """
        Convert faint pencil strokes into high-contrast black-on-white image
        matching the YOLO training data distribution. This is the most effective
        preprocessing for pencil drawings on wood grain.
        """
        gray = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2GRAY)
        gray_masked = gray.copy()
        gray_masked[crop_mask == 0] = 200

        adapt = cv2.adaptiveThreshold(
            gray_masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 8
        )
        adapt_bgr = cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)
        adapt_bgr[crop_mask == 0] = [180, 180, 180]
        return adapt_bgr

    @staticmethod
    def _preprocess_unsharp_clahe(masked_crop, crop_mask):
        """
        Unsharp mask sharpening followed by CLAHE contrast enhancement.
        Strong at revealing medium-contrast pencil strokes.
        """
        blur = cv2.GaussianBlur(masked_crop, (0, 0), 3)
        unsharp = cv2.addWeighted(masked_crop, 2.0, blur, -1.0, 0)
        lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl_enh = clahe.apply(l_chan)
        result = cv2.cvtColor(cv2.merge((cl_enh, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
        result[crop_mask == 0] = [180, 180, 180]
        return result

    @staticmethod
    def _preprocess_clahe(masked_crop, crop_mask):
        """Standard CLAHE contrast enhancement."""
        lab = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl_enh = clahe.apply(l_chan)
        result = cv2.cvtColor(cv2.merge((cl_enh, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
        result[crop_mask == 0] = [180, 180, 180]
        return result

    def _run_inference(self, img_bgr, imgsz=640):
        """Run single YOLO inference pass and return list of (bx0, by0, bx1, by1, score, cls)."""
        results = self.model.predict(
            img_bgr, device=self.device, conf=self.confidence,
            iou=self.iou, imgsz=imgsz, agnostic_nms=True, verbose=False
        )[0]
        candidates = []
        if results.boxes is not None and len(results.boxes) > 0:
            for row in results.boxes.data.detach().cpu().numpy():
                bx0, by0, bx1, by1, score, cls = map(float, row[:6])
                candidates.append((bx0, by0, bx1, by1, score, int(cls)))
        return candidates

    def _predict_crop(self, masked_crop, contour_shifted, crop_mask=None):
        """
        Run multi-variant ensemble YOLO inference on a masked timber crop.

        Preprocessing variants (in order of effectiveness for pencil on wood):
          1. Adaptive threshold  – converts faint pencil to black-on-white (97%+ conf)
          2. Unsharp + CLAHE     – sharpens then enhances contrast (78%+ conf)
          3. Standard CLAHE      – baseline contrast enhancement (56%+ conf)

        All variants are run and their candidates are merged via NMS to produce
        a robust, high-confidence detection set.

        Returns raw candidate boxes: list of (bx0, by0, bx1, by1, score, cls).
        """
        ch, cw = masked_crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return []

        # Build the contour mask if not provided
        if crop_mask is None:
            crop_mask = np.zeros((ch, cw), dtype=np.uint8)
            if contour_shifted is not None and len(contour_shifted) >= 3:
                cv2.drawContours(crop_mask, [contour_shifted], -1, 255, cv2.FILLED)
            else:
                crop_mask[:] = 255

        raw_candidates = []

        # Determine optimal YOLO imgsz based on crop dimensions
        target_imgsz = min(1024, max(640, max(cw, ch)))

        # --- Variant 1: Adaptive threshold (best for faint pencil) ---
        adapt_img = self._preprocess_adaptive_threshold(masked_crop, crop_mask)
        raw_candidates.extend(self._run_inference(adapt_img, target_imgsz))

        # --- Variant 2: Unsharp mask + CLAHE (good for medium contrast) ---
        unsharp_img = self._preprocess_unsharp_clahe(masked_crop, crop_mask)
        raw_candidates.extend(self._run_inference(unsharp_img, target_imgsz))

        # --- Variant 3: Standard CLAHE (baseline) ---
        clahe_img = self._preprocess_clahe(masked_crop, crop_mask)
        raw_candidates.extend(self._run_inference(clahe_img, target_imgsz))

        return raw_candidates

    def detect(self, frame, timbers):
        """Return one detection list per timber, with shapes strictly mapped to their timber ID."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected an undistorted BGR image")
        height, width = frame.shape[:2]
        output = [[] for _ in timbers]

        for index, timber in enumerate(timbers):
            contour = np.asarray(timber["contour"], dtype=np.int32).reshape(-1, 1, 2)
            if len(contour) < 3:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            # Ensure bounds inside frame
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            cw, ch = x1 - x0, y1 - y0
            if cw <= 0 or ch <= 0:
                continue

            # Create contour mask inside crop coordinates
            cnt_crop = contour - np.array([x0, y0])
            crop_mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.drawContours(crop_mask, [cnt_crop], -1, 255, cv2.FILLED)

            # Gentle mask erosion to avoid erasing hand-drawn shapes near edges
            k_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            eroded_mask = cv2.erode(crop_mask, kernel)

            # Isolate timber: replace background outside timber contour with neutral gray (180, 180, 180)
            masked_crop = frame[y0:y1, x0:x1].copy()
            masked_crop[eroded_mask == 0] = [180, 180, 180]

            # Get raw candidates from multi-variant ensemble inference
            raw_candidates = self._predict_crop(masked_crop, cnt_crop, crop_mask=eroded_mask)

            if not raw_candidates:
                continue

            boxes_list = []
            scores_list = []
            valid_candidates = []

            for cand in raw_candidates:
                bx0, by0, bx1, by1, score, cls = cand
                bx0, bx1 = np.clip([bx0, bx1], 0, cw)
                by0, by1 = np.clip([by0, by1], 0, ch)
                if bx1 <= bx0 or by1 <= by0:
                    continue

                # Reject detections touching timber crop boundaries (false edge artifacts)
                if self._is_edge_detection(bx0, by0, bx1, by1, cw, ch):
                    continue

                boxes_list.append([bx0, by0, bx1 - bx0, by1 - by0])
                scores_list.append(score)
                valid_candidates.append((bx0, by0, bx1, by1, score, cls))

            if not boxes_list:
                continue

            indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, self.confidence, self.iou)
            if len(indices) == 0:
                continue
            indices = np.asarray(indices).flatten()

            # Collect NMS survivors
            nms_results = [valid_candidates[idx] for idx in indices]

            # Merge overlapping same-class detections (containment-based)
            merged = self._merge_overlapping_detections(nms_results)

            for bx0, by0, bx1, by1, score, cls in merged:
                gx0, gy0, gx1, gy1 = map(float, (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0))
                c_name = self.class_names.get(cls, f"shape_{cls}")
                shape_pts, shape_type = generate_shape_polygon(c_name, [gx0, gy0, gx1, gy1], image=frame)

                output[index].append({
                    "class_id": cls,
                    "class_name": c_name,
                    "shape_type": shape_type,
                    "confidence": float(score),
                    "timber_id": index + 1,
                    "bbox_xyxy_px": [gx0, gy0, gx1, gy1],
                    "center_px": [(gx0 + gx1) / 2, (gy0 + gy1) / 2],
                    "shape_points_px": shape_pts,
                    "dimensions_px": {"width": float(gx1 - gx0), "height": float(gy1 - gy0)},
                    "coordinate_space": "undistorted_image",
                })
        return output

    @staticmethod
    def _merge_overlapping_detections(detections, containment_thresh=0.5):
        """
        Merge same-class detections where one box is significantly overlapping
        or contained within another.  Keeps the highest-confidence detection
        per spatial cluster.

        This handles the common case where the multi-variant ensemble produces
        a high-confidence full detection AND low-confidence partial detections
        (e.g. just an edge or corner of the same drawn shape).

        Two detections A, B of the same class are merged when:
          - Their IoU > containment_thresh, OR
          - The fraction of the smaller box's area that overlaps the larger box
            exceeds containment_thresh (one is "mostly inside" the other).
        """
        if len(detections) <= 1:
            return detections

        keep = [True] * len(detections)

        for i in range(len(detections)):
            if not keep[i]:
                continue
            bx0_i, by0_i, bx1_i, by1_i, score_i, cls_i = detections[i]
            area_i = (bx1_i - bx0_i) * (by1_i - by0_i)
            if area_i <= 0:
                keep[i] = False
                continue

            for j in range(i + 1, len(detections)):
                if not keep[j]:
                    continue
                bx0_j, by0_j, bx1_j, by1_j, score_j, cls_j = detections[j]

                # Only merge same-class detections
                if cls_i != cls_j:
                    continue

                area_j = (bx1_j - bx0_j) * (by1_j - by0_j)
                if area_j <= 0:
                    keep[j] = False
                    continue

                # Compute intersection
                ix0 = max(bx0_i, bx0_j)
                iy0 = max(by0_i, by0_j)
                ix1 = min(bx1_i, bx1_j)
                iy1 = min(by1_i, by1_j)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)

                if inter <= 0:
                    continue

                # Check IoU
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0

                # Check containment: what fraction of the smaller box overlaps?
                smaller_area = min(area_i, area_j)
                containment = inter / smaller_area if smaller_area > 0 else 0

                if iou > containment_thresh or containment > containment_thresh:
                    # Suppress the lower-confidence detection
                    if score_i >= score_j:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break  # i is suppressed, move on

        return [d for d, k in zip(detections, keep) if k]


    def detect_workspace(self, frame, workspace_polygon):
        """Fallback: detect drawings across workspace when timber contour is not yet isolated."""
        if frame is None or workspace_polygon is None or len(workspace_polygon) < 3:
            return []
        pts = np.asarray(workspace_polygon, dtype=np.int32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(pts)
        height, width = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            return []

        crop = frame[y0:y1, x0:x1].copy()
        result = self.model.predict(
            crop, device=self.device, conf=self.confidence,
            iou=self.iou, imgsz=self.image_size,
            agnostic_nms=True, verbose=False
        )[0]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for row in result.boxes.data.detach().cpu().numpy():
                bx0, by0, bx1, by1, score, cls = map(float, row[:6])
                cls = int(cls)
                if not np.isfinite(row[:6]).all():
                    continue
                gx0, gy0, gx1, gy1 = map(float, (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0))
                c_name = self.class_names.get(cls, f"shape_{cls}")
                shape_pts, shape_type = generate_shape_polygon(c_name, [gx0, gy0, gx1, gy1], image=frame)
                detections.append({
                    "class_id": cls,
                    "class_name": c_name,
                    "shape_type": shape_type,
                    "confidence": float(score),
                    "timber_id": 1,
                    "bbox_xyxy_px": [gx0, gy0, gx1, gy1],
                    "center_px": [(gx0 + gx1) / 2, (gy0 + gy1) / 2],
                    "shape_points_px": shape_pts,
                    "dimensions_px": {"width": float(gx1 - gx0), "height": float(gy1 - gy0)},
                    "coordinate_space": "undistorted_image",
                })
        return [detections]

    def metadata(self):
        return {
            "model": self.model_path, "device": str(self.device),
            "confidence_threshold": self.confidence, "image_size": self.image_size,
            "iou_threshold": self.iou,
            "classes": self.class_names
        }


def project_detections(detections, homography, scale=1.0, origin=(0.0, 0.0)):
    """Map detected shape polygons and bounding boxes into real-world mm coordinates."""
    if not np.isfinite(homography).all() or not 0 < scale <= 1:
        raise ValueError("Invalid homography or parallax scale")
    projected = []
    for detection in detections:
        item = dict(detection)
        x0, y0, x1, y1 = item["bbox_xyxy_px"]
        
        # 1. Map bounding box & center
        bbox_pts = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1], item["center_px"]],
                              dtype=np.float32).reshape(-1, 1, 2)
        mm_bbox = cv2.perspectiveTransform(bbox_pts, homography).reshape(-1, 2)
        mm_bbox = np.asarray(origin) + (mm_bbox - np.asarray(origin)) * scale

        # 2. Map full shape polygon (ellipse/rectangle/triangle vertices)
        shape_px = item.get("shape_points_px")
        if shape_px is None or len(shape_px) == 0:
            shape_px = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        raw_shape = np.asarray(shape_px, dtype=np.float32).reshape(-1, 1, 2)
        mm_shape = cv2.perspectiveTransform(raw_shape, homography).reshape(-1, 2)
        mm_shape = np.asarray(origin) + (mm_shape - np.asarray(origin)) * scale

        # Calculate metric dimensions (mm)
        w_mm = float(np.linalg.norm(mm_bbox[1] - mm_bbox[0]))
        h_mm = float(np.linalg.norm(mm_bbox[2] - mm_bbox[1]))

        item["bbox_corners_mm"] = mm_bbox[:4].tolist()
        item["center_mm"] = mm_bbox[4].tolist()
        item["shape_points_mm"] = mm_shape.tolist()
        item["dimensions_mm"] = {
            "width": round(w_mm, 2),
            "height": round(h_mm, 2),
            "size_str": f"{w_mm:.1f}x{h_mm:.1f}mm"
        }
        item["parallax_corrected"] = scale != 1.0
        projected.append(item)
    return projected


def draw_detections(display, detections):
    """Draw class-specific shapes (Ellipse, Rectangle, Triangle) with timber badges."""
    for detection in detections:
        cls = detection.get("class_id", 0)
        c_name = detection.get("class_name", "shape").lower()
        t_id = detection.get("timber_id", 1)
        conf = detection.get("confidence", 0.0)
        color = DEFAULT_COLORS.get(cls, (0, 255, 255))
        
        # Draw the exact geometric shape
        if "shape_points_px" in detection and len(detection["shape_points_px"]) >= 3:
            pts = np.round(np.asarray(detection["shape_points_px"])).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts], True, color, 3, cv2.LINE_AA)
        else:
            # Fallback to rectangle box
            x0, y0, x1, y1 = (int(round(v)) for v in detection["bbox_xyxy_px"])
            cv2.rectangle(display, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)

        # Draw Center Crosshair
        cx, cy = int(round(detection["center_px"][0])), int(round(detection["center_px"][1]))
        cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

        # Label badge with Timber ID, Shape Type & Confidence
        x0, y0 = int(round(detection["bbox_xyxy_px"][0])), int(round(detection["bbox_xyxy_px"][1]))
        label_text = f"T{t_id}: {c_name.capitalize()} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.60
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        label_y0 = max(0, y0 - th - 10)
        cv2.rectangle(display, (x0, label_y0), (x0 + tw + 10, label_y0 + th + 8), color, -1)
        cv2.putText(display, label_text, (x0 + 5, label_y0 + th + 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return display
