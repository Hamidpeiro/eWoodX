"""
================================================================================
eWoodX - Grasshopper GhPython Component: Single-Timber Reader & Live Capture
================================================================================

Description:
  Reads timber geometry, dimensions, and hand-drawn YOLO annotations from JSON.
  Features a live Button input to remotely trigger capture and measurement
  in 03_timber_measurement_handdraw_yolo.py.

  Timber Number 0 is strictly reserved for Grasshopper captures:
    - Capturing from Grasshopper updates 'timber_00_measurement.json' & 'timber_00.jpg'.
    - Does NOT alter the manual camera counter (timbers 1, 2, 3... remain intact).
    - Designed specifically to work with this single JSON.

How to use in Grasshopper:
1. Place a "GhPython Script" component on the Grasshopper canvas.
2. Configure Component Inputs:
     - capture    : (Item/List Access, Type Hint: bool) [Button: press to capture live frame]
                    (Also supports names: refresh, btn_capture, button, trigger)
     - json_path  : (Item Access, Type Hint: str, Optional)
                    [Folder or single JSON. Leave empty to automatically use reserved timber_00_measurement.json]
     - index      : (Item Access, Type Hint: int, Optional)
                    [Index of timber if file has multiple, default: 0]
     - host       : (Item Access, Type Hint: str, Optional) [Default: 127.0.0.1]
     - port       : (Item Access, Type Hint: int, Optional) [Default: 9998]

3. Configure Component Outputs (Grafted DataTree {0} per timber for full compatibility):
     - timber_contour  : PolylineCurve outline of the timber
     - timber_corners  : 4 Point3d corner points
     - timber_holes    : Inner cutout / hole curves
     - timber_id       : Timber ID (0 for Grasshopper captures)
     - length_mm       : Calibrated length (mm)
     - width_mm        : Calibrated width (mm)
     - thickness_mm    : Timber thickness (mm)
     - surface_area_cm2: Surface area (cm²)
     - side_lengths_mm : Length of 4 sides (mm)
     - color_rgb       : System.Drawing.Color / RGB
     - color_hex       : Hex color string (#rrggbb)
     - defects         : List of defect Point3d
     - drawing_curves  : Hand-drawn shape curves (ellipse, rect, triangle)
     - drawing_centers : Center Point3d of hand-drawn shapes
     - drawing_classes : Class names of shapes ('ellipse', 'rectangle', 'triangle')
     - drawing_sizes   : Dimension string of shapes (e.g. '120.5x80.2mm')
     - drawing_confidences: YOLO detection confidence scores (0.0 - 1.0)
     - table_boundary  : PolylineCurve of physical table workspace
     - markers_corners : ArUco marker boundary curves
     - file_path       : Path to the loaded JSON file
     - status          : Live capture status & connection diagnostics
     - info            : Formatted summary text

4. Paste this entire code into the GhPython component editor.
================================================================================
"""

import os
import glob
import json
import re
import socket
import time
import datetime

# Rhino, Grasshopper & .NET imports with safe fallback for testing outside Rhino
try:
    import Rhino.Geometry as rg
    import System.Drawing as sd
    import Grasshopper
    from Grasshopper.Kernel.Data import GH_Path
    from Grasshopper import DataTree
    IN_RHINO = True
    HAS_DATATREE = True
except ImportError:
    rg = None
    sd = None
    DataTree = None
    GH_Path = None
    IN_RHINO = False
    HAS_DATATREE = False


# ============================================================
# DATATREE & VALUE EXTRACTION HELPERS
# ============================================================

def make_grafted_tree(items_per_timber):
    """
    Convert a list of items (where item i belongs to timber i)
    into a Grasshopper DataTree with branch paths {0}, {1}...
    """
    if not HAS_DATATREE or DataTree is None:
        return items_per_timber

    tree = DataTree[object]()
    for i, val in enumerate(items_per_timber):
        path = GH_Path(i)
        if isinstance(val, (list, tuple)):
            for sub_val in val:
                tree.Add(sub_val, path)
        elif val is not None:
            tree.Add(val, path)
    return tree


def extract_bool(val):
    """
    Safely extract boolean state from any GH_Boolean, System.Boolean,
    list, DataTree, or primitive. Essential for Grasshopper Button components.
    """
    if val is None:
        return False
    # GH_Boolean or wrapped Goo with .Value property
    if hasattr(val, "Value"):
        try:
            return bool(val.Value)
        except Exception:
            pass
    # DataTree with .AllData()
    if hasattr(val, "AllData"):
        try:
            for item in val.AllData():
                if extract_bool(item):
                    return True
        except Exception:
            pass
        return False
    # Python list or tuple
    if isinstance(val, (list, tuple)):
        for item in val:
            if extract_bool(item):
                return True
        return False
    # .NET collection / IEnumerable (e.g. List[Boolean])
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            for item in val:
                if extract_bool(item):
                    return True
        except Exception:
            pass
        return False
    # Standard Python bool
    if isinstance(val, bool):
        return val
    # Numeric
    if isinstance(val, (int, float)):
        return bool(val)
    # String representation
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "capture", "pressed")
    try:
        s = str(val).strip().lower()
        if s in ("true", "1", "yes"):
            return True
    except Exception:
        pass
    return False


def extract_single_string(val):
    """Safely extract a single string from any DataTree, list, or primitive."""
    if val is None:
        return None
    if hasattr(val, "Value"):
        try:
            return str(val.Value).strip()
        except Exception:
            pass
    if hasattr(val, "AllData"):
        try:
            for item in val.AllData():
                s = extract_single_string(item)
                if s:
                    return s
        except Exception:
            pass
        return None
    if isinstance(val, (list, tuple)):
        for item in val:
            s = extract_single_string(item)
            if s:
                return s
        return None
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            for item in val:
                s = extract_single_string(item)
                if s:
                    return s
        except Exception:
            pass
        return None
    s_val = str(val).strip().strip('"').strip("'")
    return s_val if s_val else None


def find_default_capture_dir():
    """Locate the captures folder across common workspace directories."""
    search_dirs = [
        r"C:\Users\hamid\Documents\GitHub\eWoodX\Webcam detection\sample_output\captures",
        r"C:\Users\hamid\Documents\GitHub\eWoodX\Webcam detection\sample_output",
        os.path.expanduser(r"~\Documents\GitHub\eWoodX\Webcam detection\sample_output\captures"),
        os.path.expanduser(r"~\Documents\GitHub\eWoodX\sample_output\captures"),
        os.path.expanduser(r"~\Documents\GitHub\eWoodX\sample_output"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_output", "captures") if "__file__" in globals() and __file__ else "",
        r".\sample_output\captures",
        r".\captures"
    ]
    for s_dir in search_dirs:
        if s_dir and os.path.isdir(s_dir):
            return os.path.normpath(s_dir)
    return None


def natural_sort_key(s):
    """Sort strings with numbers naturally (timber_1, timber_2, ..., timber_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


# ============================================================
# REMOTE CAPTURE TRIGGER
# ============================================================

def trigger_remote_capture(host="127.0.0.1", port=9998, number=0, timeout=15.0):
    """
    Remotely signal 03_timber_measurement_handdraw_yolo.py to capture and measure.
    Uses TCP socket first (port 9998). If socket fails, uses capture_trigger.json fallback.
    Returns (success: bool, status_msg: str, json_path: str)
    """
    # 1. Primary Method: TCP Socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, int(port)))
        payload = json.dumps({"action": "capture", "number": int(number)}) + "\n"
        sock.sendall(payload.encode("utf-8"))

        raw_resp = sock.recv(4096)
        sock.close()

        if raw_resp:
            resp_str = raw_resp.decode("utf-8", errors="ignore").strip()
            data = json.loads(resp_str)
            if data.get("status") == "success":
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                j_path = data.get("json_path", "")
                return True, f"Capture OK: Timber #{number:02d} captured and saved at {now_str}", j_path
            else:
                return False, f"Capture error from camera script: {data.get('message', 'Unknown error')}", ""
    except Exception as e:
        # Socket failed, fall through to file trigger
        pass

    # 2. Fallback Method: File-based trigger
    cap_dir = find_default_capture_dir()
    if cap_dir and os.path.isdir(cap_dir):
        trig_file = os.path.join(cap_dir, "capture_trigger.json")
        status_file = os.path.join(cap_dir, "capture_status.json")
        target_json = os.path.join(cap_dir, f"timber_{number:02d}_measurement.json")
        start_time = time.time()
        prev_mtime = os.path.getmtime(target_json) if os.path.exists(target_json) else 0

        try:
            with open(trig_file, "w", encoding="utf-8") as tf:
                json.dump({"action": "capture", "number": int(number), "time": start_time}, tf)

            while (time.time() - start_time) < timeout:
                time.sleep(0.15)
                if os.path.exists(status_file):
                    try:
                        with open(status_file, "r", encoding="utf-8") as sf:
                            sdata = json.load(sf)
                        os.remove(status_file)
                        if sdata.get("status") == "success":
                            now_str = datetime.datetime.now().strftime("%H:%M:%S")
                            return True, f"Capture OK: Timber #{number:02d} updated via file trigger at {now_str}", target_json
                    except Exception:
                        pass

                if os.path.exists(target_json) and os.path.getmtime(target_json) > prev_mtime:
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    return True, f"Capture OK: Timber #{number:02d} updated at {now_str}", target_json

            return False, "Capture timeout: Camera script did not respond in time.", ""
        except Exception as e:
            return False, f"File trigger error: {e}", ""

    return False, "Camera script is NOT running! Please run '03_timber_measurement_handdraw_yolo.py' in a terminal.", ""


def resolve_single_json_file(raw_input):
    """
    Resolve the single JSON file to load.
    Prioritizes reserved timber #00 (timber_00_measurement.json).
    If timber_00 does not exist yet, falls back to the most recent measurement JSON.
    """
    clean_path = extract_single_string(raw_input)
    cap_dir = None

    if clean_path:
        clean_path = os.path.normpath(clean_path)
        if os.path.isfile(clean_path):
            return clean_path
        elif os.path.isdir(clean_path):
            cap_dir = clean_path

    if not cap_dir:
        cap_dir = find_default_capture_dir()

    if cap_dir and os.path.isdir(cap_dir):
        # 1. Target reserved timber_00_measurement.json
        t00 = os.path.join(cap_dir, "timber_00_measurement.json")
        if os.path.isfile(t00):
            return t00
        t00_sub = os.path.join(cap_dir, "captures", "timber_00_measurement.json")
        if os.path.isfile(t00_sub):
            return t00_sub

        # 2. If timber_00 not created yet, preview the latest available capture
        all_measurements = glob.glob(os.path.join(cap_dir, "timber_*_measurement.json"))
        if not all_measurements and os.path.isdir(os.path.join(cap_dir, "captures")):
            all_measurements = glob.glob(os.path.join(cap_dir, "captures", "timber_*_measurement.json"))
        if not all_measurements:
            all_measurements = glob.glob(os.path.join(cap_dir, "*.json"))

        if all_measurements:
            # Sort naturally so highest number (latest) is picked
            all_measurements.sort(key=natural_sort_key)
            return all_measurements[-1]

    return None


def parse_single_timber(data, file_path):
    """Extract geometry and properties from a single timber JSON dictionary."""
    t_id = data.get("timber_id", data.get("capture_id", 0))
    l_mm = float(data.get("length_mm", 0.0))
    w_mm = float(data.get("width_mm", 0.0))
    th_mm = float(data.get("timber_thickness_mm", data.get("thickness_mm", 0.0)))
    sides = [float(s) for s in data.get("side_lengths_mm", [])]

    if "surface_area_cm2" in data:
        area_cm2 = float(data["surface_area_cm2"])
    elif "surface_area_mm2" in data:
        area_cm2 = float(data["surface_area_mm2"]) / 100.0
    else:
        area_cm2 = (l_mm * w_mm) / 100.0

    # Color
    color_dict = data.get("color_rgb", {})
    r = int(color_dict.get("R", 200))
    g = int(color_dict.get("G", 160))
    b = int(color_dict.get("B", 120))
    c_hex = data.get("color_hex", f"#{r:02x}{g:02x}{b:02x}")
    c_rgb = sd.Color.FromArgb(r, g, b) if sd is not None else (r, g, b)

    # Hand-drawn YOLO shapes
    drawing_crvs = []
    drawing_centers_pts = []
    drawing_cls_names = []
    drawing_conf_vals = []
    drawing_sizes = []

    raw_drawings = data.get("hand_drawn_shapes", [])
    for drw in raw_drawings:
        c_name = drw.get("class_name", "unknown")
        conf = float(drw.get("confidence", 0.0))
        c_mm = drw.get("center_mm", [0.0, 0.0])
        corners_mm = drw.get("bbox_corners_mm", [])

        # Dimensions
        w_d_mm = 0.0
        h_d_mm = 0.0
        if len(corners_mm) >= 4:
            dx1 = corners_mm[1][0] - corners_mm[0][0]
            dy1 = corners_mm[1][1] - corners_mm[0][1]
            w_d_mm = (dx1**2 + dy1**2)**0.5
            dx2 = corners_mm[2][0] - corners_mm[1][0]
            dy2 = corners_mm[2][1] - corners_mm[1][1]
            h_d_mm = (dx2**2 + dy2**2)**0.5

        shape_pts_mm = drw.get("shape_points_mm") or drw.get("bbox_corners_mm", [])

        if IN_RHINO and rg is not None:
            drawing_centers_pts.append(rg.Point3d(float(c_mm[0]), float(c_mm[1]), 0.0))
            if len(shape_pts_mm) >= 3:
                d_pts = [rg.Point3d(float(p[0]), float(p[1]), 0.0) for p in shape_pts_mm]
                d_pts.append(d_pts[0])  # close polyline
                drawing_crvs.append(rg.Polyline(d_pts).ToPolylineCurve())
            else:
                drawing_crvs.append(None)
        else:
            drawing_centers_pts.append(c_mm)
            drawing_crvs.append(shape_pts_mm)

        dim_info = drw.get("dimensions_mm", {})
        size_str = dim_info.get("size_str", f"{w_d_mm:.1f}x{h_d_mm:.1f}mm")

        drawing_cls_names.append(c_name)
        drawing_conf_vals.append(conf)
        drawing_sizes.append(size_str)

    # Geometry
    contour_crv = None
    corners_pts = []
    defect_pts = []
    holes_crvs = []

    raw_holes = data.get("holes_mm", data.get("inner_contours_mm", []))

    if IN_RHINO and rg is not None:
        raw_contour = data.get("contour_mm", data.get("corners_mm", []))
        if raw_contour and len(raw_contour) >= 3:
            pts = [rg.Point3d(float(p[0]), float(p[1]), 0.0) for p in raw_contour]
            pts.append(pts[0])  # close polyline
            contour_crv = rg.Polyline(pts).ToPolylineCurve()

        for h_pts_coords in raw_holes:
            if h_pts_coords and len(h_pts_coords) >= 3:
                h_pts = [rg.Point3d(float(p[0]), float(p[1]), 0.0) for p in h_pts_coords]
                h_pts.append(h_pts[0])
                holes_crvs.append(rg.Polyline(h_pts).ToPolylineCurve())

        for c_pt in data.get("corners_mm", []):
            if len(c_pt) >= 2:
                corners_pts.append(rg.Point3d(float(c_pt[0]), float(c_pt[1]), 0.0))

        for d in data.get("defects", []):
            if isinstance(d, dict):
                dx = float(d.get("center_mm", [0, 0])[0])
                dy = float(d.get("center_mm", [0, 0])[1])
                defect_pts.append(rg.Point3d(dx, dy, 0.0))
            elif isinstance(d, (list, tuple)) and len(d) >= 2:
                defect_pts.append(rg.Point3d(float(d[0]), float(d[1]), 0.0))
    else:
        raw_contour = data.get("contour_mm", data.get("corners_mm", []))
        if raw_contour and len(raw_contour) >= 3:
            contour_crv = raw_contour
        holes_crvs = raw_holes
        corners_pts = data.get("corners_mm", [])
        defect_pts = data.get("defects", [])

    return {
        "timber_id": t_id,
        "length_mm": l_mm,
        "width_mm": w_mm,
        "thickness_mm": th_mm,
        "surface_area_cm2": area_cm2,
        "side_lengths_mm": sides,
        "color_rgb": c_rgb,
        "color_hex": c_hex,
        "contour": contour_crv,
        "holes": holes_crvs,
        "hole_count": len(holes_crvs),
        "corners": corners_pts,
        "defects": defect_pts,
        "drawing_curves": drawing_crvs,
        "drawing_centers": drawing_centers_pts,
        "drawing_classes": drawing_cls_names,
        "drawing_confidences": drawing_conf_vals,
        "drawing_sizes": drawing_sizes,
        "raw": data,
        "file": file_path
    }


# ============================================================
# MAIN COMPONENT EXECUTION
# ============================================================

# 1. Detect Button Press (Supports any input name: capture, refresh, btn_capture, button, trigger)
btn_pressed = False
btn_name_found = None
for candidate in ["capture", "refresh", "btn_capture", "capture_button", "button", "trigger", "btn"]:
    if candidate in globals() and globals()[candidate] is not None:
        raw_val = globals()[candidate]
        if extract_bool(raw_val):
            btn_pressed = True
            btn_name_found = candidate
            break

# Remote server host/port
remote_host = str(extract_single_string(globals().get("host", "127.0.0.1")) or "127.0.0.1")
try:
    remote_port = int(globals().get("port", 9998))
except Exception:
    remote_port = 9998

status_messages = []
captured_file_path = None

# 2. Trigger Capture if Button Pressed
if btn_pressed:
    print(f"[GhTimberReader] Button '{btn_name_found}' pressed! Requesting capture from camera ({remote_host}:{remote_port})...")
    cap_ok, cap_msg, cap_json = trigger_remote_capture(host=remote_host, port=remote_port, number=0, timeout=15.0)
    print(f"[GhTimberReader] {cap_msg}")
    status_messages.append(cap_msg)
    if cap_ok and cap_json and os.path.isfile(cap_json):
        captured_file_path = cap_json

# 3. Resolve Target Single JSON
if captured_file_path:
    single_json_path = captured_file_path
else:
    # Resolve from input or default
    raw_path_input = None
    for p_name in ["json_path", "json_paths", "folder", "path", "file", "json"]:
        if p_name in globals() and globals()[p_name] is not None:
            raw_path_input = globals()[p_name]
            break
    single_json_path = resolve_single_json_file(raw_path_input)

# 4. Parse the Single JSON
parsed_timber = None
root_data = None

if single_json_path and os.path.exists(single_json_path):
    try:
        with open(single_json_path, "r", encoding="utf-8") as f:
            root_data = json.load(f)
            root_data["_file_path"] = single_json_path

            # If JSON contains a list of timbers, pick selected index or timber #0
            if "timbers" in root_data and isinstance(root_data["timbers"], list) and len(root_data["timbers"]) > 0:
                timber_list = root_data["timbers"]
                sel_idx = globals().get("index", globals().get("selected_index", 0))
                try:
                    t_idx = int(sel_idx) if sel_idx is not None else 0
                    if t_idx < 0 or t_idx >= len(timber_list):
                        t_idx = 0
                except Exception:
                    t_idx = 0
                t_data = timber_list[t_idx]
            else:
                t_data = root_data

            # Inherit markers / reference frame if needed
            if "markers_world_mm" in root_data and "markers_world_mm" not in t_data:
                t_data["markers_world_mm"] = root_data["markers_world_mm"]

            parsed_timber = parse_single_timber(t_data, single_json_path)
    except Exception as exc:
        err_msg = f"Error parsing JSON ({os.path.basename(single_json_path)}): {exc}"
        print(f"[GhTimberReader] {err_msg}")
        status_messages.append(err_msg)

# 5. Table & Markers Geometry
table_boundary = None
markers_corners_list = []

if IN_RHINO and rg is not None:
    if root_data and "markers_world_mm" in root_data:
        for m_id, m_info in root_data["markers_world_mm"].items():
            if "corners_mm" in m_info and len(m_info["corners_mm"]) == 4:
                m_pts = [rg.Point3d(float(p[0]), float(p[1]), 0.0) for p in m_info["corners_mm"]]
                m_pts.append(m_pts[0])
                markers_corners_list.append(rg.Polyline(m_pts).ToPolylineCurve())

    # Physical Table Rectangle (0,0 to 1780,1040)
    table_pts = [
        rg.Point3d(0.0, 0.0, 0.0),
        rg.Point3d(1780.0, 0.0, 0.0),
        rg.Point3d(1780.0, 1040.0, 0.0),
        rg.Point3d(0.0, 1040.0, 0.0),
        rg.Point3d(0.0, 0.0, 0.0)
    ]
    table_boundary = rg.Polyline(table_pts).ToPolylineCurve()

# 6. Populate Output Variables (Grafted Tree {0} for full Grasshopper compatibility)
if parsed_timber is not None:
    timber_contour = timber_contours = make_grafted_tree([parsed_timber["contour"]])
    timber_holes = inner_cuts = holes = make_grafted_tree([parsed_timber["holes"]])
    timber_corners = corners = make_grafted_tree([parsed_timber["corners"]])
    timber_id = timber_ids = make_grafted_tree([parsed_timber["timber_id"]])
    length_mm = lengths_mm = make_grafted_tree([parsed_timber["length_mm"]])
    width_mm = widths_mm = make_grafted_tree([parsed_timber["width_mm"]])
    thickness_mm = thicknesses_mm = make_grafted_tree([parsed_timber["thickness_mm"]])
    surface_area_cm2 = surface_areas_cm2 = make_grafted_tree([parsed_timber["surface_area_cm2"]])
    side_lengths_mm = make_grafted_tree([parsed_timber["side_lengths_mm"]])
    color_rgb = colors_rgb = make_grafted_tree([parsed_timber["color_rgb"]])
    color_hex = colors_hex = make_grafted_tree([parsed_timber["color_hex"]])
    defects = make_grafted_tree([parsed_timber["defects"]])
    drawing_curves = hand_drawn_shapes = make_grafted_tree([parsed_timber["drawing_curves"]])
    drawing_centers = make_grafted_tree([parsed_timber["drawing_centers"]])
    drawing_classes = make_grafted_tree([parsed_timber["drawing_classes"]])
    drawing_sizes = make_grafted_tree([parsed_timber["drawing_sizes"]])
    drawing_confidences = make_grafted_tree([parsed_timber["drawing_confidences"]])
    file_path = file_paths = make_grafted_tree([single_json_path])
    raw_data = make_grafted_tree([parsed_timber["raw"]])
    markers_corners = make_grafted_tree(markers_corners_list)

    fname = os.path.basename(single_json_path)
    drw_cls = parsed_timber["drawing_classes"]
    drw_str = f" | Drawings: {', '.join(drw_cls)}" if drw_cls else ""
    info = (
        f"Timber #{parsed_timber['timber_id']}: {parsed_timber['length_mm']:.1f} x {parsed_timber['width_mm']:.1f} mm, "
        f"T={parsed_timber['thickness_mm']:.1f}mm{drw_str}\n"
        f"File: {fname}"
    )
    if not status_messages:
        status_messages.append(f"READY: Loaded {fname} (Timber #{parsed_timber['timber_id']})")
    
    # Print status summary to Grasshopper 'out' parameter
    print(f"[GhTimberReader] Active File: {fname} (Timber #{parsed_timber['timber_id']}) -> {parsed_timber['length_mm']:.1f} x {parsed_timber['width_mm']:.1f} mm{drw_str}")
else:
    timber_contour = timber_contours = make_grafted_tree([])
    timber_holes = inner_cuts = holes = make_grafted_tree([])
    timber_corners = corners = make_grafted_tree([])
    timber_id = timber_ids = make_grafted_tree([])
    length_mm = lengths_mm = make_grafted_tree([])
    width_mm = widths_mm = make_grafted_tree([])
    thickness_mm = thicknesses_mm = make_grafted_tree([])
    surface_area_cm2 = surface_areas_cm2 = make_grafted_tree([])
    side_lengths_mm = make_grafted_tree([])
    color_rgb = colors_rgb = make_grafted_tree([])
    color_hex = colors_hex = make_grafted_tree([])
    defects = make_grafted_tree([])
    drawing_curves = hand_drawn_shapes = make_grafted_tree([])
    drawing_centers = make_grafted_tree([])
    drawing_classes = make_grafted_tree([])
    drawing_sizes = make_grafted_tree([])
    drawing_confidences = make_grafted_tree([])
    file_path = file_paths = make_grafted_tree([])
    raw_data = make_grafted_tree([])
    markers_corners = make_grafted_tree(markers_corners_list)

    info = "No JSON found. Connect a Button to 'refresh' / 'capture' and press to capture Timber #00 from live camera."
    if not status_messages:
        status_messages.append("WAITING: Press Button to capture Timber #00 from 03_timber_measurement_handdraw_yolo.py")
    print("[GhTimberReader] WAITING: No timber JSON found. Press Button to capture Timber #00 from camera.")

status = " | ".join(status_messages)
