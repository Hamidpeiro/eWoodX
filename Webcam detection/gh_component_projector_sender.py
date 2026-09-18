"""
================================================================================
eWoodX - Grasshopper GhPython Component 2: Projector Live Sender
================================================================================

How to use in Grasshopper:
1. Place a "GhPython Script" component on the Grasshopper canvas.
2. Configure Component Inputs (Set Type Hints & List Access accordingly):
     - main_border   : (List or Item Access, Type Hint: Curve)   [Timber contour outline]
     - timber_color  : (Item Access, Type Hint: Color / str)     [Default: White/Yellow]
     - timber_width  : (Item Access, Type Hint: int)             [Stroke width in px, Default: 2]
     - cut           : (List or Item Access, Type Hint: Curve)   [Cutting curves/lines]
     - cut_color     : (Item Access, Type Hint: Color / str)     [Default: #FF2020 Red]
     - cut_thickness : (Item Access, Type Hint: int)             [Cut line width in px, Default: 3]
     - mill          : (List or Item Access, Type Hint: Curve)   [Milling curves/pockets/hatches]
     - mill_color    : (Item Access, Type Hint: Color / str)     [Default: #00D0FF Cyan]
     - mill_thickness: (Item Access, Type Hint: int)             [Mill line width in px, Default: 2]
     - points        : (List or Item Access, Type Hint: Point3d) [Drill / mark points]
     - point_color   : (Item Access, Type Hint: Color / str)     [Default: #00FF7F Green]
     - drill_size    : (Item Access, Type Hint: int)             [Hole circle radius in px, Default: 6]
     - labels        : (List or Item Access, Type Hint: str)     [Text annotations / notes]
     - label_points  : (List or Item Access, Type Hint: Point3d) [Positions for text labels]
     - label_color   : (Item Access, Type Hint: Color / str)     [Default: #FFA500 Amber]
     - label_scale   : (Item Access, Type Hint: float)           [Text font scale, Default: 0.45]
     - thickness     : (Item Access, Type Hint: float)           [Physical timber thickness in mm]
     - active        : (Item Access, Type Hint: bool)            [Optional toggle, Default: True]
     - host / port   : (Optional, automatically hardcoded to 127.0.0.1:9999)

3. Configure Component Outputs:
     - status        : Diagnostic info & live streaming status

4. Paste the entire code below into the GhPython component editor.
================================================================================
"""

import json
import socket
import time
import datetime

# Rhino imports with safe fallback for testing
try:
    import Rhino.Geometry as rg
    import System.Drawing as sd
    IN_RHINO = True
except ImportError:
    rg = None
    sd = None
    IN_RHINO = False


# ============================================================
# DATATREE & SCALAR EXTRACTION HELPERS
# ============================================================

def flatten_input(val):
    """Recursively flatten any Grasshopper DataTree, list, or nested collection."""
    if val is None:
        return []
    if hasattr(val, "AllData"):
        try:
            return list(val.AllData())
        except Exception:
            pass
    if hasattr(val, "Branches"):
        try:
            out = []
            for b in val.Branches:
                for item in b:
                    out.append(item)
            return out
        except Exception:
            pass
    if isinstance(val, (list, tuple)):
        out = []
        for item in val:
            if isinstance(item, (list, tuple)) or hasattr(item, "AllData") or hasattr(item, "Branches"):
                out.extend(flatten_input(item))
            else:
                out.append(item)
        return out
    return [val]


def extract_single_scalar(val, default=None):
    """Safely extract a single scalar from any DataTree, list, or primitive."""
    if val is None:
        return default
    if isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, str):
        try:
            return val.strip().strip('"').strip("'")
        except Exception:
            return val

    # 1. Try AllData
    if hasattr(val, "AllData"):
        try:
            for item in val.AllData():
                res = extract_single_scalar(item, None)
                if res is not None:
                    return res
        except Exception:
            pass

    # 2. Try Branches
    if hasattr(val, "Branches"):
        try:
            for b in val.Branches:
                for item in b:
                    res = extract_single_scalar(item, None)
                    if res is not None:
                        return res
        except Exception:
            pass

    # 3. Try Branch(0)
    if hasattr(val, "Branch"):
        try:
            b = val.Branch(0)
            if len(b) > 0:
                return extract_single_scalar(b[0], default)
        except Exception:
            pass

    # 4. Try indexing or iteration
    try:
        for item in val:
            res = extract_single_scalar(item, None)
            if res is not None:
                return res
    except Exception:
        pass

    try:
        return val[0]
    except Exception:
        pass

    return default


def to_int(val, default_val=0):
    """Convert any DataTree, string, float, or integer to a clean integer."""
    if val is None:
        return default_val
    scalar = extract_single_scalar(val, None)
    if scalar is None:
        return default_val
    try:
        return int(float(str(scalar).strip().strip('"').strip("'")))
    except (ValueError, TypeError):
        return default_val


def to_float(val, default_val=None):
    """Convert any DataTree, string, or number to a clean float."""
    if val is None:
        return default_val
    scalar = extract_single_scalar(val, None)
    if scalar is None:
        return default_val
    try:
        return float(str(scalar).strip().strip('"').strip("'"))
    except (ValueError, TypeError):
        return default_val



def to_bool(val, default_val=True):
    """Convert any DataTree or value to boolean."""
    scalar = extract_single_scalar(val, default_val)
    if scalar is None:
        return default_val
    if isinstance(scalar, bool):
        return scalar
    s_str = str(scalar).strip().lower()
    return s_str in ("true", "1", "yes", "on")


def extract_rgb(color_in, default_rgb=[255, 255, 255]):
    """Convert .NET Color, hex string, DataTree[Color], or tuple to [R, G, B] integer list."""
    if color_in is None:
        return default_rgb

    # If DataTree of Colors, unwrap first element
    c_obj = extract_single_scalar(color_in, default_rgb)
    if c_obj is None:
        return default_rgb

    # 1. System.Drawing.Color from Grasshopper Colour Swatch / Picker
    if hasattr(c_obj, "R") and hasattr(c_obj, "G") and hasattr(c_obj, "B"):
        return [int(c_obj.R), int(c_obj.G), int(c_obj.B)]

    # 2. Hex String e.g. "#FF0000" or "FF0000"
    if isinstance(c_obj, str):
        c_str = c_obj.strip().strip('"').strip("'").lstrip("#")
        if len(c_str) >= 6:
            try:
                r = int(c_str[0:2], 16)
                g = int(c_str[2:4], 16)
                b = int(c_str[4:6], 16)
                return [r, g, b]
            except Exception:
                pass

    # 3. Tuple / List [r, g, b]
    if isinstance(c_obj, (list, tuple)) and len(c_obj) >= 3:
        try:
            return [int(c_obj[0]), int(c_obj[1]), int(c_obj[2])]
        except Exception:
            pass

    return default_rgb


def curve_to_polyline_pts(curve_obj, max_chord_error=0.5):
    """Convert any Rhino Curve, Line, Arc, or Polyline into a list of [X, Y] points (mm)."""
    if curve_obj is None:
        return []

    # If already a list of points or raw coordinate lists
    if isinstance(curve_obj, (list, tuple)):
        if len(curve_obj) > 0 and isinstance(curve_obj[0], (list, tuple, rg.Point3d if rg else tuple)):
            pts_out = []
            for p in curve_obj:
                if hasattr(p, "X") and hasattr(p, "Y"):
                    pts_out.append([round(float(p.X), 2), round(float(p.Y), 2)])
                elif len(p) >= 2:
                    pts_out.append([round(float(p[0]), 2), round(float(p[1]), 2)])
            return pts_out

    if not IN_RHINO or rg is None:
        return []

    # If Rhino Curve
    if isinstance(curve_obj, rg.Curve):
        pline_try = None
        success, pline = curve_obj.TryGetPolyline()
        if success and pline is not None:
            return [[round(p.X, 2), round(p.Y, 2)] for p in pline]

        poly_curve = curve_obj.ToPolyline(
            mainSegmentCount=0,
            subSegmentCount=0,
            maxAngleRadians=0.1,
            maxChordLengthRatio=0.0,
            maxAspectRatio=0.0,
            tolerance=max_chord_error,
            minEdgeLength=0.0,
            maxEdgeLength=0.0
        )
        if poly_curve:
            return [[round(poly_curve.Point(i).X, 2), round(poly_curve.Point(i).Y, 2)] for i in range(poly_curve.PointCount)]

        div_count = max(8, int(curve_obj.GetLength() / 5.0))
        params = curve_obj.DivideByCount(div_count, True)
        if params:
            return [[round(curve_obj.PointAt(t).X, 2), round(curve_obj.PointAt(t).Y, 2)] for t in params]

    elif isinstance(curve_obj, rg.Line):
        return [
            [round(curve_obj.From.X, 2), round(curve_obj.From.Y, 2)],
            [round(curve_obj.To.X, 2), round(curve_obj.To.Y, 2)]
        ]

    elif isinstance(curve_obj, rg.Polyline):
        return [[round(p.X, 2), round(p.Y, 2)] for p in curve_obj]

    return []


def point_to_xy(pt_obj):
    """Extract [X, Y] in mm from Point3d or coordinate list."""
    if pt_obj is None:
        return None
    if hasattr(pt_obj, "X") and hasattr(pt_obj, "Y"):
        return [round(float(pt_obj.X), 2), round(float(pt_obj.Y), 2)]
    if isinstance(pt_obj, (list, tuple)) and len(pt_obj) >= 2:
        return [round(float(pt_obj[0]), 2), round(float(pt_obj[1]), 2)]
    return None


# ============================================================
# HARDCODED PROJECTOR VIEWER CONNECTION SETTINGS
# ============================================================
HOST = "127.0.0.1"
PORT = 9999


# ============================================================
# MAIN SENDER EXECUTION
# ============================================================

# 1. Inputs with defaults
is_active = to_bool(globals().get("active", True), True)

raw_host = extract_single_scalar(globals().get("host", None))
if raw_host is not None:
    cleaned_host = str(raw_host).strip().strip('"').strip("'")
    if cleaned_host in ("0.0.0.0", "", "None", "localhost", "0"):
        target_host = HOST
    else:
        target_host = cleaned_host
else:
    target_host = HOST

target_port = to_int(globals().get("port", PORT), PORT)

timber_input = globals().get("main_border", None)
timber_col_in = globals().get("timber_color", None)
timber_rgb = extract_rgb(timber_col_in, [255, 255, 255])

cut_input = globals().get("cut", [])
cut_col_in = globals().get("cut_color", None)
cut_rgb = extract_rgb(cut_col_in, [255, 30, 30])

mill_input = globals().get("mill", [])
mill_col_in = globals().get("mill_color", None)
mill_rgb = extract_rgb(mill_col_in, [0, 220, 255])

points_input = globals().get("points", [])
pt_col_in = globals().get("point_color", None)
point_rgb = extract_rgb(pt_col_in, [0, 255, 100])

labels_input = globals().get("labels", [])
label_pts_input = globals().get("label_points", [])
label_col_in = globals().get("label_color", None)
label_rgb = extract_rgb(label_col_in, [255, 180, 0])

thickness_val = to_float(globals().get("thickness", None), None)

if not is_active:
    status = "Paused (active = False)"
else:
    # 2. Process Timber Contour (supports single curve, list, or grafted DataTree)
    contour_list = []
    if timber_input is not None:
        flat_timber = flatten_input(timber_input)
        for item in flat_timber:
            pts = curve_to_polyline_pts(item)
            if pts:
                contour_list.extend(pts)

    # 3. Process Cut Curves (supports list or grafted DataTree)
    cut_list = []
    if cut_input is not None:
        flat_cuts = flatten_input(cut_input)
        for item in flat_cuts:
            pts = curve_to_polyline_pts(item)
            if pts and len(pts) >= 2:
                cut_list.append(pts)

    # 4. Process Mill Curves (supports list or grafted DataTree)
    mill_list = []
    if mill_input is not None:
        flat_mills = flatten_input(mill_input)
        for item in flat_mills:
            pts = curve_to_polyline_pts(item)
            if pts and len(pts) >= 2:
                mill_list.append(pts)

    # 5. Process Drill / Alignment Points (supports list or grafted DataTree)
    pts_list = []
    if points_input is not None:
        flat_pts = flatten_input(points_input)
        for item in flat_pts:
            xy = point_to_xy(item)
            if xy:
                pts_list.append(xy)

    # 6. Process Labels
    label_items = []
    if labels_input is not None:
        flat_labels = flatten_input(labels_input)
        flat_pos = flatten_input(label_pts_input) if label_pts_input is not None else []

        for i, txt in enumerate(flat_labels):
            if txt:
                pos = [200, 200]
                if i < len(flat_pos) and flat_pos[i]:
                    xy = point_to_xy(flat_pos[i])
                    if xy:
                        pos = xy
                label_items.append({"text": str(txt), "pos": pos})

    # 7. Stroke Thicknesses & Sizes (Safely unwraps any DataTree / Int32 / float slider)
    cut_th = to_int(globals().get("cut_thickness", globals().get("cut_width", 3)), 3)
    mill_th = to_int(globals().get("mill_thickness", globals().get("mill_width", 2)), 2)
    timber_th = to_int(globals().get("timber_thickness_px", globals().get("timber_width", 2)), 2)
    drill_rad = to_int(globals().get("drill_size", globals().get("point_size", globals().get("drill_radius", 6))), 6)
    label_sz = to_float(globals().get("label_scale", globals().get("text_size", 0.45)), 0.45)

    # 8. Construct Payload
    payload = {
        "timestamp": time.time(),
        "thickness_mm": thickness_val,
        "timber_contour": contour_list,
        "timber_color": timber_rgb,
        "timber_thickness_px": timber_th,
        "cut_geo": cut_list,
        "cut_color": cut_rgb,
        "cut_thickness_px": cut_th,
        "mill_geo": mill_list,
        "mill_color": mill_rgb,
        "mill_thickness_px": mill_th,
        "points_geo": pts_list,
        "point_color": point_rgb,
        "drill_radius_px": drill_rad,
        "labels": label_items,
        "label_color": label_rgb,
        "label_scale": label_sz
    }

    # 9. Send JSON frame via TCP Socket
    json_bytes = (json.dumps(payload) + "\n").encode("utf-8")
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        sock.connect((target_host, target_port))
        sock.sendall(json_bytes)
        sock.close()

        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        status = (
            f"[LIVE OK {now_str}] -> {target_host}:{target_port} | "
            f"Contour: {len(contour_list)} pts | Cuts: {len(cut_list)} | "
            f"Mills: {len(mill_list)} | Holes: {len(pts_list)} | Labels: {len(label_items)}"
        )
    except socket.timeout:
        status = f"Timeout connecting to Projector Live Viewer at {target_host}:{target_port}"
    except socket.error as e:
        status = f"Waiting for Projector Live Viewer on {target_host}:{target_port}... ({e})"
    except Exception as e:
        status = f"Error streaming data: {e}"
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass
