"""
================================================================================
eWoodX - Grasshopper GhPython Component 1: Multi-Timber JSON Reader (Grafted DataTree)
================================================================================

How to use in Grasshopper:
1. Place a "GhPython Script" component on the Grasshopper canvas.
2. Configure Component Inputs:
     - json_paths : (Item or List Access, Type Hint: str, Optional)
                    [Can be:
                      * Folder path (e.g. captures folder) -> loads ALL JSONs inside
                      * List of JSON file paths
                      * Single JSON file path
                      * Left empty / disconnected -> automatically loads ALL captures]
     - index      : (Item Access, Type Hint: int, Optional)
                    [Select specific timber index (0, 1, 2... or -1 for latest),
                     or leave disconnected to output ALL timbers grafted]
     - refresh    : (Item Access, Type Hint: bool, Optional) [Button / Timer]

3. Output Names (ALL Outputs are Grafted into Grasshopper DataTrees {0}, {1}, {2}... per Timber):
     - timber_contour  / timber_contours   : DataTree of PolylineCurve outline on branch {i}
     - timber_corners  / corners           : DataTree of 4 Point3d corner points on branch {i}
     - timber_id       / timber_ids        : DataTree of integer ID on branch {i}
     - length_mm       / lengths_mm        : DataTree of length (mm) on branch {i}
     - width_mm        / widths_mm         : DataTree of width (mm) on branch {i}
     - thickness_mm    / thicknesses_mm    : DataTree of thickness (mm) on branch {i}
     - surface_area_cm2/ surface_areas_cm2 : DataTree of surface area (cm²) on branch {i}
     - side_lengths_mm                     : DataTree of 4 side lengths (mm) on branch {i}
     - color_rgb       / colors_rgb        : DataTree of System.Drawing.Color on branch {i}
     - color_hex       / colors_hex        : DataTree of hex string on branch {i}
     - defects                             : DataTree of defect Point3d on branch {i}
     - drawing_curves  / hand_drawn_shapes : DataTree of detected shape PolylineCurves (ellipse/rect/tri)
     - drawing_centers                     : DataTree of detected shape center Point3d (mm)
     - drawing_classes                     : DataTree of class strings ('ellipse', 'rectangle', 'triangle')
     - drawing_sizes                       : DataTree of shape sizes in mm (e.g. '120.5x80.2mm')
     - drawing_confidences                 : DataTree of YOLO detection confidence scores (0.0 - 1.0)
     - table_boundary                      : PolylineCurve of physical table
     - markers_corners                     : DataTree of ArUco marker PolylineCurves
     - file_path       / file_paths        : DataTree of file path on branch {i}
     - raw_data                            : Raw dictionary data on branch {i}
     - info                                : Formatted summary text

4. Paste the entire code below into the GhPython component editor.
================================================================================
"""

import os
import glob
import json
import re

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


def make_grafted_tree(items_per_timber):
    """
    Convert a list of items (where item i belongs to timber i)
    into a Grasshopper DataTree with branch paths {0}, {1}, {2}...
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
            return s_dir
    return None


def natural_sort_key(s):
    """Sort strings with numbers naturally (timber_1, timber_2, ..., timber_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def collect_json_files(input_paths):
    """Resolve input paths (folder, list of files, single file, or empty) into list of JSON files."""
    files_to_load = []

    if input_paths is None or (isinstance(input_paths, (list, tuple)) and len(input_paths) == 0) or str(input_paths).strip() == "":
        cap_dir = find_default_capture_dir()
        if cap_dir:
            # Look in captures directly or in subfolder captures
            found = glob.glob(os.path.join(cap_dir, "timber_*_measurement.json"))
            if not found and os.path.isdir(os.path.join(cap_dir, "captures")):
                found = glob.glob(os.path.join(cap_dir, "captures", "timber_*_measurement.json"))
            if not found:
                found = glob.glob(os.path.join(cap_dir, "*.json"))
            found.sort(key=natural_sort_key)
            files_to_load = found
    else:
        raw_list = input_paths if isinstance(input_paths, (list, tuple)) else [input_paths]
        for p in raw_list:
            if not p:
                continue
            p_str = str(p).strip().strip('"').strip("'")
            if os.path.isdir(p_str):
                # 1. Look for timber_*_measurement.json
                found = glob.glob(os.path.join(p_str, "timber_*_measurement.json"))
                # 2. Look in subfolder 'captures'
                if not found and os.path.isdir(os.path.join(p_str, "captures")):
                    found = glob.glob(os.path.join(p_str, "captures", "timber_*_measurement.json"))
                # 3. Look for any json in folder
                if not found:
                    found = glob.glob(os.path.join(p_str, "*.json"))
                # 4. Recursive search
                if not found:
                    found = glob.glob(os.path.join(p_str, "**", "*.json"), recursive=True)
                found.sort(key=natural_sort_key)
                files_to_load.extend(found)
            elif os.path.isfile(p_str):
                files_to_load.append(p_str)
            elif "*" in p_str or "?" in p_str:
                found = glob.glob(p_str)
                found.sort(key=natural_sort_key)
                files_to_load.extend(found)

    seen = set()
    unique_files = []
    for f in files_to_load:
        norm = os.path.normpath(f)
        if norm not in seen and os.path.exists(norm):
            seen.add(norm)
            unique_files.append(norm)

    return unique_files


def parse_single_timber(data, file_path):
    """Extract geometry and properties from a single timber JSON dictionary."""
    t_id = data.get("timber_id", 1)
    l_mm = float(data.get("length_mm", 0.0))
    w_mm = float(data.get("width_mm", 0.0))
    th_mm = float(data.get("timber_thickness_mm", 0.0))
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

        # Calculate bounding dimensions (mm)
        w_d_mm = 0.0
        h_d_mm = 0.0
        if len(corners_mm) >= 4:
            dx1 = corners_mm[1][0] - corners_mm[0][0]
            dy1 = corners_mm[1][1] - corners_mm[0][1]
            w_d_mm = (dx1**2 + dy1**2)**0.5
            dx2 = corners_mm[2][0] - corners_mm[1][0]
            dy2 = corners_mm[2][1] - corners_mm[1][1]
            h_d_mm = (dx2**2 + dy2**2)**0.5

        # Shape geometry points (ellipse curve, rectangle, triangle)
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

    if IN_RHINO and rg is not None:
        raw_contour = data.get("contour_mm", data.get("corners_mm", []))
        if raw_contour and len(raw_contour) >= 3:
            pts = [rg.Point3d(float(p[0]), float(p[1]), 0.0) for p in raw_contour]
            pts.append(pts[0])  # close polyline
            contour_crv = rg.Polyline(pts).ToPolylineCurve()

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

# 1. Resolve inputs
raw_input = globals().get("json_paths", None)
if raw_input is None:
    raw_input = globals().get("json_path", None)
if raw_input is None:
    raw_input = globals().get("folder", None)
if raw_input is None:
    raw_input = globals().get("path", None)
if raw_input is None:
    raw_input = globals().get("file", None)
if raw_input is None:
    raw_input = globals().get("json", None)

json_files = collect_json_files(raw_input)

# 2. Parse all JSON files
list_contours = []
list_corners = []
list_ids = []
list_lengths = []
list_widths = []
list_thicknesses = []
list_areas = []
list_sides = []
list_colors_rgb = []
list_colors_hex = []
list_defects = []
list_drawing_curves = []
list_drawing_centers = []
list_drawing_classes = []
list_drawing_sizes = []
list_drawing_confidences = []
list_files = []
list_raw = []
first_data = None

for f_path in json_files:
    try:
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_file_path"] = f_path
            if first_data is None:
                first_data = data

            # Check if this JSON contains multiple timbers under "timbers" list
            if "timbers" in data and isinstance(data["timbers"], list) and len(data["timbers"]) > 0:
                timbers_in_file = data["timbers"]
            else:
                timbers_in_file = [data]

            for t_data in timbers_in_file:
                # Inherit marker info if needed
                if "markers_world_mm" in data and "markers_world_mm" not in t_data:
                    t_data["markers_world_mm"] = data["markers_world_mm"]

                res = parse_single_timber(t_data, f_path)
                if res["contour"] is not None:
                    list_contours.append(res["contour"])
                list_corners.append(res["corners"])
                list_ids.append(res["timber_id"])
                list_lengths.append(res["length_mm"])
                list_widths.append(res["width_mm"])
                list_thicknesses.append(res["thickness_mm"])
                list_areas.append(res["surface_area_cm2"])
                list_sides.append(res["side_lengths_mm"])
                list_colors_rgb.append(res["color_rgb"])
                list_colors_hex.append(res["color_hex"])
                list_defects.append(res["defects"])
                list_drawing_curves.append(res["drawing_curves"])
                list_drawing_centers.append(res["drawing_centers"])
                list_drawing_classes.append(res["drawing_classes"])
                list_drawing_sizes.append(res["drawing_sizes"])
                list_drawing_confidences.append(res["drawing_confidences"])
                list_files.append(f_path)
                list_raw.append(t_data)
    except Exception as e:
        pass

# 3. Table & Markers Geometry
table_boundary = None
markers_corners_list = []

if IN_RHINO and rg is not None:
    if first_data and "markers_world_mm" in first_data:
        for m_id, m_info in first_data["markers_world_mm"].items():
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

# 4. Handle optional Index Filter
sel_idx = globals().get("index", None)
if sel_idx is None:
    sel_idx = globals().get("selected_index", None)

if sel_idx is not None and len(list_contours) > 0:
    try:
        i = int(sel_idx)
        if i == -1:
            i = len(list_contours) - 1
        if 0 <= i < len(list_contours):
            list_contours = [list_contours[i]]
            list_corners = [list_corners[i]]
            list_ids = [list_ids[i]]
            list_lengths = [list_lengths[i]]
            list_widths = [list_widths[i]]
            list_thicknesses = [list_thicknesses[i]]
            list_areas = [list_areas[i]]
            list_sides = [list_sides[i]]
            list_colors_rgb = [list_colors_rgb[i]]
            list_colors_hex = [list_colors_hex[i]]
            list_defects = [list_defects[i]]
            list_drawing_curves = [list_drawing_curves[i]]
            list_drawing_centers = [list_drawing_centers[i]]
            list_drawing_classes = [list_drawing_classes[i]]
            list_drawing_sizes = [list_drawing_sizes[i]]
            list_drawing_confidences = [list_drawing_confidences[i]]
            list_files = [list_files[i]]
            list_raw = [list_raw[i]]
    except Exception:
        pass

# 5. Populate Grafted DataTrees for Grasshopper outputs (Branches {0}, {1}, {2}...)
timber_contour = timber_contours = make_grafted_tree(list_contours)
timber_corners = corners = make_grafted_tree(list_corners)
timber_id = timber_ids = make_grafted_tree(list_ids)
length_mm = lengths_mm = make_grafted_tree(list_lengths)
width_mm = widths_mm = make_grafted_tree(list_widths)
thickness_mm = thicknesses_mm = make_grafted_tree(list_thicknesses)
surface_area_cm2 = surface_areas_cm2 = make_grafted_tree(list_areas)
side_lengths_mm = make_grafted_tree(list_sides)
color_rgb = colors_rgb = make_grafted_tree(list_colors_rgb)
color_hex = colors_hex = make_grafted_tree(list_colors_hex)
defects = make_grafted_tree(list_defects)
drawing_curves = hand_drawn_shapes = make_grafted_tree(list_drawing_curves)
drawing_centers = make_grafted_tree(list_drawing_centers)
drawing_classes = make_grafted_tree(list_drawing_classes)
drawing_sizes = make_grafted_tree(list_drawing_sizes)
drawing_confidences = make_grafted_tree(list_drawing_confidences)
file_path = file_paths = make_grafted_tree(list_files)
raw_data = make_grafted_tree(list_raw)
markers_corners = make_grafted_tree(markers_corners_list)

# 6. Formatted Summary Info
if len(json_files) == 0:
    info = "No timber JSON files found in specified path or captures directory."
else:
    info_lines = [f"Loaded {len(json_files)} timber JSON file(s) (Grafted into DataTrees):"]
    for i, (t_id, l, w, th, fn, drw_cls) in enumerate(zip(list_ids, list_lengths, list_widths, list_thicknesses, list_files, list_drawing_classes)):
        fname = os.path.basename(fn)
        drw_str = f" | Drawings: {', '.join(drw_cls)}" if drw_cls else ""
        info_lines.append(f" {{{i}}} Timber #{t_id}: {l:.1f} x {w:.1f} mm, T={th:.1f}mm{drw_str} ({fname})")
    info = "\n".join(info_lines)

