"""
Step 4: Export timber measurements to a Rhino .3dm model file.

Features:
    - Interactive file dialog to select one or multiple timber JSON files (or pass via CLI).
    - Exports the exact timber contour (organic boundary outline in real-world mm), NOT the bounding box.
    - Preserves timber color (dedicated Rhino layer and object attributes per timber with RGB color).
    - Annotates with TextDot containing Dimensions, Surface Area (cm² / mm²), Thickness, and Color.
    - Saves directly to sample_output/boards.3dm.
"""

import os
import sys
import json
import rhino3dm
import config


def select_json_files():
    """Open a graphical file picker dialog to select JSON file(s)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        init_dir = config.CAPTURE_DIR if os.path.exists(config.CAPTURE_DIR) else config.OUTPUT_DIR

        file_paths = filedialog.askopenfilenames(
            title="Select Timber Measurement JSON File(s)",
            initialdir=init_dir,
            filetypes=[
                ("JSON files", "*_measurement.json *.json"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return list(file_paths)
    except Exception as e:
        print(f"Warning: Could not open file dialog ({e}).")
        return []


def add_timber_to_model(model, data, index=0, z_offset=0.0):
    """
    Add timber contour, layer with color, and text annotation to Rhino model.
    """
    timber_id = data.get("timber_id", index + 1)
    
    # 1. Timber Contour (use exact contour_mm if available, fallback to corners_mm)
    if "contour_mm" in data and len(data["contour_mm"]) >= 3:
        pts_coords = data["contour_mm"]
    elif "corners_mm" in data and len(data["corners_mm"]) >= 3:
        pts_coords = data["corners_mm"]
    else:
        print(f"  Warning: No contour or corner data found for timber {timber_id}.")
        return False

    # Create closed 3D polyline points
    pts = [rhino3dm.Point3d(float(x), float(y), float(z_offset)) for x, y in pts_coords]
    pts.append(pts[0])  # Close the loop

    polyline = rhino3dm.Polyline(pts)
    curve = polyline.ToPolylineCurve()

    # 2. Extract Color
    if "color_rgb" in data:
        r = int(data["color_rgb"].get("R", 190))
        g = int(data["color_rgb"].get("G", 150))
        b = int(data["color_rgb"].get("B", 110))
    elif "color_lab" in data:
        lab = data["color_lab"]
        r = int(np.clip(lab.get("L", 180), 0, 255))
        g = int(np.clip(lab.get("a", 150), 0, 255))
        b = int(np.clip(lab.get("b", 110), 0, 255))
    else:
        r, g, b = 200, 160, 120

    # 3. Create a dedicated Layer with timber color
    layer = rhino3dm.Layer()
    layer.Name = f"Timber_{timber_id:02d}" if isinstance(timber_id, int) else f"Timber_{timber_id}"
    layer.Color = (r, g, b, 255)
    layer_index = model.Layers.Add(layer)

    # 4. Object Attributes
    attr = rhino3dm.ObjectAttributes()
    attr.LayerIndex = layer_index
    attr.ColorSource = rhino3dm.ObjectColorSource.ColorFromLayer
    attr.Name = f"Timber_{timber_id}_Contour"

    # Add curve to model
    model.Objects.AddCurve(curve, attr)

    # 5. Centroid for Annotation Placement
    cx = sum(p.X for p in pts[:-1]) / len(pts[:-1])
    cy = sum(p.Y for p in pts[:-1]) / len(pts[:-1])

    # 6. Surface Area & Dimensions
    length_mm = data.get("length_mm", 0.0)
    width_mm = data.get("width_mm", 0.0)
    thickness_mm = data.get("timber_thickness_mm", 0.0)
    
    if "surface_area_cm2" in data:
        area_cm2 = data["surface_area_cm2"]
    elif "surface_area_mm2" in data:
        area_cm2 = data["surface_area_mm2"] / 100.0
    else:
        area_cm2 = (length_mm * width_mm) / 100.0

    # Text Dot Annotation
    label_text = (
        f"Timber {timber_id:02d}\n"
        f"Dim: {length_mm:.1f} x {width_mm:.1f} mm\n"
        f"Area: {area_cm2:.1f} cm²\n"
        f"Thick: {thickness_mm:.1f} mm\n"
        f"RGB({r}, {g}, {b})"
    )
    model.Objects.AddTextDot(label_text, rhino3dm.Point3d(cx, cy, float(z_offset)), attr)

    # 7. Defect Markers (if present)
    for defect in data.get("defects", []):
        dpt = rhino3dm.Point3d(float(defect.get("x_mm", 0)), float(defect.get("y_mm", 0)), float(z_offset))
        model.Objects.AddPoint(dpt, attr)

    return True


def add_workspace_to_model(model, data):
    """Add workspace boundary and ArUco markers to the Rhino model."""
    layer = rhino3dm.Layer()
    layer.Name = "Workspace_Table"
    layer.Color = (100, 100, 100, 255)
    layer_index = model.Layers.Add(layer)

    attr = rhino3dm.ObjectAttributes()
    attr.LayerIndex = layer_index
    attr.ColorSource = rhino3dm.ObjectColorSource.ColorFromLayer
    attr.Name = "Table_Boundary"

    # Outer table boundary rectangle from config
    m1 = config.MARKER_WORLD_POSITIONS_MM.get(1, (0.0, 0.0))
    m0 = config.MARKER_WORLD_POSITIONS_MM.get(0, (776.0, 0.0))
    m3 = config.MARKER_WORLD_POSITIONS_MM.get(3, (776.0, 440.0))
    m2 = config.MARKER_WORLD_POSITIONS_MM.get(2, (0.0, 440.0))

    pts = [
        rhino3dm.Point3d(float(m1[0]), float(m1[1]), 0.0),
        rhino3dm.Point3d(float(m0[0]), float(m0[1]), 0.0),
        rhino3dm.Point3d(float(m3[0]), float(m3[1]), 0.0),
        rhino3dm.Point3d(float(m2[0]), float(m2[1]), 0.0),
        rhino3dm.Point3d(float(m1[0]), float(m1[1]), 0.0),
    ]
    model.Objects.AddCurve(rhino3dm.Polyline(pts).ToPolylineCurve(), attr)

    # ArUco marker squares and label text dots
    markers = data.get("markers_world_mm", {})
    for m_id, m_info in markers.items():
        if "corners_mm" in m_info and len(m_info["corners_mm"]) == 4:
            c_pts = [rhino3dm.Point3d(float(x), float(y), 0.0) for x, y in m_info["corners_mm"]]
            c_pts.append(c_pts[0])
            m_attr = rhino3dm.ObjectAttributes()
            m_attr.LayerIndex = layer_index
            m_attr.Name = f"ArUco_Marker_{m_id}"
            model.Objects.AddCurve(rhino3dm.Polyline(c_pts).ToPolylineCurve(), m_attr)
        if "center_mm" in m_info:
            cx, cy = m_info["center_mm"]
            model.Objects.AddTextDot(f"ArUco {m_id}", rhino3dm.Point3d(float(cx), float(cy), 0.0), attr)


def main():
    # If file paths are provided on command line, use them; otherwise open file picker
    if len(sys.argv) > 1:
        json_files = sys.argv[1:]
    else:
        print("Opening file selection dialog...")
        json_files = select_json_files()

    if not json_files:
        print("No JSON files selected. Exiting.")
        return

    print("\n" + "=" * 60)
    print("RHINO 3DM EXPORT - TIMBER CONTOURS")
    print("=" * 60)
    print(f"Selected {len(json_files)} file(s) for export:\n")

    model = rhino3dm.File3dm()
    model.Settings.ModelUnitSystem = rhino3dm.UnitSystem.Millimeters

    success_count = 0
    workspace_added = False
    for i, json_path in enumerate(json_files):
        if not os.path.exists(json_path):
            print(f"  ERROR: File not found: {json_path}")
            continue

        with open(json_path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"  ERROR reading {json_path}: {e}")
                continue

        if not workspace_added:
            add_workspace_to_model(model, data)
            workspace_added = True

        ok = add_timber_to_model(model, data, index=i)
        if ok:
            t_id = data.get("timber_id", i + 1)
            l = data.get("length_mm", 0.0)
            w = data.get("width_mm", 0.0)
            area = data.get("surface_area_cm2", data.get("surface_area_mm2", 0.0) / 100.0)
            pts_cnt = len(data.get("contour_mm", data.get("corners_mm", [])))
            print(f"  [+] Timber {t_id:02d}: {l:.1f} x {w:.1f} mm | Area: {area:.1f} cm² | Contour: {pts_cnt} points ({os.path.basename(json_path)})")
            success_count += 1

    if success_count == 0:
        print("\nNo valid timber data exported.")
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, "boards.3dm")
    model.Write(out_path, 7)

    print("\n" + "-" * 60)
    print(f"Successfully exported {success_count} timber contour(s) to:")
    print(f"  {out_path}")
    print("-" * 60)
    print("Open this file directly in Rhino or Grasshopper to view contours, colors, and surface areas.\n")


if __name__ == "__main__":
    main()
