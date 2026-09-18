"""
eWoodX - Interactive Direct 4-Corner Projector Calibration

Directly aligns the 4 projector corners with the 4 physical ArUco markers on the table:
  - Marker 1: (0, 0) mm         (Bottom-Left / Origin)
  - Marker 0: (1780, 0) mm      (Bottom-Right)
  - Marker 2: (0, 1040) mm      (Top-Left)
  - Marker 3: (1780, 1040) mm   (Top-Right)

Controls:
  - Click & Drag: Drag any of the 4 crosshairs directly onto the corresponding marker.
  - Keys 1, 0, 2, 3 (or TAB): Select active crosshair.
  - Arrow Keys / W, A, S, D: Nudge the active crosshair by 1 px (or fine step).
  - + / -: Change nudge step (0.2 px, 0.5 px, 1 px, 5 px).
  - SPACE / ENTER: Save calibration to calibration_data/projector_homography.npz.
  - ESC / Q: Exit without saving.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import config

# ============================================================
# PROJECTOR SETTINGS
# ============================================================

# Projector display settings are now loaded from config.py
PROJECTOR_WIDTH = config.PROJECTOR_WIDTH
PROJECTOR_HEIGHT = config.PROJECTOR_HEIGHT
PROJECTOR_SCREEN_ORIGIN_X = config.PROJECTOR_SCREEN_ORIGIN_X
PROJECTOR_SCREEN_ORIGIN_Y = config.PROJECTOR_SCREEN_ORIGIN_Y

WINDOW_NAME = "eWoodX - 4-Corner Projector Calibration"

OUTPUT_NPZ = os.path.join(config.CALIB_DIR, "projector_homography.npz")
OUTPUT_TXT = os.path.join(config.CALIB_DIR, "projector_homography.txt")

# Physical marker coordinates (mm)
MARKERS_MM = {
    1: np.array([0.0, 0.0], dtype=np.float32),                                      # Bottom-Left
    0: np.array([config.TABLE_WIDTH_MM, 0.0], dtype=np.float32),                   # Bottom-Right
    2: np.array([0.0, config.TABLE_HEIGHT_MM], dtype=np.float32),                   # Top-Left
    3: np.array([config.TABLE_WIDTH_MM, config.TABLE_HEIGHT_MM], dtype=np.float32) # Top-Right
}

# Initial projector pixel estimates for the 4 corners
# (Based on standard 1280x800 layout with margins)
margin_ratio = 0.05
margin_x = PROJECTOR_WIDTH * margin_ratio
margin_y = PROJECTOR_HEIGHT * margin_ratio
initial_positions = {
    1: [margin_x, PROJECTOR_HEIGHT - margin_y],   # Bottom-Left (Marker 1)
    0: [PROJECTOR_WIDTH - margin_x, PROJECTOR_HEIGHT - margin_y],  # Bottom-Right (Marker 0)
    2: [margin_x, margin_y],   # Top-Left (Marker 2)
    3: [PROJECTOR_WIDTH - margin_x, margin_y],  # Top-Right (Marker 3)
}

# If an existing homography exists, use it to seed initial crosshair positions
if os.path.exists(OUTPUT_NPZ):
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle=True)
        H_exist = data["H_world_to_projector"]
        for m_id, mm_pt in MARKERS_MM.items():
            proj_pt = cv2.perspectiveTransform(mm_pt.reshape(1, 1, 2), H_exist).reshape(2)
            initial_positions[m_id] = [float(proj_pt[0]), float(proj_pt[1])]
    except Exception:
        pass

# Current active points
marker_px = {m_id: np.array(pos, dtype=np.float32) for m_id, pos in initial_positions.items()}

active_id = 1
step_px = 1.0
dragging_id = None

# Colors (BGR)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GRAY = (150, 150, 150)


# ============================================================
# MOUSE INTERACTION
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global marker_px, active_id, dragging_id

    if event == cv2.EVENT_LBUTTONDOWN:
        # Find closest marker
        min_dist = 50.0  # click radius
        closest = None
        for m_id, pt in marker_px.items():
            dist = np.hypot(pt[0] - x, pt[1] - y)
            if dist < min_dist:
                min_dist = dist
                closest = m_id
        if closest is not None:
            active_id = closest
            dragging_id = closest
            marker_px[closest] = np.array([float(x), float(y)], dtype=np.float32)

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_id is not None:
            marker_px[dragging_id] = np.array([float(x), float(y)], dtype=np.float32)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_id = None


# ============================================================
# RENDER FUNCTION
# ============================================================

def draw_crosshair(canvas, pt, label, is_active=False, size=24):
    x, y = int(round(pt[0])), int(round(pt[1]))
    color = YELLOW if is_active else WHITE
    thick = 2 if is_active else 1

    # Outer circle
    cv2.circle(canvas, (x, y), 12, color, thick, cv2.LINE_AA)
    # Center dot
    cv2.circle(canvas, (x, y), 2, RED if is_active else color, -1, cv2.LINE_AA)

    # Cross lines with center gap
    gap = 4
    cv2.line(canvas, (x - size, y), (x - gap, y), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x + gap, y), (x + size, y), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x, y - size), (x, y - gap), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x, y + gap), (x, y + size), color, thick, cv2.LINE_AA)

    # Label
    cv2.putText(
        canvas, label, (x + 16, y - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA
    )
    coord_str = f"({x}, {y})"
    cv2.putText(
        canvas, coord_str, (x + 16, y + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA
    )


def render():
    canvas = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # 1. Connect the 4 markers with guideline rectangle (BL -> BR -> TR -> TL -> BL)
    ordered_ids = [1, 0, 3, 2]
    quad_pts = np.array([marker_px[i] for i in ordered_ids], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [quad_pts], True, (40, 40, 40), 1, cv2.LINE_AA)

    # Center diagonals
    p1 = tuple(np.round(marker_px[1]).astype(int))
    p3 = tuple(np.round(marker_px[3]).astype(int))
    p0 = tuple(np.round(marker_px[0]).astype(int))
    p2 = tuple(np.round(marker_px[2]).astype(int))
    cv2.line(canvas, p1, p3, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.line(canvas, p0, p2, (25, 25, 25), 1, cv2.LINE_AA)

    # 2. Draw the 4 crosshairs
    labels = {
        1: "Marker 1 (0,0) Origin",
        0: "Marker 0 (1780,0)",
        2: "Marker 2 (0,1040)",
        3: "Marker 3 (1780,1040)"
    }

    for m_id in [1, 0, 2, 3]:
        draw_crosshair(canvas, marker_px[m_id], labels[m_id], is_active=(m_id == active_id))

    # 3. Top title & status HUD
    title = f"eWoodX 4-Corner Alignment  |  Active: Marker {active_id}  |  Step: {step_px:.1f} px"
    cv2.putText(canvas, title, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)

    instructions = (
        "Controls: [1/0/2/3/TAB] Select Marker  |  [W/S/A/D or Arrows] Nudge  |  "
        "[+/-] Step  |  [SPACE/ENTER] Save  |  [ESC] Exit"
    )
    cv2.putText(canvas, instructions, (25, PROJECTOR_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1, cv2.LINE_AA)

    return canvas


# ============================================================
# SAVE HOMOGRAPHY
# ============================================================

def save_calibration():
    # World points (mm)
    world_pts = np.array([
        MARKERS_MM[1],
        MARKERS_MM[0],
        MARKERS_MM[2],
        MARKERS_MM[3]
    ], dtype=np.float32)

    # Projector pixel points
    proj_pts = np.array([
        marker_px[1],
        marker_px[0],
        marker_px[2],
        marker_px[3]
    ], dtype=np.float32)

    # Compute World -> Projector Homography
    H_world_to_projector, _ = cv2.findHomography(world_pts, proj_pts, method=0)

    if H_world_to_projector is None:
        print("ERROR: Failed to compute homography matrix.")
        return False

    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)

    # Save NPZ
    np.savez(
        OUTPUT_NPZ,
        H_world_to_projector=H_world_to_projector,
        world_points_mm=world_pts,
        projector_points_px=proj_pts,
        projector_resolution=np.array([PROJECTOR_WIDTH, PROJECTOR_HEIGHT])
    )

    # Save TXT report
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("eWoodX DIRECT 4-CORNER PROJECTOR CALIBRATION\n")
        f.write("=" * 55 + "\n\n")
        f.write("World -> Projector Homography:\n")
        f.write(np.array2string(H_world_to_projector, precision=10) + "\n\n")
        f.write("Aligned Points:\n")
        for m_id in [1, 0, 2, 3]:
            w = MARKERS_MM[m_id]
            p = marker_px[m_id]
            f.write(f"Marker {m_id}: World=({w[0]:.1f}, {w[1]:.1f}) mm  ->  Projector=({p[0]:.2f}, {p[1]:.2f}) px\n")

    print("\n" + "=" * 60)
    print("CALIBRATION SAVED SUCCESSFULLY!")
    print(f"NPZ: {OUTPUT_NPZ}")
    print(f"TXT: {OUTPUT_TXT}")
    print("\nHomography Matrix:")
    print(H_world_to_projector)
    print("=" * 60 + "\n")
    return True


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global active_id, step_px, marker_px

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, PROJECTOR_SCREEN_ORIGIN_X, PROJECTOR_SCREEN_ORIGIN_Y)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print()
    print("============================================================")
    print("eWoodX Direct 4-Corner Projector Calibration")
    print("============================================================")
    print("Look at your table surface and align the 4 crosshairs")
    print("directly onto the physical ArUco markers:")
    print("  Marker 1: Bottom-Left  (0, 0) mm")
    print("  Marker 0: Bottom-Right (1780, 0) mm")
    print("  Marker 2: Top-Left     (0, 1040) mm")
    print("  Marker 3: Top-Right    (1780, 1040) mm")
    print()
    print("Press SPACE or ENTER when all 4 are centered.")
    print("============================================================")
    print()

    id_list = [1, 0, 3, 2]

    while True:
        canvas = render()
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKeyEx(20)

        if key in (27, ord('q'), ord('Q')):
            print("Calibration cancelled.")
            break

        # Save on SPACE or ENTER
        elif key in (32, 10, 13):
            if save_calibration():
                break

        # Select Marker
        elif key == ord('1'):
            active_id = 1
            print(f"Selected Marker 1 (Bottom-Left)")
        elif key == ord('0'):
            active_id = 0
            print(f"Selected Marker 0 (Bottom-Right)")
        elif key == ord('2'):
            active_id = 2
            print(f"Selected Marker 2 (Top-Left)")
        elif key == ord('3'):
            active_id = 3
            print(f"Selected Marker 3 (Top-Right)")
        elif key == 9:  # TAB
            curr_idx = id_list.index(active_id) if active_id in id_list else 0
            active_id = id_list[(curr_idx + 1) % len(id_list)]
            print(f"Selected Marker {active_id}")

        # Nudge Up (move -Y in projector screen)
        elif key in (2490368, ord('w'), ord('W')):
            marker_px[active_id][1] -= step_px
        # Nudge Down (move +Y in projector screen)
        elif key in (2621440, ord('s'), ord('S')):
            marker_px[active_id][1] += step_px
        # Nudge Left (move -X in projector screen)
        elif key in (2424832, ord('a'), ord('A')):
            marker_px[active_id][0] -= step_px
        # Nudge Right (move +X in projector screen)
        elif key in (2555904, ord('d'), ord('D')):
            marker_px[active_id][0] += step_px

        # Change Step Size
        elif key in (ord('+'), ord('='), ord(']')):
            steps = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
            curr_idx = steps.index(step_px) if step_px in steps else 3
            step_px = steps[min(len(steps) - 1, curr_idx + 1)]
            print(f"Nudge step: {step_px} px")
        elif key in (ord('-'), ord('_'), ord('[')):
            steps = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
            curr_idx = steps.index(step_px) if step_px in steps else 3
            step_px = steps[max(0, curr_idx - 1)]
            print(f"Nudge step: {step_px} px")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
