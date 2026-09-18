"""
eWoodX - Test Grasshopper Live Sender Simulation

Simulates Grasshopper sending live timber contours, cutting lines (Red),
milling pockets (Cyan), drill holes (Green), and text annotations (Amber)
to projector_live_viewer.py over TCP socket (port 9999).
"""

import socket
import json
import time
import math
import glob
import os
import sys
from pathlib import Path
import config


def get_latest_json():
    pattern = os.path.join(config.CAPTURE_DIR, "timber_*_measurement.json")
    json_files = glob.glob(pattern)
    if json_files:
        json_files.sort(key=os.path.getmtime, reverse=True)
        return Path(json_files[0])
    return Path(os.path.join(config.CAPTURE_DIR, "timber_18_measurement.json"))


def send_packet(host, port, payload):
    data = (json.dumps(payload) + "\n").encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        s.connect((host, port))
        s.sendall(data)


def main():
    json_path = get_latest_json()
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        timber_data = json.load(f)

    contour = timber_data.get("contour_mm", [])
    corners = timber_data.get("corners_mm", [])
    thickness = float(timber_data.get("timber_thickness_mm", 17.0))
    timber_id = timber_data.get("timber_id", 1)

    # Compute bounding box center and extent
    all_pts = np_contour = contour if contour else corners
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    length = max_x - min_x

    print("=" * 60)
    print("eWoodX - Simulating Grasshopper Real-Time Stream")
    print(f"Target:       localhost:9999")
    print(f"Timber:       #{timber_id} ({length:.1f} mm long)")
    print(f"Thickness:    {thickness:.1f} mm")
    print("Streaming animated cut lines, milling pockets, drill holes...")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    t = 0.0
    while True:
        try:
            # Animate cut line moving across timber
            cut_x = min_x + (0.3 + 0.4 * (0.5 + 0.5 * math.sin(t * 1.5))) * length
            cut_line1 = [[cut_x, min_y - 20], [cut_x, max_y + 20]]
            cut_line2 = [[max_x - 40, min_y - 10], [max_x - 10, max_y + 10]]

            # Milling rectangle / pocket
            mill_x = min_x + 50
            mill_rect = [
                [mill_x, min_y + 10],
                [mill_x + 80, min_y + 10],
                [mill_x + 80, max_y - 10],
                [mill_x, max_y - 10],
                [mill_x, min_y + 10]
            ]
            # Milling zig-zag hatch
            mill_hatch = []
            for hx in range(int(mill_x + 10), int(mill_x + 80), 10):
                mill_hatch.append([[hx, min_y + 12], [hx, max_y - 12]])

            # Drill points
            drill_points = [
                [min_x + 25, cy],
                [max_x - 25, cy],
                [cx, min_y + 15],
                [cx, max_y - 15]
            ]

            # Labels
            labels = [
                {"text": f"TIMBER #{timber_id} - LIVE GH", "pos": [min_x, max_y + 35]},
                {"text": f"CUT POS: {cut_x:.1f} mm", "pos": [cut_x + 5, min_y - 25]},
                {"text": "POCKET 5mm", "pos": [mill_x + 5, min_y + 25]}
            ]

            payload = {
                "timestamp": time.time(),
                "thickness_mm": thickness,
                "timber_contour": contour,
                "timber_corners": corners,
                "timber_color": [255, 255, 255],  # White
                "cut_geo": [cut_line1, cut_line2],
                "cut_color": [255, 30, 30],         # Vibrant Red
                "mill_geo": [mill_rect] + mill_hatch,
                "mill_color": [0, 220, 255],        # Cyan
                "points_geo": drill_points,
                "point_color": [0, 255, 100],       # Green
                "labels": labels,
                "label_color": [255, 180, 0]        # Amber
            }

            send_packet("127.0.0.1", 9999, payload)
            t += 0.05
            time.sleep(0.05)

        except ConnectionRefusedError:
            print("[Waiting] projector_live_viewer.py is not running. Retrying in 1s...")
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
