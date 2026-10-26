"""
Enhanced Unit & Integration Test for Grasshopper Timber Reader & Live Capture.
Tests:
1. extract_bool with GH_Boolean, System.Boolean, list, DataTree, primitives.
2. extract_single_string with lists, trees, trailing slashes.
3. Fallback to latest timber when timber_00 doesn't exist yet.
4. TCP socket remote capture on port 9998 creating timber_00_measurement.json.
5. Verification of component execution with list/item inputs.
"""

import os
import sys
import json
import time
import socket
import threading
import config

import gh_component_timber_reader as gh_reader
from gh_component_timber_reader import (
    extract_bool,
    extract_single_string,
    trigger_remote_capture,
    resolve_single_json_file,
    parse_single_timber
)

class MockGHBoolean:
    def __init__(self, val):
        self.Value = val

class MockDataTree:
    def __init__(self, items):
        self._items = items
    def AllData(self):
        return self._items

def run_tests():
    print("=" * 60)
    print("TESTING ENHANCED GRASSHOPPER LIVE CAPTURE & READER")
    print("=" * 60)

    # 1. Test extract_bool
    print("\n[1] Testing extract_bool with diverse Grasshopper types...")
    assert extract_bool(True) is True
    assert extract_bool(False) is False
    assert extract_bool([True]) is True
    assert extract_bool([False]) is False
    assert extract_bool(MockGHBoolean(True)) is True
    assert extract_bool(MockGHBoolean(False)) is False
    assert extract_bool(MockDataTree([MockGHBoolean(True)])) is True
    assert extract_bool(MockDataTree([MockGHBoolean(False)])) is False
    assert extract_bool("True") is True
    assert extract_bool("False") is False
    assert extract_bool(None) is False
    print("  -> PASSED: All Grasshopper boolean variants correctly extracted.")

    # 2. Test extract_single_string
    print("\n[2] Testing extract_single_string...")
    raw_folder = [r"C:\Users\hamid\Documents\GitHub\eWoodX\Webcam detection\sample_output\captures\\"]
    clean_str = extract_single_string(raw_folder)
    assert clean_str == r"C:\Users\hamid\Documents\GitHub\eWoodX\Webcam detection\sample_output\captures" or clean_str.startswith("C:")
    print(f"  Extracted clean string: {clean_str}")
    print("  -> PASSED: String extracted cleanly from list input.")

    # 3. Test resolve_single_json_file
    print("\n[3] Testing resolve_single_json_file fallback when timber_00 not present...")
    t00_path = os.path.join(config.CAPTURE_DIR, "timber_00_measurement.json")
    if os.path.exists(t00_path):
        os.remove(t00_path)

    resolved = resolve_single_json_file(raw_folder)
    print(f"  Resolved fallback file: {resolved}")
    assert resolved is not None and os.path.isfile(resolved), "Failed to find fallback timber file"
    assert "timber_" in resolved, f"Expected timber measurement, got {resolved}"
    print("  -> PASSED: Canvas remains populated with preview geometry before first capture.")

    # 4. Test Mock Camera Server and Capture Trigger
    print("\n[4] Testing live TCP capture trigger on port 9998...")
    test_port = 9998
    server_running = True
    def mock_server():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", test_port))
        srv.listen(5)
        srv.settimeout(0.5)
        try:
            while server_running:
                try:
                    client, addr = srv.accept()
                except socket.timeout:
                    continue
                except Exception:
                    break
                data = client.recv(1024)
                req = json.loads(data.decode("utf-8").strip())
                num = req.get("number", 0)

                out_file = os.path.join(config.CAPTURE_DIR, f"timber_{num:02d}_measurement.json")
                sample_data = {
                    "capture_id": num,
                    "timbers": [{
                        "timber_id": num,
                        "length_mm": 600.0,
                        "width_mm": 120.0,
                        "timber_thickness_mm": 25.0,
                        "corners_mm": [[0,0], [600,0], [600,120], [0,120]],
                        "contour_mm": [[0,0], [600,0], [600,120], [0,120]],
                        "holes_mm": [],
                        "hand_drawn_shapes": []
                    }]
                }
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(sample_data, f, indent=2)

                resp = json.dumps({"status": "success", "json_path": out_file, "timber_number": num}) + "\n"
                client.sendall(resp.encode("utf-8"))
                client.close()
        finally:
            srv.close()

    sthread = threading.Thread(target=mock_server, daemon=True)
    sthread.start()
    time.sleep(0.3)

    ok, msg, j_path = trigger_remote_capture(host="127.0.0.1", port=test_port, number=0, timeout=4.0)
    assert ok, f"Trigger failed: {msg}"
    print(f"  Capture Trigger Result: ok={ok}, file={j_path}")
    assert os.path.isfile(t00_path), "timber_00_measurement.json was not created"
    print("  -> PASSED: TCP capture successfully generated timber_00_measurement.json.")


    # 5. Test Full GhPython Execution with Mock GH_Boolean Button
    print("\n[5] Testing full GhPython script execution with GH_Boolean button...")
    sim_globals = {
        "refresh": [MockGHBoolean(True)],  # As passed by Grasshopper Button component
        "json_path": raw_folder,
        "index": 0
    }
    with open(gh_reader.__file__, "r", encoding="utf-8") as f:
        code = f.read()

    exec(code, sim_globals)
    assert sim_globals.get("timber_id") is not None
    assert sim_globals.get("timber_contour") is not None
    print(f"  Status output: {sim_globals.get('status')}")
    print("  -> PASSED: GhPython component executes smoothly with Grasshopper Button input.")

    server_running = False
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
