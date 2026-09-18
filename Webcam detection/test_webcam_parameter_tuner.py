"""
Interactive Webcam Control & Parameter Tuner (Focus, Zoom, Light, Exposure).
Use this tool to find the optimal camera settings for your timber rig,
then easily copy them or have them saved to config.py.

Controls:
    Focus:
      1 - 6      -> Quick focus presets (300, 350, 400, 450, 500, 600)
      + / =      -> Increase focus (+10)
      - / _      -> Decrease focus (-10)
      A          -> Set Focus to AUTO
      M          -> Set Focus to MANUAL
    
    Zoom (Real-Time Sensor & Digital Zoom):
      [ / ] or z / Z -> Zoom out / in (step 0.1x)
      0          -> Reset Zoom to 1.0x (full 120° wide angle)
      7, 8, 9    -> Zoom presets (1.5x, 2.0x, 3.0x)
    
    Exposure & Lighting:
      E          -> Toggle Exposure AUTO / MANUAL
      ( / )      -> Adjust manual exposure down / up
      B / Shift+B-> Brightness down / up (1 - 64)
      C / Shift+C-> Contrast down / up (1 - 64)
    
    Config Actions:
      P          -> Print config.py settings snippet to terminal
      S          -> Save current values directly to config.py
      Q / ESC    -> Quit
"""

import os
import sys
import time
import cv2
import config
import camera_utils

try:
    import duvc_ctl as duvc
except ImportError:
    duvc = None
    print("Warning: duvc_ctl not installed. Running with OpenCV DirectShow properties only.")


CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 1)


def print_current_config_snippet(settings):
    print("\n" + "=" * 60)
    print("COPY-PASTE CONFIGURATION SNIPPET FOR config.py")
    print("=" * 60)
    print(f'CAMERA_INDEX = {CAMERA_INDEX}')
    print(f'CAMERA_FOCUS_MODE = "{settings.get("focus_mode", "manual")}"')
    print(f'CAMERA_FOCUS_VALUE = {settings.get("focus_value", 450)}')
    print(f'CAMERA_ZOOM = {settings.get("zoom", 1.0):.2f}')
    print(f'CAMERA_EXPOSURE_MODE = "{settings.get("exposure_mode", "auto")}"')
    print(f'CAMERA_EXPOSURE_VALUE = {settings.get("exposure_value", -5)}')
    if settings.get("brightness") is not None:
        print(f'CAMERA_BRIGHTNESS = {settings.get("brightness")}')
    if settings.get("contrast") is not None:
        print(f'CAMERA_CONTRAST = {settings.get("contrast")}')
    print("=" * 60 + "\n")


def update_config_file(settings):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    if not os.path.exists(config_path):
        print(f"Error: Could not find {config_path}")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith("CAMERA_FOCUS_MODE ="):
                new_lines.append(f'CAMERA_FOCUS_MODE = "{settings.get("focus_mode", "manual")}"\n')
            elif line.startswith("CAMERA_FOCUS_VALUE ="):
                new_lines.append(f'CAMERA_FOCUS_VALUE = {settings.get("focus_value", 450)}\n')
            elif line.startswith("CAMERA_ZOOM =") or line.startswith("CAMERA_ZOOM_VALUE ="):
                new_lines.append(f'CAMERA_ZOOM = {settings.get("zoom", 1.0):.2f}\n')
            elif line.startswith("CAMERA_EXPOSURE_MODE ="):
                new_lines.append(f'CAMERA_EXPOSURE_MODE = "{settings.get("exposure_mode", "auto")}"\n')
            elif line.startswith("CAMERA_EXPOSURE_VALUE ="):
                new_lines.append(f'CAMERA_EXPOSURE_VALUE = {settings.get("exposure_value", -5)}\n')
            elif line.startswith("CAMERA_BRIGHTNESS =") and settings.get("brightness") is not None:
                new_lines.append(f'CAMERA_BRIGHTNESS = {settings.get("brightness")}\n')
            elif line.startswith("CAMERA_CONTRAST =") and settings.get("contrast") is not None:
                new_lines.append(f'CAMERA_CONTRAST = {settings.get("contrast")}\n')
            else:
                new_lines.append(line)

        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"\nSuccessfully saved tuned camera parameters to {config_path}!")
    except Exception as e:
        print(f"Failed to update config.py: {e}")


def main():
    print("=" * 60)
    print("WEBCAM PARAMETER TUNER (Focus, Zoom, Light/Exposure)")
    print("=" * 60)
    print(f"Connecting to camera index {CAMERA_INDEX}...")

    controller = None
    prop_ranges = {}

    if duvc is not None:
        try:
            controller = duvc.CameraController(device_index=CAMERA_INDEX)
            dev_name = getattr(controller, 'device_name', 'Webcam')
            print(f"Connected to {dev_name} via duvc_ctl.")

            # Read hardware ranges
            for p in ['focus', 'exposure', 'brightness', 'contrast', 'saturation', 'sharpness', 'gain', 'white_balance']:
                try:
                    r = controller.get_property_range(p)
                    if isinstance(r, dict):
                        prop_ranges[p] = r
                except Exception:
                    pass
        except Exception as e:
            print(f"duvc_ctl connection note: {e}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened() and CAMERA_INDEX != 0:
        print(f"Failed to open index {CAMERA_INDEX}, trying index 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        if controller:
            controller.close()
        raise RuntimeError("Could not open camera with OpenCV.")

    w = getattr(config, "IMAGE_WIDTH", 1920)
    h = getattr(config, "IMAGE_HEIGHT", 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, min(w, 1920))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, min(h, 1080))

    # Limits
    focus_min = prop_ranges.get("focus", {}).get("min", 0)
    focus_max = prop_ranges.get("focus", {}).get("max", 1023)
    exp_min = prop_ranges.get("exposure", {}).get("min", -13)
    exp_max = prop_ranges.get("exposure", {}).get("max", -1)
    bright_min = prop_ranges.get("brightness", {}).get("min", 1)
    bright_max = prop_ranges.get("brightness", {}).get("max", 64)
    contrast_min = prop_ranges.get("contrast", {}).get("min", 1)
    contrast_max = prop_ranges.get("contrast", {}).get("max", 64)

    # Initial state
    focus_mode = getattr(config, "CAMERA_FOCUS_MODE", "manual")
    focus_val = getattr(config, "CAMERA_FOCUS_VALUE", 450)
    zoom_val = float(getattr(config, "CAMERA_ZOOM", getattr(config, "CAMERA_ZOOM_VALUE", 1.0)))
    if zoom_val < 1.0:
        zoom_val = 1.0
    exp_mode = getattr(config, "CAMERA_EXPOSURE_MODE", "auto")
    exp_val = getattr(config, "CAMERA_EXPOSURE_VALUE", -5)
    brightness_val = getattr(config, "CAMERA_BRIGHTNESS", 32)
    contrast_val = getattr(config, "CAMERA_CONTRAST", 32)

    # Clamp initial values
    focus_val = max(focus_min, min(focus_max, focus_val))
    exp_val = max(exp_min, min(exp_max, exp_val))
    brightness_val = max(bright_min, min(bright_max, brightness_val))
    contrast_val = max(contrast_min, min(contrast_max, contrast_val))

    # Apply initial values to hardware
    if controller is not None:
        try:
            controller.focus_mode = focus_mode
            if focus_mode == "manual":
                controller.focus = focus_val
            if exp_mode == "auto":
                if hasattr(controller, "exposure_mode"):
                    controller.exposure_mode = "auto"
                elif hasattr(controller, "_set_property_auto"):
                    controller._set_property_auto("exposure")
            else:
                controller.exposure = exp_val
            if hasattr(controller, "brightness"):
                controller.brightness = brightness_val
            if hasattr(controller, "contrast"):
                controller.contrast = contrast_val
        except Exception as e:
            print(f"Initial setup note: {e}")

    window_name = "Webcam Control & Parameter Tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\nTuning Controls Active. View image window.")
    print("  Focus range: 0 - 1023 (manual)")
    print("  Zoom range:  1.0x - 4.0x (real-time)")
    print("Keys: [1-6]=Focus presets, [+/-]=Focus step, [[/]] or [z/Z]=Zoom in/out, [0,7,8,9]=Zoom presets, [E]=Exp, [P]=Print, [S]=Save, [Q]=Quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame.")
                break

            # Apply real-time digital/sensor zoom
            zoomed_frame = camera_utils.apply_digital_zoom(frame, zoom_val)

            # Read live stats from hardware
            curr_focus = getattr(controller, "focus", focus_val) if controller else focus_val
            curr_exp = getattr(controller, "exposure", exp_val) if controller else exp_val
            curr_bright = getattr(controller, "brightness", brightness_val) if controller else brightness_val
            curr_contrast = getattr(controller, "contrast", contrast_val) if controller else contrast_val

            # Draw HUD
            hud_lines = [
                f"Focus: {curr_focus} [{focus_min}-{focus_max}] (Mode: {focus_mode.upper()}) [Keys: 1-6, +/-]",
                f"Zoom: {zoom_val:.2f}x [1.0x - 4.0x] [Keys: [ / ] or z/Z, presets: 0, 7, 8, 9]",
                f"Exposure: {curr_exp} [{exp_min}..{exp_max}] (Mode: {exp_mode.upper()}) [Keys: E, ( / )]",
                f"Brightness: {curr_bright} [{bright_min}-{bright_max}] [Keys: B/Shift+B] | Contrast: {curr_contrast} [{contrast_min}-{contrast_max}] [Keys: C/Shift+C]",
                f"[P] Print Config | [S] Save to config.py | [Q] Quit",
            ]

            # Semi-transparent overlay box
            overlay = zoomed_frame.copy()
            cv2.rectangle(overlay, (20, 15), (780, 175), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, zoomed_frame, 0.35, 0, zoomed_frame)

            for i, line in enumerate(hud_lines):
                cv2.putText(
                    zoomed_frame,
                    line,
                    (30, 45 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255) if i < 2 else ((100, 255, 100) if i == 2 else (255, 255, 255)),
                    2
                )

            cv2.imshow(window_name, zoomed_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break

            # Focus presets
            elif key == ord('1'):
                focus_mode = "manual"
                focus_val = 300
                if controller:
                    controller.focus_mode = "manual"
                    controller.focus = focus_val
                print(f"Focus set -> {focus_val}")
            elif key == ord('2'):
                focus_mode = "manual"
                focus_val = 350
                if controller:
                    controller.focus_mode = "manual"
                    controller.focus = focus_val
                print(f"Focus set -> {focus_val}")
            elif key == ord('3'):
                focus_mode = "manual"
                focus_val = 400
                if controller:
                    controller.focus_mode = "manual"
                    controller.focus = focus_val
                print(f"Focus set -> {focus_val}")
            elif key == ord('4'):
                focus_mode = "manual"
                focus_val = 450
                if controller:
                    controller.focus_mode = "manual"
                    controller.focus = focus_val
                print(f"Focus set -> {focus_val}")
            elif key == ord('5'):
                focus_mode = "manual"
                focus_val = 500
                if controller:
                    controller.focus_mode = "manual"
                    controller.focus = focus_val
                print(f"Focus set -> {focus_val}")
            elif key == ord('6'):
                focus_mode = "manual"
                focus_val = 600
                if controller:
                    controller.focus_mode = "manual"
                    controller.focus = focus_val
                print(f"Focus set -> {focus_val}")

            # Step Focus
            elif key in (ord('+'), ord('=')):
                focus_mode = "manual"
                focus_val = min(focus_max, focus_val + 10)
                if controller:
                    controller.focus_mode = "manual"
                    controller.focus = focus_val
                print(f"Focus -> {focus_val}")
            elif key in (ord('-'), ord('_')):
                focus_mode = "manual"
                focus_val = max(focus_min, focus_val - 10)
                if controller:
                    controller.focus_mode = "manual"
                    controller.focus = focus_val
                print(f"Focus -> {focus_val}")

            # Focus Mode
            elif key in (ord('a'), ord('A')):
                focus_mode = "auto"
                if controller:
                    controller.focus_mode = "auto"
                print("Focus Mode -> AUTO")
            elif key in (ord('m'), ord('M')):
                focus_mode = "manual"
                if controller:
                    controller.focus_mode = "manual"
                    controller.focus = focus_val
                print("Focus Mode -> MANUAL")

            # Zoom controls (keys [ / ] and z / Z)
            elif key in (ord('['), ord('z')):
                zoom_val = max(1.0, round(zoom_val - 0.1, 2))
                print(f"Zoom -> {zoom_val:.2f}x")

            elif key in (ord(']'), ord('Z')):
                zoom_val = min(4.0, round(zoom_val + 0.1, 2))
                print(f"Zoom -> {zoom_val:.2f}x")

            # Zoom presets
            elif key == ord('0'):
                zoom_val = 1.0
                print("Zoom -> 1.0x (Full Wide View)")
            elif key == ord('7'):
                zoom_val = 1.5
                print("Zoom -> 1.5x")
            elif key == ord('8'):
                zoom_val = 2.0
                print("Zoom -> 2.0x")
            elif key == ord('9'):
                zoom_val = 3.0
                print("Zoom -> 3.0x")

            # Exposure Mode & adjust
            elif key in (ord('e'), ord('E')):
                exp_mode = "manual" if exp_mode == "auto" else "auto"
                if controller:
                    if exp_mode == "auto":
                        if hasattr(controller, "exposure_mode"):
                            controller.exposure_mode = "auto"
                        elif hasattr(controller, "_set_property_auto"):
                            controller._set_property_auto("exposure")
                    else:
                        controller.exposure = exp_val
                print(f"Exposure Mode -> {exp_mode.upper()}")
            elif key == ord('('):
                exp_mode = "manual"
                exp_val = max(exp_min, exp_val - 1)
                if controller and hasattr(controller, "exposure"):
                    try:
                        controller.exposure = exp_val
                        print(f"Exposure -> {controller.exposure}")
                    except Exception as e:
                        print(f"Exposure error: {e}")
            elif key == ord(')'):
                exp_mode = "manual"
                exp_val = min(exp_max, exp_val + 1)
                if controller and hasattr(controller, "exposure"):
                    try:
                        controller.exposure = exp_val
                        print(f"Exposure -> {controller.exposure}")
                    except Exception as e:
                        print(f"Exposure error: {e}")

            # Brightness / Contrast
            elif key == ord('b'):
                brightness_val = max(bright_min, brightness_val - 2)
                if controller and hasattr(controller, "brightness"):
                    try:
                        controller.brightness = brightness_val
                        print(f"Brightness -> {controller.brightness}")
                    except Exception as e:
                        print(f"Brightness error: {e}")
            elif key == ord('B'):
                brightness_val = min(bright_max, brightness_val + 2)
                if controller and hasattr(controller, "brightness"):
                    try:
                        controller.brightness = brightness_val
                        print(f"Brightness -> {controller.brightness}")
                    except Exception as e:
                        print(f"Brightness error: {e}")
            elif key == ord('c'):
                contrast_val = max(contrast_min, contrast_val - 2)
                if controller and hasattr(controller, "contrast"):
                    try:
                        controller.contrast = contrast_val
                        print(f"Contrast -> {controller.contrast}")
                    except Exception as e:
                        print(f"Contrast error: {e}")
            elif key == ord('C'):
                contrast_val = min(contrast_max, contrast_val + 2)
                if controller and hasattr(controller, "contrast"):
                    try:
                        controller.contrast = contrast_val
                        print(f"Contrast -> {controller.contrast}")
                    except Exception as e:
                        print(f"Contrast error: {e}")

            # Print or Save
            elif key in (ord('p'), ord('P')):
                settings = {
                    "focus_mode": focus_mode,
                    "focus_value": focus_val,
                    "zoom": zoom_val,
                    "exposure_mode": exp_mode,
                    "exposure_value": exp_val,
                    "brightness": brightness_val,
                    "contrast": contrast_val,
                }
                print_current_config_snippet(settings)

            elif key in (ord('s'), ord('S')):
                settings = {
                    "focus_mode": focus_mode,
                    "focus_value": focus_val,
                    "zoom": zoom_val,
                    "exposure_mode": exp_mode,
                    "exposure_value": exp_val,
                    "brightness": brightness_val,
                    "contrast": contrast_val,
                }
                update_config_file(settings)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass
        print("Tuner closed.")


if __name__ == "__main__":
    main()