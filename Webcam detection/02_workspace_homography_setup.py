"""
Step 2: Compute the pixel -> real-world-millimetre mapping for your table.

Run this ONCE after the rig (camera height/angle) is physically fixed in
place, and re-run it any time the camera or the markers move.

The script:
    1. Opens the Arducam camera.
    2. Shows a live view of the empty table.
    3. Press SPACE to capture the workspace image.
    4. Undistorts the image using the camera calibration from Step 1.
    5. Detects the 4 ArUco markers.
    6. Uses the OUTER corner of each marker.
    7. Computes the homography from pixel coordinates -> real-world mm.
    8. Saves the homography.

Controls:
    SPACE = capture and calibrate
    ENTER = exit

Usage:
    python 02_workspace_homography_setup.py
"""

import cv2
import numpy as np
import os
import config
import camera_utils


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

# Folder for workspace photos
WORKSPACE_PHOTOS_DIR = os.path.join(
    config.CALIB_DIR,
    "workspace_photos"
)

os.makedirs(WORKSPACE_PHOTOS_DIR, exist_ok=True)


# ---------------------------------------------------------
# Undistort image
# ---------------------------------------------------------

def undistort(img):
    return camera_utils.undistort(img)


# ---------------------------------------------------------
# Capture workspace image
# ---------------------------------------------------------

def capture_workspace():

    cap, controller = camera_utils.open_configured_camera()

    print("Camera opened.")
    print("Place the EMPTY table and all 4 ArUco markers in view.")
    print("SPACE = capture and calibrate")
    print("ENTER = exit")

    window_name = "Workspace Calibration (Live Undistorted View)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    zoom_factor = float(getattr(config, "CAMERA_ZOOM", getattr(config, "CAMERA_ZOOM_VALUE", 1.0)))
    try:
        while True:

            ret, raw_frame = cap.read()

            if not ret:
                print("Failed to read frame")
                break

            frame = camera_utils.apply_digital_zoom(raw_frame, zoom_factor)

            # Show live undistorted preview so user sees straight table lines
            frame_undist = undistort(frame)

            cv2.imshow(window_name, frame_undist)

            # IMPORTANT:
            # waitKey must be called after imshow
            key = cv2.waitKey(1)

            # SPACE
            if key == 32:

                filename = os.path.join(
                    WORKSPACE_PHOTOS_DIR,
                    "workspace.jpg"
                )

                success = cv2.imwrite(
                    filename,
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 100]
                )

                if success:
                    print(f"\nWorkspace photo saved: {filename}")
                    cap.release()
                    cv2.destroyAllWindows()
                    if controller is not None:
                        try:
                            controller.close()
                        except Exception:
                            pass
                    return filename

                else:
                    print("ERROR: Could not save workspace image")

            # ENTER or ESC
            elif key in (10, 13, 27):

                print("Exit requested.")

                cap.release()
                cv2.destroyAllWindows()
                if controller is not None:
                    try:
                        controller.close()
                    except Exception:
                        pass

                raise SystemExit(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass

    raise SystemExit(1)


# ---------------------------------------------------------
# Calculate workspace homography
# ---------------------------------------------------------

def calculate_homography(image_path):

    print("\nStarting workspace calibration...")

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read image: {image_path}")
        raise SystemExit(1)

    # -----------------------------------------------------
    # Undistort
    # -----------------------------------------------------

    img_undist = undistort(img)

    # -----------------------------------------------------
    # ArUco detector
    # -----------------------------------------------------

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(
            cv2.aruco,
            config.ARUCO_DICT
        )
    )

    aruco_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        aruco_params
    )

    gray = cv2.cvtColor(
        img_undist,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, _ = detector.detectMarkers(gray)

    # -----------------------------------------------------
    # Check markers
    # -----------------------------------------------------

    if ids is None:

        print(
            "No ArUco markers detected.\n"
            "Check lighting, focus, and that all 4 markers "
            "are visible and flat."
        )

        raise SystemExit(1)

    ids = ids.flatten()

    print(
        f"Detected marker IDs: "
        f"{sorted([int(i) for i in ids])}"
    )

    # -----------------------------------------------------
    # Get OUTER corner of every marker
    #
    # OpenCV ArUco corner order:
    #
    #       [0] -------- [1]
    #        |            |
    #        |            |
    #       [3] -------- [2]
    #
    # For markers arranged like:
    #
    #       0 -------- 1
    #       |          |
    #       |          |
    #       3 -------- 2
    #
    # the outer corners are:
    #
    # marker 0 -> corner [0]
    # marker 1 -> corner [1]
    # marker 2 -> corner [2]
    # marker 3 -> corner [3]
    # -----------------------------------------------------

    outer_corner_index = {
        0: 0,   # top-left marker
        1: 1,   # top-right marker
        2: 2,   # bottom-right marker
        3: 3,   # bottom-left marker
    }

    detected = {}

    for marker_id, marker_corners in zip(
        ids,
        corners
    ):

        marker_id = int(marker_id)

        if marker_id not in outer_corner_index:
            print(
                f"Warning: unexpected marker ID "
                f"{marker_id}, ignoring it."
            )
            continue

        corner_index = outer_corner_index[marker_id]

        point = marker_corners[0][corner_index]

        detected[marker_id] = point

        print(
            f"Marker {marker_id}: "
            f"outer corner pixel = "
            f"({point[0]:.2f}, {point[1]:.2f})"
        )

    # -----------------------------------------------------
    # Check that all expected markers exist
    # -----------------------------------------------------

    expected_ids = set(
        config.MARKER_WORLD_POSITIONS_MM.keys()
    )

    missing = expected_ids - set(detected.keys())

    if missing:

        print(
            f"\nMissing expected markers: {missing}"
        )

        print(
            "Make sure all four ArUco markers are visible "
            "and correctly detected."
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # Build pixel and world point arrays
    # -----------------------------------------------------

    pixel_pts = []
    world_pts = []

    for marker_id, world_xy in (
        config.MARKER_WORLD_POSITIONS_MM.items()
    ):

        pixel_pts.append(
            detected[marker_id]
        )

        world_pts.append(
            world_xy
        )

    pixel_pts = np.array(
        pixel_pts,
        dtype=np.float32
    )

    world_pts = np.array(
        world_pts,
        dtype=np.float32
    )

    print("\nPixel points:")
    print(pixel_pts)

    print("\nWorld points (mm):")
    print(world_pts)

    # -----------------------------------------------------
    # Calculate homography
    # -----------------------------------------------------

    H, mask = cv2.findHomography(
        pixel_pts,
        world_pts,
        method=0
    )

    if H is None:

        print(
            "\nERROR: Could not calculate homography."
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # Sanity check
    # -----------------------------------------------------

    reprojected = cv2.perspectiveTransform(
        pixel_pts.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    errors_mm = np.linalg.norm(
        reprojected - world_pts,
        axis=1
    )

    print(
        "\nPer-marker reprojection error "
        "(should be near 0 mm):"
    )

    for marker_id, err in zip(
        config.MARKER_WORLD_POSITIONS_MM.keys(),
        errors_mm
    ):

        print(
            f"  marker {marker_id}: "
            f"{err:.2f} mm"
        )

    if errors_mm.max() > 5.0:

        print(
            "\nWARNING: >5 mm error on a marker."
        )

        print(
            "Double-check the measured marker positions "
            "in config.py and make sure the markers are flat."
        )

    # -----------------------------------------------------
    # Save homography
    # -----------------------------------------------------

    np.savez(
        config.HOMOGRAPHY_FILE,
        H=H
    )

    print(
        f"\nSaved homography to "
        f"{config.HOMOGRAPHY_FILE}"
    )

    # -----------------------------------------------------
    # Save readable TXT file as well
    # -----------------------------------------------------

    txt_file = os.path.splitext(
        config.HOMOGRAPHY_FILE
    )[0] + ".txt"

    with open(txt_file, "w") as f:

        f.write(
            "WORKSPACE HOMOGRAPHY CALIBRATION\n"
        )

        f.write(
            "=" * 50 + "\n\n"
        )

        f.write(
            f"Workspace image: {image_path}\n\n"
        )

        f.write(
            "Detected marker IDs:\n"
        )

        f.write(
            str(sorted(detected.keys()))
        )

        f.write("\n\n")

        f.write(
            "Pixel points (outer marker corners):\n"
        )

        f.write(
            np.array2string(
                pixel_pts,
                precision=4
            )
        )

        f.write("\n\n")

        f.write(
            "World points (mm):\n"
        )

        f.write(
            np.array2string(
                world_pts,
                precision=4
            )
        )

        f.write("\n\n")

        f.write(
            "Homography matrix H:\n"
        )

        f.write(
            np.array2string(
                H,
                precision=10
            )
        )

        f.write("\n\n")

        f.write(
            "Reprojection errors (mm):\n"
        )

        for marker_id, err in zip(
            config.MARKER_WORLD_POSITIONS_MM.keys(),
            errors_mm
        ):

            f.write(
                f"Marker {marker_id}: "
                f"{err:.6f} mm\n"
            )

    print(
        f"Saved readable homography to "
        f"{txt_file}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    image_path = capture_workspace()

    calculate_homography(image_path)


if __name__ == "__main__":
    main()