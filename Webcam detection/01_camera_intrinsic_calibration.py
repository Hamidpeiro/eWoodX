"""
Step 1: Intrinsic camera calibration (lens distortion correction).
Supports both OpenCV Fisheye (cv2.fisheye for 120°+ lenses) and Standard Pinhole models.

How to shoot the calibration photos:
    1. Print a checkerboard (config.CHECKERBOARD_INNER_CORNERS = 13x8 inner corners by default).
       Measure printed square size with calipers and set config.CHECKERBOARD_SQUARE_SIZE_MM.
    2. Tape it FLAT to something rigid (a flat board or acrylic).
    3. Mount camera in its fixed rig position with manual focus and lighting locked.
    4. Take 5-10 photos of the checkerboard covering different regions of the frame
       (corners, edges, center, tilted at moderate angles).
       Save them into calibration_data/checkerboard_photos/ or run with --live to snap them interactively.
    5. Run this script.

Usage:
    python 01_camera_intrinsic_calibration.py
    python 01_camera_intrinsic_calibration.py --live    (interactively capture checkerboard photos from webcam)
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
import config
import camera_utils

PHOTOS_DIR = os.path.join(config.CALIB_DIR, "checkerboard_photos")
os.makedirs(PHOTOS_DIR, exist_ok=True)


def capture_checkerboard_live():
    """Interactive tool to capture checkerboard photos with hardware settings applied."""
    print("\n" + "=" * 60)
    print("LIVE CHECKERBOARD PHOTO CAPTURE")
    print("=" * 60)
    print("Controls:")
    print("  SPACE / C -> Capture current frame and check for checkerboard")
    print("  Q / ESC   -> Finish capturing and proceed to calibration")
    print("=" * 60)

    cap, controller = camera_utils.open_configured_camera()
    cols, rows = config.CHECKERBOARD_INNER_CORNERS

    count = len(glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")))

    window_name = "Live Checkerboard Capture - Press SPACE to capture, Q to finish"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    zoom_factor = float(getattr(config, "CAMERA_ZOOM", getattr(config, "CAMERA_ZOOM_VALUE", 1.0)))
    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("Failed to read camera frame.")
                break

            frame = camera_utils.apply_digital_zoom(raw_frame, zoom_factor)
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            status_text = f"Photos saved: {count} | Board detected: {'YES' if found else 'NO'}"
            color = (0, 255, 0) if found else (0, 0, 255)
            if found:
                cv2.drawChessboardCorners(display, (cols, rows), corners, found)

            cv2.putText(display, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, "SPACE = Save Photo | Q = Finish & Calibrate", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord(' '), ord('c'), ord('C')):
                if not found:
                    print("Warning: Checkerboard not fully detected in frame. Saving anyway...")
                count += 1
                img_path = os.path.join(PHOTOS_DIR, f"calib_{count:03d}.jpg")
                cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                print(f"Saved: {img_path} (Total: {count})")

            elif key in (ord('q'), ord('Q'), 27):
                print(f"\nFinished capturing. Total photos in folder: {count}")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass


def calibrate():
    cols, rows = config.CHECKERBOARD_INNER_CORNERS
    square_mm = config.CHECKERBOARD_SQUARE_SIZE_MM
    calib_model = getattr(config, "CALIBRATION_MODEL", "fisheye").lower()
    is_fisheye_mode = (calib_model == "fisheye")

    print(f"\n============================================================")
    print(f"CAMERA INTRINSIC CALIBRATION ({'OPENCV FISHEYE' if is_fisheye_mode else 'STANDARD PINHOLE'})")
    print(f"Checkerboard inner corners: {cols}x{rows} | Square size: {square_mm} mm")
    print(f"============================================================")

    # 3D points of checkerboard corners in board coordinate space (Z=0)
    # Shape for standard: (cols*rows, 3) float32
    # Shape for fisheye:  (cols*rows, 1, 3) float64
    if is_fisheye_mode:
        objp = np.zeros((1, cols * rows, 3), np.float64)
        objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    else:
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    valid_paths = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_paths = sorted(glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")) +
                         glob.glob(os.path.join(PHOTOS_DIR, "*.png")))

    if not image_paths:
        print(f"\nNo photos found in {PHOTOS_DIR}")
        print("Run 'python 01_camera_intrinsic_calibration.py --live' to capture photos with the webcam,")
        print("or add 15-25 checkerboard images into that folder.")
        sys.exit(1)

    image_size = None
    print(f"Processing {len(image_paths)} images from {PHOTOS_DIR}...")

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [SKIP] Unreadable: {os.path.basename(path)}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        found, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print(f"  [SKIP] Checkerboard not found: {os.path.basename(path)}")
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        if is_fisheye_mode:
            # cv2.fisheye requires (1, N, 2) float64
            imgpoints.append(np.asarray(corners_refined, dtype=np.float64).reshape(1, -1, 2))
            objpoints.append(objp.copy())
        else:
            imgpoints.append(corners_refined)
            objpoints.append(objp.copy())

        valid_paths.append(path)
        print(f"  [OK]   Detected corners in: {os.path.basename(path)}")

    accepted = len(imgpoints)
    print(f"\nAccepted {accepted}/{len(image_paths)} photos.")

    if accepted < 5:
        print("ERROR: Fewer than 5 valid checkerboard photos detected.")
        print("Calibration requires at least 10-20 good photos across the field of view.")
        sys.exit(1)

    print("\nRunning camera calibration algorithm...")

    if is_fisheye_mode:
        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)

        calib_flags = 0
        flag_check_cond = getattr(cv2, "CALIB_CHECK_COND", getattr(cv2.fisheye, "CALIB_CHECK_COND", 16777216))
        flag_recompute = getattr(cv2, "CALIB_RECOMPUTE_EXTRINSIC", getattr(cv2.fisheye, "CALIB_RECOMPUTE_EXTRINSIC", 8388608))
        flag_fix_skew = getattr(cv2, "CALIB_FIX_SKEW", getattr(cv2.fisheye, "CALIB_FIX_SKEW", 33554432))

        if getattr(config, "FISHEYE_CHECK_COND", True):
            calib_flags |= flag_check_cond
        if getattr(config, "FISHEYE_RECOMPUTE_EXTRINSIC", True):
            calib_flags |= flag_recompute
        if getattr(config, "FISHEYE_FIX_SKEW", True):
            calib_flags |= flag_fix_skew

        fisheye_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        try:
            rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                image_size,
                K,
                D,
                None,
                None,
                calib_flags,
                fisheye_criteria
            )
        except cv2.error as e:
            print(f"\n[Fisheye Calibrate Warning] Optimization failed with CHECK_COND: {e}")
            print("Retrying without CALIB_CHECK_COND...")
            calib_flags &= ~flag_check_cond
            rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                image_size,
                K,
                D,
                None,
                None,
                calib_flags,
                fisheye_criteria
            )
    else:
        rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Model:                    {'OpenCV Fisheye' if is_fisheye_mode else 'Standard Pinhole'}")
    print(f"Image Resolution:         {image_size[0]} x {image_size[1]}")
    print(f"Accepted Photos:          {accepted}")
    print(f"RMS Reprojection Error:   {rms_error:.4f} px (lower is better; <0.5 px is ideal)")
    print("\nCamera Matrix (K):\n", np.array2string(camera_matrix, precision=4, suppress_small=True))
    print("\nDistortion Coefficients (D):\n", np.array2string(dist_coeffs.ravel(), precision=6))

    # Save to NPZ
    np.savez(
        config.CAMERA_CALIB_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms_error=rms_error,
        is_fisheye=is_fisheye_mode,
        calibration_model="fisheye" if is_fisheye_mode else "standard",
    )
    print(f"\nSaved calibration data -> {config.CAMERA_CALIB_FILE}")

    # Save human-readable TXT
    txt_file = os.path.splitext(config.CAMERA_CALIB_FILE)[0] + ".txt"
    with open(txt_file, "w") as f:
        f.write("CAMERA INTRINSIC CALIBRATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Calibration Model: {'OpenCV Fisheye' if is_fisheye_mode else 'Standard Pinhole'}\n")
        f.write(f"Calibration File:  {config.CAMERA_CALIB_FILE}\n")
        f.write(f"Accepted Photos:   {accepted}\n")
        f.write(f"Image Resolution:  {image_size[0]} x {image_size[1]}\n")
        f.write(f"RMS Error:         {rms_error:.6f} px\n\n")
        f.write("Camera Matrix (K):\n")
        f.write(np.array2string(camera_matrix, precision=8))
        f.write("\n\nDistortion Coefficients (D):\n")
        f.write(np.array2string(dist_coeffs.ravel(), precision=8))
        f.write("\n")

    print(f"Saved human-readable report -> {txt_file}")

    # Generate a test undistorted image if sample exists
    if valid_paths:
        test_img = cv2.imread(valid_paths[0])
        if test_img is not None:
            undistorted = camera_utils.get_undistorter(config.CAMERA_CALIB_FILE).undistort(test_img)
            preview_path = os.path.join(config.CALIB_DIR, "undistort_test_sample.jpg")
            cv2.imwrite(preview_path, undistorted)
            print(f"Saved undistorted sample test -> {preview_path}")

    print("\nStep 1 Complete! You can now proceed to Step 2: python 02_workspace_homography_setup.py")


def main():
    parser = argparse.ArgumentParser(description="Intrinsic Camera & Fisheye Calibration")
    parser.add_argument("--live", action="store_true", help="Launch live camera feed to capture checkerboard photos interactively")
    args = parser.parse_args()

    if args.live:
        capture_checkerboard_live()

    calibrate()


if __name__ == "__main__":
    main()