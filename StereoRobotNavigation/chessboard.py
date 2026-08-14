import cv2
import numpy as np


def find_chessboard(frame, cols, rows):
    """
    This function detects internal corners of a cheesboard pattern in a frame.
    It applies histogram equalization to improve contrast, and runs findChessboardCorners with adaptive thresholding and sub-pixel refinement on success.

    Args:
        frame : BGR input frame
        cols : number of internal corners along columns
        rows : number of internal corners along rows

    Returns:
        found : bool, whether the pattern was detected
        corners : array of corner coordinates (None if not found)
        gray : preprocessed grayscale image used for detection
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  #enhance contrast

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH + #adaptive threshold
        cv2.CALIB_CB_NORMALIZE_IMAGE + #normalize illumination
        cv2.CALIB_CB_FILTER_QUADS #filter out false quad candidates
    )

    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)

    if found:
        #sub-pixel refinement for more accurate corner localization
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return found, corners, gray


def compute_chessboard_dimension(corners, cols, rows):
    """
    This function computes the pixel span (w_px, h_px) of the chessboard from detected corners.
    It measures the Euclidean distance between the outermost internal corners along the top row (width) and left column (height).

    Args:
        corners : sub-pixel corner array from find_chessboard
        cols : number of internal corners along columns
        rows : number of internal corners along rows

    Returns:
        w_px : pixel width between first and last corner of the top row
        h_px : pixel height between first and last corner of the left column
    """
    pts = corners.reshape(-1, 2)
    top_left = pts[0]
    top_right = pts[cols - 1]
    bot_left = pts[(rows - 1) * cols]

    w_px = float(np.linalg.norm(top_right - top_left))
    h_px = float(np.linalg.norm(bot_left  - top_left))

    return w_px, h_px


def compute_real_chessboard_dimension(z_mm, w_px, h_px, f):
    """
    This function estimates the real-world dimensions of the chessboard using the thin-lens projection model.

    W (mm) = z (mm) * w_px / f (px)
    H (mm) = z (mm) * h_px / f (px)

    Args:
        z_mm : obstacle distance in mm (from stereo triangulation)
        w_px : pixel width of the chessboard
        h_px : pixel height of the chessboard
        f : focal length in pixels

    Returns:
        W_est : estimated real width in mm
        H_est : estimated real height in mm
    """
    W_est = (z_mm * w_px) / f
    H_est = (z_mm * h_px) / f
    return W_est, H_est


def draw_chessboard_overlay(frame, corners, found, cols, rows,
                             w_px, h_px, W_est, H_est, z_m, w_real, h_real):
    """
    This function draws chessboard corners, bounding box, and dimension estimates on a frame.

    Args:
        frame : BGR frame to annotate (will be copied)
        corners : detected corner array
        found : bool from findChessboardCorners
        cols, rows : internal corner grid size
        w_px, h_px : pixel dimensions of the detected pattern
        W_est, H_est : estimated real-world dimensions in mm
        z_m : current obstacle distance in metres
        w_real, h_real : ground-truth dimensions in mm

    Returns:
        out : copy of the input frame
    """
    out = frame.copy()

    if found:
        cv2.drawChessboardCorners(out, (cols, rows), corners, found)

        #bounding box around the detected pattern
        pts        = corners.reshape(-1, 2).astype(int)
        x0, y0     = pts[:, 0].min(), pts[:, 1].min()
        x1, y1     = pts[:, 0].max(), pts[:, 1].max()
        cv2.rectangle(out, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), (0, 255, 255), 2)

        #absolute errors
        err_W = abs(W_est - w_real)
        err_H = abs(H_est - h_real)

        lines = [
            f"z = {z_m:.3f} m",
            f"W est: {W_est:.1f} mm  (real: {w_real:.0f} mm, err: {err_W:.1f} mm)",
            f"H est: {H_est:.1f} mm  (real: {h_real:.0f} mm, err: {err_H:.1f} mm)",
            f"w_px={w_px:.1f}  h_px={h_px:.1f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(out, line, (x0, y0 - 15 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return out