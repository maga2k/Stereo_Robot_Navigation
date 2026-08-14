"""
Optional 2 — Multi-stripe disparity analysis and planar obstacle view

Divides the frame into N vertical stripes, estimates a separate d_main
for each stripe, computes the corresponding obstacle distance, and
reconstructs a top-down planar view with the angle tau between the
obstacle plane and the camera image plane.
"""
import cv2
import numpy as np
import math

N_STRIPES = 5


def compute_stripe_disparities(disp_map, n_stripes, d_range, SAD_module):
    """
    Divide the disparity map into n_stripes vertical stripes and estimate
    d_main for each stripe using the histogram mode.

    Args:
        disp_map   : disparity map (float32, H x W)
        n_stripes  : number of vertical stripes
        d_range    : disparity range (histogram bins)
        SAD_module : unused, kept for interface compatibility

    Returns:
        stripes : list of dicts with keys idx, x0, x1, cx, d
    """
    h, w     = disp_map.shape
    stripe_w = w // n_stripes
    stripes  = []

    for i in range(n_stripes):
        x0 = i * stripe_w
        x1 = x0 + stripe_w if i < n_stripes - 1 else w
        cx = (x0 + x1) // 2

        region = disp_map[:, x0:x1]
        valid  = region[region > 0]

        if len(valid) == 0:
            stripes.append({"idx": i, "x0": x0, "x1": x1,
                             "cx": cx, "d": None, "z": None})
            continue

        hist, _ = np.histogram(valid, bins=d_range, range=(1, d_range))
        d = int(np.argmax(hist) + 1)
        stripes.append({"idx": i, "x0": x0, "x1": x1, "cx": cx, "d": d})

    return stripes


def compute_stripe_distances(stripes, f, b):
    """
    Add z_mm and z_m distance estimates to each stripe dict.

    Args:
        stripes : list of stripe dicts from compute_stripe_disparities
        f       : focal length (px)
        b       : stereo baseline (mm)

    Returns:
        stripes : updated list with z_mm and z_m fields
    """
    for s in stripes:
        if s["d"] is not None and s["d"] > 0:
            s["z_mm"] = (b * f) / s["d"]
            s["z_m"]  = s["z_mm"] / 1000.0
        else:
            s["z_mm"] = None
            s["z_m"]  = None
    return stripes


def compute_planar_angle(stripes, f, frame_width):
    """
    Estimate the angle tau between the obstacle plane and the camera
    image plane using linear regression on the 3D stripe positions.

    For each stripe, the real-world X coordinate is:
        X_mm = z_mm * (cx - frame_width / 2) / f

    A line Z = m*X + q is fitted, and tau = atan(|m|).

    Args:
        stripes      : list of stripe dicts with z_mm and cx
        f            : focal length (px)
        frame_width  : frame width in pixels

    Returns:
        tau      : angle in degrees (None if fewer than 2 valid stripes)
        points_X : list of real-world X coordinates (mm)
        points_Z : list of real-world Z coordinates (mm)
    """
    points_X, points_Z = [], []

    for s in stripes:
        if s["z_mm"] is None:
            continue
        X_mm = s["z_mm"] * (s["cx"] - frame_width / 2) / f
        points_X.append(X_mm)
        points_Z.append(s["z_mm"])

    if len(points_X) < 2:
        return None, points_X, points_Z

    coeffs = np.polyfit(points_X, points_Z, 1)
    tau    = math.degrees(math.atan(abs(coeffs[0])))
    return tau, points_X, points_Z


def draw_planar_view_with_angle(stripes, f, frame_width, scale=150):
    """
    Render a top-down planar view of the obstacle (style from project PDF):
      - Grey rectangles for each stripe, intensity proportional to distance
      - Dashed horizontal line representing the direction of motion
      - Linear fit of the obstacle profile
      - Angle tau between the obstacle and the image plane

    Args:
        stripes     : list of stripe dicts with z_mm and cx
        f           : focal length (px)
        frame_width : frame width in pixels
        scale       : pixels per metre in the canvas

    Returns:
        canvas : BGR image of the planar view
        tau    : estimated angle in degrees (None if unavailable)
    """
    canvas_w, canvas_h = 500, 400
    canvas   = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    origin_x = 60
    origin_y = canvas_h - 60

    # Camera marker
    cv2.rectangle(canvas, (origin_x - 20, origin_y - 10),
                  (origin_x + 20, origin_y + 10), (200, 0, 0), -1)
    cv2.putText(canvas, "CAM", (origin_x - 15, origin_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Dashed horizontal line (direction of motion)
    for x in range(origin_x, canvas_w - 20, 15):
        cv2.line(canvas, (x, origin_y), (x + 8, origin_y), (0, 0, 0), 2)
    cv2.arrowedLine(canvas, (canvas_w - 60, origin_y),
                    (canvas_w - 20, origin_y), (0, 0, 0), 2, tipLength=0.4)

    tau, _, _ = compute_planar_angle(stripes, f, frame_width)

    grey_levels  = [(60, 60, 60), (90, 90, 90), (120, 120, 120),
                    (150, 150, 150), (190, 190, 190)]
    obstacle_pts = []

    for s in stripes:
        if s["z_mm"] is None:
            continue

        X_mm = s["z_mm"] * (s["cx"] - frame_width / 2) / f
        px   = int(np.clip(origin_x + int(s["z_m"]  * scale),       10, canvas_w - 10))
        py   = int(np.clip(origin_y - int(X_mm / 1000.0 * scale * 0.6), 10, canvas_h - 10))

        color   = grey_levels[s["idx"] % len(grey_levels)]
        rw, rh  = 50, 18
        cv2.rectangle(canvas, (px - rw//2, py - rh//2),
                      (px + rw//2, py + rh//2), color, -1)
        cv2.rectangle(canvas, (px - rw//2, py - rh//2),
                      (px + rw//2, py + rh//2), (0, 0, 0), 1)
        cv2.putText(canvas, f"{s['z_m']:.2f}m",
                    (px - rw//2, py - rh//2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)
        obstacle_pts.append((px, py))

    # Obstacle profile line
    for i in range(len(obstacle_pts) - 1):
        cv2.line(canvas, obstacle_pts[i], obstacle_pts[i + 1], (80, 80, 80), 1)

    # Angle arc and label
    if tau is not None and obstacle_pts:
        arc_c = obstacle_pts[0]
        cv2.ellipse(canvas, arc_c, (30, 30), 0, -180, -180 + int(tau), (0, 0, 0), 1)
        cv2.putText(canvas, f"tau={tau:.1f}°",
                    (arc_c[0] + 5, arc_c[1] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    cv2.putText(canvas, "Top-down planar view",
                (canvas_w // 2 - 75, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return canvas, tau


def draw_stripes_overlay(frame, stripes, f, b):
    """
    Draw vertical stripe boundaries and per-stripe distance labels on a frame.

    Args:
        frame   : BGR input frame (will be copied)
        stripes : list of stripe dicts with z_m and d
        f, b    : unused, kept for interface compatibility

    Returns:
        out : annotated copy of the input frame
    """
    out = frame.copy()
    h   = out.shape[0]

    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255),
              (255, 255, 80), (255, 80, 255)]

    for s in stripes:
        color   = colors[s["idx"] % len(colors)]
        cx_text = (s["x0"] + s["x1"]) // 2 - 25

        cv2.line(out, (s["x0"], 0), (s["x0"], h), color, 1)
        cv2.putText(out, f"{s['z_m']:.2f}m" if s["z_m"] is not None else "N/A",
                    (cx_text, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(out, f"d={s.get('d', '?')}",
                    (cx_text, h // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return out