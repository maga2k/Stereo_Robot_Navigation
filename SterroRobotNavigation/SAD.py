import cv2
import numpy as np


def compute_sad_disparity(grayL, grayR, win_size, disp_range, d_min=0):
    """
    Compute a dense disparity map using the Sum of Absolute Differences (SAD).

    For each pixel in the reference (left) image, shifts the right image by d
    pixels horizontally and selects the disparity minimizing the SAD score
    within a square window.

    Args:
        grayL      : left grayscale frame (reference)
        grayR      : right grayscale frame
        win_size   : side length of the matching window (pixels)
        disp_range : number of disparities to search
        d_min      : starting disparity offset (Optional 1 dynamic range)

    Returns:
        disparity_map : float32 array, shape (H, W)
    """
    L = grayL.astype(np.float32)
    R = grayR.astype(np.float32)
    h, w = L.shape

    best_sad     = None
    disparity_map = np.zeros((h, w), dtype=np.float32)

    for d in range(d_min, d_min + disp_range):
        # Shift right image by d pixels and zero out wrapped columns
        shifted_R = np.roll(R, d, axis=1)
        shifted_R[:, :d] = 0

        # Compute SAD over the window using a box filter
        sad = cv2.boxFilter(np.abs(L - shifted_R), ddepth=-1,
                            ksize=(win_size, win_size), normalize=False)

        # Keep track of the best (minimum) SAD and corresponding disparity
        if best_sad is None:
            best_sad      = sad.copy()
            disparity_map = np.full((h, w), d, dtype=np.float32)
        else:
            better = sad < best_sad
            best_sad[better]      = sad[better]
            disparity_map[better] = d

    return disparity_map


def get_dmain(disp_map, cx, cy, area_size, d_range):
    """
    Estimate the main disparity d_main over a central square ROI.

    Uses the mode of the disparity histogram (most frequent value),
    excluding zero-disparity pixels (unmatched regions).

    Args:
        disp_map  : disparity map (float32)
        cx, cy    : centre of the ROI (pixels)
        area_size : side length of the ROI (pixels)
        d_range   : disparity range (histogram bins)

    Returns:
        d_main : integer disparity in pixels, or None if no valid pixels
    """
    half   = area_size // 2
    region = disp_map[cy - half:cy + half, cx - half:cx + half]
    valid  = region[region > 0]

    if len(valid) == 0:
        return None

    hist, _ = np.histogram(valid, bins=d_range, range=(1, d_range))
    d_main  = int(np.argmax(hist) + 1)
    return d_main


def compute_distance(b, f, d_main):
    """
    Compute obstacle distance using the stereo triangulation formula.

    z (mm) = b (mm) * f (px) / d_main (px)

    Args:
        b      : stereo baseline (mm)
        f      : focal length (px)
        d_main : main disparity (px)

    Returns:
        z : distance in mm
    """
    return (b * f) / d_main