"""
Optional 3 — Robust disparity estimation

Two complementary approaches to filter unreliable disparity estimates:

  A) Uniqueness filtering — a match is accepted only if the best SAD score
     is sufficiently lower than the second-best, ensuring an unambiguous
     minimum in the cost curve.

  B) Moravec interest operator — disparity is computed only at pixels with
     sufficient local texture (high intensity variation in all directions),
     discarding matches in uniform, featureless regions.

Both methods are designed to be combined: Moravec selects candidate pixels
based on texture, uniqueness filtering removes ambiguous matches among them.
"""
import cv2
import numpy as np


# ── A: Uniqueness filtering ───────────────────────────────────────────────────

def compute_sad_robust(grayL, grayR, win_size, disp_range, d_min=0,
                        uniqueness_ratio=0.15):
    """
    SAD stereo matching with uniqueness filtering.

    A pixel's disparity is accepted only if:
        (second_best_SAD - best_SAD) / best_SAD > uniqueness_ratio

    i.e. the best match is distinctly better than the second-best.
    Ambiguous pixels are set to zero in the output disparity map.

    Args:
        grayL            : left grayscale frame (reference)
        grayR            : right grayscale frame
        win_size         : matching window size (pixels)
        disp_range       : number of disparities to search
        d_min            : starting disparity offset
        uniqueness_ratio : minimum relative gap between best and second-best SAD
                           (higher = more selective, typical range 0.10–0.20)

    Returns:
        disparity  : float32 disparity map (unreliable pixels set to 0)
        valid_mask : boolean mask, True where match is considered reliable
    """
    L = grayL.astype(np.float32)
    R = grayR.astype(np.float32)
    h, w = L.shape

    best_sad    = np.full((h, w), np.inf, dtype=np.float32)
    second_best = np.full((h, w), np.inf, dtype=np.float32)
    disparity   = np.zeros((h, w), dtype=np.float32)

    for d in range(d_min, d_min + disp_range):
        shifted_R          = np.roll(R, d, axis=1)
        shifted_R[:, :d]   = 0
        sad = cv2.boxFilter(np.abs(L - shifted_R), ddepth=-1,
                            ksize=(win_size, win_size), normalize=False)

        is_best                  = sad < best_sad
        second_best[is_best]     = best_sad[is_best]   # demote previous best
        best_sad[is_best]        = sad[is_best]
        disparity[is_best]       = d

        is_second                = (~is_best) & (sad < second_best)
        second_best[is_second]   = sad[is_second]

    with np.errstate(divide='ignore', invalid='ignore'):
        uniqueness = np.where(best_sad > 0,
                              (second_best - best_sad) / best_sad, 0)

    valid_mask          = uniqueness > uniqueness_ratio
    disparity[~valid_mask] = 0

    return disparity, valid_mask


# ── B: Moravec interest operator ─────────────────────────────────────────────

def moravec_interest(gray, win_size=7, threshold=100):
    """
    Compute the Moravec interest operator for every pixel.

    For each pixel, measures the minimum sum of squared intensity differences
    across four shift directions (horizontal, vertical, and two diagonals).
    Pixels with interest above the threshold are considered sufficiently
    textured for reliable stereo matching.

    Args:
        gray      : grayscale image (uint8)
        win_size  : local summation window size
        threshold : minimum interest value to be considered textured

    Returns:
        mask     : boolean array, True at textured pixels
        interest : float32 interest map
    """
    img      = gray.astype(np.float32)
    interest = np.zeros(img.shape, dtype=np.float32)

    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        shifted   = np.roll(np.roll(img, dx, axis=1), dy, axis=0)
        local_sum = cv2.boxFilter((img - shifted) ** 2, ddepth=-1,
                                   ksize=(win_size, win_size), normalize=False)
        interest  = local_sum if np.all(interest == 0) else np.minimum(interest, local_sum)

    return interest > threshold, interest


def compute_sad_moravec(grayL, grayR, win_size, disp_range, d_min=0,
                         moravec_win=7, moravec_thresh=100):
    """
    SAD stereo matching restricted to textured pixels (Moravec operator).

    Computes the Moravec interest map on the reference (left) image and
    sets disparity to zero at non-textured pixels after matching.

    Args:
        grayL          : left grayscale frame (reference)
        grayR          : right grayscale frame
        win_size       : matching window size (pixels)
        disp_range     : number of disparities to search
        d_min          : starting disparity offset
        moravec_win    : window size for Moravec interest computation
        moravec_thresh : interest threshold for texture detection

    Returns:
        disparity    : float32 disparity map (non-textured pixels set to 0)
        texture_mask : boolean mask, True at textured pixels
        interest_map : float32 raw interest values
    """
    texture_mask, interest_map = moravec_interest(grayL, moravec_win, moravec_thresh)

    L = grayL.astype(np.float32)
    R = grayR.astype(np.float32)
    h, w = L.shape

    best_sad  = None
    disparity = np.zeros((h, w), dtype=np.float32)

    for d in range(d_min, d_min + disp_range):
        shifted_R        = np.roll(R, d, axis=1)
        shifted_R[:, :d] = 0
        sad = cv2.boxFilter(np.abs(L - shifted_R), ddepth=-1,
                            ksize=(win_size, win_size), normalize=False)
        if best_sad is None:
            best_sad  = sad.copy()
            disparity = np.full((h, w), d, dtype=np.float32)
        else:
            better            = sad < best_sad
            best_sad[better]  = sad[better]
            disparity[better] = d

    disparity[~texture_mask] = 0
    return disparity, texture_mask, interest_map


# ── Robust d_main with IQR outlier removal ────────────────────────────────────

def get_dmain_robust(disp_map, cx, cy, area_size, disp_range):
    """
    Estimate d_main over a central ROI with IQR-based outlier removal.

    Filters the disparity values in the ROI using the interquartile range
    before computing the histogram mode, reducing the influence of outliers.

    Args:
        disp_map  : disparity map (float32)
        cx, cy    : centre of the ROI (pixels)
        area_size : side length of the ROI (pixels)
        disp_range: disparity range (histogram bins)

    Returns:
        d_main : integer disparity in pixels, or None if no valid pixels
    """
    half = area_size // 2
    h, w = disp_map.shape
    region = disp_map[max(0, cy-half):min(h, cy+half),
                      max(0, cx-half):min(w, cx+half)]
    valid  = region[region > 0]

    if len(valid) == 0:
        return None

    q1, q3   = np.percentile(valid, 25), np.percentile(valid, 75)
    filtered = valid[(valid >= q1 - 1.5 * (q3 - q1)) &
                     (valid <= q3 + 1.5 * (q3 - q1))]

    if len(filtered) == 0:
        return None

    hist, _ = np.histogram(filtered, bins=disp_range, range=(1, disp_range))
    return int(np.argmax(hist) + 1)


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_moravec_overlay(frame, texture_mask):
    """
    Highlight textured pixels (Moravec) with a green tint on the frame.

    Args:
        frame        : BGR input frame (will be copied)
        texture_mask : boolean mask from moravec_interest

    Returns:
        out : annotated copy of the input frame
    """
    out              = frame.copy()
    out[texture_mask] = (out[texture_mask] * 0.5 +
                         np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.putText(out, f"Moravec points: {texture_mask.sum()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out