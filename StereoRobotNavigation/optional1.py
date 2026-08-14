"""
Optional 1 — Dynamic disparity range

Instead of always searching over [0, 128], restricts the search to a
reduced window of size RANGE_SIZE centred around the previous frame's
d_main. This halves computation time and prevents range saturation when
the vehicle is very close to the obstacle (high disparity values).
"""

RANGE_SIZE = 64   # reduced search window (half of the full 128 range)
MAX_DISP   = 128  # absolute disparity limit


def compute_offset(d_main_prev, range_size=RANGE_SIZE):
    """
    Compute the disparity offset o such that d_main_prev lies at the
    centre of the search range [o, o + range_size].

    o = d_main_prev - range_size // 2
    Clamped to [0, MAX_DISP - range_size] to stay within valid bounds.

    Args:
        d_main_prev : d_main from the previous frame (px), or None
        range_size  : size of the reduced disparity window

    Returns:
        o : integer offset in pixels
    """
    if d_main_prev is None:
        return 0  # first frame: start from zero

    o = d_main_prev - range_size // 2
    o = max(0, min(o, MAX_DISP - range_size))
    return o


def get_dynamic_range(d_main_prev, range_size=RANGE_SIZE):
    """
    Return the dynamic disparity search range (d_min, d_max).

    Args:
        d_main_prev : d_main from the previous frame (px), or None
        range_size  : size of the reduced disparity window

    Returns:
        d_min : lower bound of the search range (px)
        d_max : upper bound of the search range (px)
    """
    o = compute_offset(d_main_prev, range_size)
    return o, o + range_size