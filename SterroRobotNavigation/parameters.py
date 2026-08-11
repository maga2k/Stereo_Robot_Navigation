#Stereo camera calibration parameters
f = 567.2    #focal length (px)
b = 92.226   #baseline (mm)

#SAD matching
window = 11    # matching window size (px)
d_range = 128   # disparity search range [0, d_range]
center_area = 100   # central ROI side length for d_main estimation (px)

alarm_threshold = 0.8   # obstacle alarm distance (metres)

#chessboard ────────────────────────────────────────────────────────────────
cols = 6 #internal corner columns
rows = 8 #internal corner rows
w_real = 125.0  #known real width  (mm)
h_real = 178.0  #known real height (mm)

# ── Optional 1 — dynamic disparity range ─────────────────────────────────────
RANGE_SIZE = 64   # reduced search window size (half of d_range)

# ── Optional 2 — multi-stripe analysis ───────────────────────────────────────
n_stripes = 5     # number of vertical stripes