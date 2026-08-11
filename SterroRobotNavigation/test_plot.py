import cv2
import numpy as np
import parameters as par
import SAD
import chessboard
import csv
import os
import plot
import optional1 as opt1
import optional2 as opt2
import optional3 as opt3

#enable optionals: [opt1, opt2, opt3]
Enables = [False, True, True] 

#output folder
os.makedirs("output", exist_ok=True)

#video input
path_L = 'robot-navigation-video/robotL_conv.mp4' #.avi converted into .mp4 by using homebrew
path_R = 'robot-navigation-video/robotR_conv.mp4'

capL = cv2.VideoCapture(path_L)
capR = cv2.VideoCapture(path_R)

if not capL.isOpened():
    print('ERROR: Cannot open robotL_conv.mp4')
    exit()
if not capR.isOpened():
    print('ERROR: Cannot open robotR_conv.mp4')
    exit()

print(f'Frames L: {int(capL.get(cv2.CAP_PROP_FRAME_COUNT))}')
print(f'Frames R: {int(capR.get(cv2.CAP_PROP_FRAME_COUNT))}')
print(f'FPS: {capL.get(cv2.CAP_PROP_FPS)}')
print(f'Resolution: {int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))}')

#get video information for video output
fps = capL.get(cv2.CAP_PROP_FPS)
width = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))

def make_writer(path, frame_width):
    """Try avc1 first, fall back to MJPG."""
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(path, fourcc, fps, (frame_width, height))
    if not writer.isOpened():
        path   = path.replace('.mp4', '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(path, fourcc, fps, (frame_width, height))
    print(f'VideoWriter [{path}]: {"OK" if writer.isOpened() else "FAILED"}')
    return writer

#main video: left frame + disparity map
out_main = make_writer("output/output_video.mp4", width * 2)

#opt2 video
out_opt2 = make_writer("output/output_opt2.mp4", width * 2) if Enables[1] else None

#opt 3 video
out_opt3 = make_writer("output/output_opt3.mp4", width * 2) if Enables[2] else None

#setup csv
csv_file       = open("output/risultati.csv",  "w", newline="")
csv_file_chess = open("output/chessboard.csv", "w", newline="")
csv_writer     = csv.writer(csv_file)
writer_chess   = csv.writer(csv_file_chess)

csv_writer.writerow(["frame", "d_main_px", "z_mm", "z_m", "alarm"])
writer_chess.writerow(["frame", "z_center_m", "d_chess_px", "z_chess_m",
                        "W_est_mm", "H_est_mm", "err_W_mm", "err_H_mm"])

#lists for final plots and analysis
frames_list = []
dmain_list  = []
dist_list   = []
alarm_list  = []

chess_frames = []
dchess_list  = []
zchess_list  = []
West_list    = []
Hest_list    = []
errW_list    = []
errH_list    = []

stripe_data  = []

#opt3 comparison: SAD vs Moravec+uniqueness
frames_base  = []
dist_base    = []
dmain_base   = []
frames_opt3  = []
dist_opt3    = []
dmain_opt3   = []

#state initialization
d_main_prev = None
frame_idx   = 0

#MAIN LOOP
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    frame_idx += 1
    h, w = frameL.shape[:2]
    cx, cy = w // 2, h // 2

    #frameL_clean → only for chessboard detection
    #frameL_display → for overlays
    frameL_clean = frameL.copy()
    frameL_display = frameL.copy()

    #convert into grayscale
    grayL = cv2.cvtColor(frameL_clean, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR,       cv2.COLOR_BGR2GRAY)

    #compute SAD
    d_min_opt1, d_max_opt1 = 0, par.d_range  #defaults

    if Enables[0] and d_main_prev is not None:  #opt1 modifies base range
        d_min_opt1, d_max_opt1 = opt1.get_dynamic_range(d_main_prev)
        disp_map_base = SAD.compute_sad_disparity(grayL, grayR, par.window,
                                                   d_max_opt1 - d_min_opt1,
                                                   d_min=d_min_opt1)
    else:
        disp_map_base = SAD.compute_sad_disparity(grayL, grayR, par.window, par.d_range)

    d_main_base = SAD.get_dmain(disp_map_base, cx, cy, par.center_area, par.d_range)

    #opt3: Moravec + uniqueness
    if Enables[2]:
        disp_map_robust, texture_mask, _ = opt3.compute_sad_moravec(
            grayL, grayR, par.window, par.d_range)
        _, valid_mask = opt3.compute_sad_robust(
            grayL, grayR, par.window, par.d_range)
        disp_map_robust[~valid_mask] = 0
        d_main_robust = opt3.get_dmain_robust(
            disp_map_robust, cx, cy, par.center_area, par.d_range)

        #use robust disparity
        disp_map = disp_map_robust
        d_main   = d_main_robust
    else:
        disp_map = disp_map_base
        d_main   = d_main_base

    if d_main is not None:
        d_main_prev = d_main

    #disparity visualization
    disp_visual = cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color  = cv2.applyColorMap(disp_visual, cv2.COLORMAP_JET)

    #central area rectangle
    half = par.center_area // 2
    cv2.rectangle(frameL_display, (cx-half, cy-half), (cx+half, cy+half), (0, 255, 0), 2)
    cv2.rectangle(disp_color,     (cx-half, cy-half), (cx+half, cy+half), (255, 255, 255), 2)

    #distance and alarm
    alarm     = False
    z_mm, z_m = None, None

    if d_main is not None and d_main > 0:
        z_mm  = SAD.compute_distance(par.b, par.f, d_main)
        z_m   = z_mm / 1000.0
        alarm = z_m < par.alarm_threshold

        color = (0, 0, 255) if alarm else (0, 255, 0)

        if alarm:
            cv2.putText(frameL_display, "ALARM",
                        (cx - 100, cy - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        cv2.putText(frameL_display, f"Distance: {z_m:.2f} m",
                    (cx - 120, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frameL_display, f"d_main: {d_main} px",
                    (cx - 120, cy + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        #chessboard (on clean frame)
        found, corners, _ = chessboard.find_chessboard(frameL_clean, par.cols, par.rows)
        if found:
            w_px, h_px = chessboard.compute_chessboard_dimension(corners, par.cols, par.rows)
            W_est, H_est = chessboard.compute_real_chessboard_dimension(z_mm, w_px, h_px, par.f)

            err_W_signed = round(W_est - par.w_real, 1)
            err_H_signed = round(H_est - par.h_real, 1)
            err_W_abs    = abs(err_W_signed)  #absolute → plot/print
            err_H_abs    = abs(err_H_signed)

            #draw chess corners
            frameL_display = chessboard.draw_chessboard_overlay(
                frameL_display, corners, found,
                par.cols, par.rows, w_px, h_px,
                W_est, H_est, z_m, par.w_real, par.h_real)

            #save data
            chess_frames.append(frame_idx)
            dchess_list.append(d_main)
            zchess_list.append(z_m)
            West_list.append(round(W_est, 1))
            Hest_list.append(round(H_est, 1))
            errW_list.append(err_W_abs)
            errH_list.append(err_H_abs)
            writer_chess.writerow([frame_idx, round(z_m, 4), d_main,
                                   round(z_m, 4), round(W_est, 1), round(H_est, 1),
                                   err_W_signed, err_H_signed])

            print(f"  ♟ Chess | W={W_est:.1f}mm (err {err_W_abs:.1f}) "
                  f"| H={H_est:.1f}mm (err {err_H_abs:.1f})")

        #save data
        frames_list.append(frame_idx)
        dmain_list.append(d_main)
        dist_list.append(z_m)
        alarm_list.append(int(alarm))
        csv_writer.writerow([frame_idx, d_main, round(z_mm, 1), round(z_m, 4), int(alarm)])

        #opt3 comparison data
        if d_main_base is not None and d_main_base > 0:
            z_base = SAD.compute_distance(par.b, par.f, d_main_base) / 1000.0
            frames_base.append(frame_idx)
            dist_base.append(z_base)
            dmain_base.append(d_main_base)

        if Enables[2] and d_main_robust is not None and d_main_robust > 0:
            frames_opt3.append(frame_idx)
            dist_opt3.append(z_m)
            dmain_opt3.append(d_main_robust)

        print(f"Frame {frame_idx:03d} | d_main={d_main}px | z={z_m:.3f}m "
              f"{'⚠️  ALARM' if alarm else ''}")

    else:
        cv2.putText(frameL_display, "Distance: N/A",
                    (cx - 100, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

    #main video
    combined_main = np.hstack((frameL_display, disp_color))
    out_main.write(combined_main)
    cv2.imshow("Main | Left + Disparity", combined_main)

    #opt2 video
    if Enables[1]:
        stripes = opt2.compute_stripe_disparities(disp_map, par.n_stripes, par.d_range, SAD)
        stripes = opt2.compute_stripe_distances(stripes, par.f, par.b)

        row = {"frame": frame_idx}
        for s in stripes:
            row[f"z{s['idx']}"] = s["z_m"]
        stripe_data.append(row)

        frameL_opt2 = frameL_display.copy()
        frameL_opt2 = opt2.draw_stripes_overlay(frameL_opt2, stripes, par.f, par.b)

        combined_opt2 = np.hstack((frameL_opt2, disp_color))
        out_opt2.write(combined_opt2)
        cv2.imshow("Opt2 | Stripes + Disparity", combined_opt2)

    #opt3 video
    if Enables[2]:
        frameL_opt3 = frameL_display.copy()
        frameL_opt3 = opt3.draw_moravec_overlay(frameL_opt3, texture_mask)

        combined_opt3 = np.hstack((frameL_opt3, disp_color))
        out_opt3.write(combined_opt3)
        cv2.imshow("Opt3 | Moravec + Disparity", combined_opt3)

    if Enables[0]:
        print(f"  OPT1 | range=[{d_min_opt1}, {d_max_opt1}] offset={d_min_opt1}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#clean up
capL.release()
capR.release()
out_main.release()
if out_opt2: out_opt2.release()
if out_opt3: out_opt3.release()
cv2.destroyAllWindows()
csv_file.close()
csv_file_chess.close()

#plots
plot.print_summary(frames_list, dist_list, alarm_list, chess_frames)
plot.plot_navigation(frames_list, dist_list, dmain_list, alarm_list)
plot.plot_chessboard(chess_frames, West_list, Hest_list, errW_list, errH_list,
                     frames_list, dist_list, par.w_real, par.h_real)
if Enables[1]:
    plot.plot_stripes(stripe_data)
if Enables[2]:
    plot.plot_opt3_comparison(frames_base, dist_base, dmain_base,
                              frames_opt3, dist_opt3, dmain_opt3)