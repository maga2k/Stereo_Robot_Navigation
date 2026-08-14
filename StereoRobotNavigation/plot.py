import matplotlib.pyplot as plt
import numpy as np


def print_summary(frames_list, dist_list, alarm_list, chess_frames, sad_times=None, opt1=False):
    """Print a summary of the processing results to the terminal."""
    print(f"\n📊 Frames processed  : {len(frames_list)}")
    print(f"   Min distance      : {min(dist_list):.3f} m")
    print(f"   Max distance      : {max(dist_list):.3f} m")
    print(f"   Alarm frames      : {sum(alarm_list)}")
    print(f"   Chessboard frames : {len(chess_frames)}")
    if sad_times:
        avg_sad = np.mean(sad_times)
        print(f"\n⏱ SAD average: {avg_sad * 1000:.2f} ms/frame")
        print(f"   SAD minimum: {np.min(sad_times) * 1000:.2f} ms/frame")
        print(f"   SAD maximum: {np.max(sad_times) * 1000:.2f} ms/frame")


def plot_navigation(frames_list, dist_list, dmain_list, alarm_list, sad_times=None, opt1=False, save_path="output/plot_navigation.png"):
    """
    Main navigation plot:
      - Estimated distance over time with alarm shading
      - d_main disparity over time
      - Distance histogram
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 9))
    title_suffix = " — Optional 1" if opt1 else ""
    fig.suptitle("Stereo Robot Navigation — Distance Estimation" + title_suffix, fontsize=13, fontweight="bold")

    # Distance over time
    axes[0].plot(frames_list, dist_list, color="steelblue", linewidth=1.5)
    axes[0].axhline(0.8, color="red", linestyle="--", linewidth=1.2, label="Alarm threshold (0.8 m)")
    axes[0].fill_between(frames_list, dist_list,
                         where=[a == 1 for a in alarm_list],
                         color="red", alpha=0.3, label="Alarm active")
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title("Estimated obstacle distance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # d_main over time
    axes[1].plot(frames_list, dmain_list, color="darkorange", linewidth=1.5)
    axes[1].set_ylabel("Disparity (px)")
    axes[1].set_title("Main disparity d_main over time")
    axes[1].grid(True, alpha=0.3)

    # Distance histogram
    axes[2].hist(dist_list, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[2].axvline(0.8, color="red", linestyle="--", label="0.8 m threshold")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("Frame count")
    axes[2].set_title("Distance distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    times_ms = np.array(sad_times) * 1000

    axes[3].plot(
        frames_list,
        times_ms,
        linewidth=1.2
    )

    mean_time = np.mean(times_ms)

    axes[3].axhline(
        mean_time,
        linestyle="--",
        linewidth=1.2,
        label=f"Mean: {mean_time:.2f} ms/frame"
    )

    axes[3].set_xlabel("Frame")
    axes[3].set_ylabel("Time (ms)")
    axes[3].set_title("SAD computational cost")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    file_suffix = "_opt1" if opt1 else ""
    if save_path.endswith(".png"):
        save_path = save_path[:-4] + file_suffix + ".png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved: {save_path}")


def plot_chessboard(chess_frames, West_list, Hest_list, errW_list, errH_list,
                    frames_list, dist_list, w_real, h_real,
                    save_path="output/plot_chessboard.png"):
    """
    Chessboard dimension estimation plot:
      - Estimated W and H vs ground truth
      - Absolute errors over time with distance on secondary axis
      - Error vs distance scatter (accuracy improves at close range)
    """
    if not chess_frames:
        print("⚠️  No chessboard data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Chessboard — Real Dimension Estimation", fontsize=13, fontweight="bold")

    # Estimated W vs ground truth
    axes[0, 0].plot(chess_frames, West_list, color="steelblue",
                    linewidth=1.5, label="W estimated")
    axes[0, 0].axhline(w_real, color="red", linestyle="--", label=f"W real ({w_real} mm)")
    axes[0, 0].set_ylabel("mm")
    axes[0, 0].set_title("Width W — estimated vs real")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Estimated H vs ground truth
    axes[0, 1].plot(chess_frames, Hest_list, color="darkorange",
                    linewidth=1.5, label="H estimated")
    axes[0, 1].axhline(h_real, color="red", linestyle="--", label=f"H real ({h_real} mm)")
    axes[0, 1].set_ylabel("mm")
    axes[0, 1].set_title("Height H — estimated vs real")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Absolute errors over time + distance on secondary axis
    axes[1, 0].plot(chess_frames, errW_list, color="steelblue",
                    linewidth=1.5, label="|Error W|")
    axes[1, 0].plot(chess_frames, errH_list, color="darkorange",
                    linewidth=1.5, label="|Error H|")
    ax_twin = axes[1, 0].twinx()
    z_at_chess = [dist_list[frames_list.index(f)] for f in chess_frames if f in frames_list]
    ax_twin.plot(chess_frames[:len(z_at_chess)], z_at_chess,
                 color="gray", linewidth=1, linestyle=":", alpha=0.6, label="z (m)")
    ax_twin.set_ylabel("z (m)", color="gray")
    axes[1, 0].set_ylabel("Absolute error (mm)")
    axes[1, 0].set_title("Estimation error vs distance")
    axes[1, 0].legend(loc="upper left")
    axes[1, 0].grid(True, alpha=0.3)

    # Scatter: error vs distance — should decrease at close range
    axes[1, 1].scatter(z_at_chess[:len(errW_list)], errW_list,
                       color="steelblue", s=20, alpha=0.7, label="|Error W|")
    axes[1, 1].scatter(z_at_chess[:len(errH_list)], errH_list,
                       color="darkorange", s=20, alpha=0.7, label="|Error H|")
    axes[1, 1].set_xlabel("Distance z (m)")
    axes[1, 1].set_ylabel("Absolute error (mm)")
    axes[1, 1].set_title("Error vs distance (scatter)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved: {save_path}")


def plot_stripes(stripe_data, save_path="output/plot_stripes.png"):
    """
    Optional 2 — Per-stripe distance over time.
    Each of the 5 vertical stripes is plotted as a separate line.
    """
    if not stripe_data:
        print("⚠️  No stripe data to plot.")
        return

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    labels = ["Stripe 1 (left)", "Stripe 2", "Stripe 3 (centre)",
              "Stripe 4", "Stripe 5 (right)"]
    frames = [row["frame"] for row in stripe_data]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Optional 2 — Per-stripe distance over time", fontweight="bold")

    for i in range(5):
        values = [row.get(f"z{i}") for row in stripe_data]
        values = [v if v is not None else float("nan") for v in values]
        ax.plot(frames, values, color=colors[i],
                linewidth=1.5, label=labels[i], alpha=0.85)

    ax.axhline(0.8, color="red", linestyle="--", linewidth=1, label="Alarm threshold (0.8 m)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Distance (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved: {save_path}")


def plot_opt3_comparison(frames_base, dist_base, dmain_base,
                          frames_opt3, dist_opt3, dmain_opt3,
                          save_path="output/plot_opt3.png"):
    """
    Optional 3 — Comparison between base SAD and Moravec + uniqueness filtering.
      - Distance over time (both methods)
      - d_main over time (both methods)
      - Distance difference (opt3 - base)
      - d_main histogram comparison
    """
    if not frames_opt3:
        print("⚠️  No opt3 data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Optional 3 — Base SAD vs Moravec + Uniqueness Filtering",
                 fontsize=13, fontweight="bold")

    # Distance over time: base vs opt3
    axes[0, 0].plot(frames_base, dist_base, color="steelblue",
                    linewidth=1.2, alpha=0.7, label="Base SAD")
    axes[0, 0].plot(frames_opt3, dist_opt3, color="darkorange",
                    linewidth=1.5, label="Opt3 (Moravec + Uniqueness)")
    axes[0, 0].axhline(0.8, color="red", linestyle="--", linewidth=1, label="0.8 m")
    axes[0, 0].set_ylabel("Distance (m)")
    axes[0, 0].set_title("Estimated distance — comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # d_main over time: base vs opt3
    axes[0, 1].plot(frames_base, dmain_base, color="steelblue",
                    linewidth=1.2, alpha=0.7, label="Base SAD")
    axes[0, 1].plot(frames_opt3, dmain_opt3, color="darkorange",
                    linewidth=1.5, label="Opt3 (Moravec + Uniqueness)")
    axes[0, 1].set_ylabel("Disparity (px)")
    axes[0, 1].set_title("d_main — comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Distance difference (opt3 - base) on common frames
    common_frames = sorted(set(frames_base) & set(frames_opt3))
    if common_frames:
        base_dict = dict(zip(frames_base, dist_base))
        opt3_dict = dict(zip(frames_opt3, dist_opt3))
        diff = [opt3_dict[f] - base_dict[f] for f in common_frames]

        axes[1, 0].plot(common_frames, diff, color="purple", linewidth=1.2)
        axes[1, 0].axhline(0, color="gray", linestyle="--", linewidth=1)
        axes[1, 0].fill_between(common_frames, diff,
                                 where=[d > 0 for d in diff],
                                 color="darkorange", alpha=0.3, label="Opt3 > Base")
        axes[1, 0].fill_between(common_frames, diff,
                                 where=[d < 0 for d in diff],
                                 color="steelblue", alpha=0.3, label="Opt3 < Base")
        axes[1, 0].set_ylabel("Difference (m)")
        axes[1, 0].set_title("Distance difference (Opt3 − Base)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # d_main histogram: base vs opt3
    axes[1, 1].hist(dmain_base, bins=30, color="steelblue",
                    alpha=0.6, edgecolor="white", label="Base SAD")
    axes[1, 1].hist(dmain_opt3, bins=30, color="darkorange",
                    alpha=0.6, edgecolor="white", label="Opt3")
    axes[1, 1].set_xlabel("d_main (px)")
    axes[1, 1].set_ylabel("Frame count")
    axes[1, 1].set_title("d_main distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved: {save_path}")