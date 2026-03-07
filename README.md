# Stereo Robot Navigation

A Python-based stereo vision system for autonomous robot navigation. Given synchronized video streams from a calibrated stereo camera, the system computes disparity maps using the SAD (Sum of Absolute Differences) algorithm to estimate obstacle distances in real time.

## Features
- Dense disparity map computation via SAD stereo matching
- Real-time obstacle distance estimation with alarm below a threshold
- Chessboard pattern detection and real-world dimension estimation
- Optional 1: Dynamic disparity range for faster and more robust matching
- Optional 2: Multi-stripe planar obstacle analysis with top-down view and angle estimation
- Optional 3: Robust disparity estimation via Moravec interest operator and uniqueness filtering
- CSV logging and matplotlib plots for post-run analysis
- Output video with overlaid distance, alarm, and disparity map

## Requirements
Python 3.11, OpenCV, NumPy, Matplotlib

## Usage
python main.py
