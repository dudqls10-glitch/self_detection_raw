#!/usr/bin/env python3
"""
Joint Velocity Smoothing 확인용 스크립트

데이터 파일의 Joint Velocity(jv1~jv6)를 읽어와서,
원본 값과 이동평균 필터를 거친 값을 비교하여 그래프로 저장합니다.

실행:
    python scripts/plot_velocity.py --file dataset/2dataset_50_25.txt --window 10
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI가 없는 환경을 위한 설정
import matplotlib.pyplot as plt
from scipy.signal import savgol_coeffs, lfilter

# 패키지 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from self_detection_raw.data.loader import load_file


# def smooth_data(data, window_size=5):
#     """
#     1D 이동 평균 필터 (loader_v.py와 동일한 로직)
#     """
#     if window_size <= 1:
#         return data
#     ...

def smooth_data_causal_savgol(data, window_length=21, polyorder=2):
    """
    Causal Savitzky-Golay 필터를 사용하여 데이터 스무딩.
    실시간 적용을 가정하여, 현재 값은 과거의 window_length개 샘플을 사용해 추정합니다.
    """
    if window_length <= polyorder:
        window_length = polyorder + 1 + (polyorder % 2) # 홀수이고 polyorder보다 크게

    # 필터 윈도우의 '끝' 지점에서 값을 추정하는 계수를 계산하여 인과성을 만족시킴
    coeffs = savgol_coeffs(window_length, polyorder, pos=window_length - 1)

    smoothed = np.zeros_like(data)
    for i in range(data.shape[1]):
        # lfilter는 인과 FIR 필터를 적용
        smoothed[:, i] = lfilter(coeffs, 1.0, data[:, i])
    return smoothed

def main():
    parser = argparse.ArgumentParser(description='Plot Joint Velocity with Smoothing')
    parser.add_argument('--file', type=str, default='2dataset_50_25.txt', help='File name in dataset folder or full path')
    parser.add_argument('--window', type=int, default=21, help='Savitzky-Golay filter window size')
    parser.add_argument('--polyorder', type=int, default=2, help='Savitzky-Golay filter polynomial order')
    parser.add_argument('--data-dir', type=str, default='/home/song/rb10_Proximity/src/self_detection_raw/dataset')
    args = parser.parse_args()

    # 파일 경로 찾기
    file_path = args.file
    if not os.path.exists(file_path):
        # data_dir에서 찾기
        candidate = os.path.join(args.data_dir, args.file)
        if os.path.exists(candidate):
            file_path = candidate
        else:
            print(f"[ERROR] File not found: {args.file}")
            return

    print(f"Loading {file_path}...")
    data = load_file(file_path)
    
    # Velocity 추출 (컬럼 7~12: jv1~jv6)
    # data structure: timestamp(0), j1..j6(1..6), jv1..jv6(7..12)
    jv_raw = data[:, 7:13]
    
    print(f"Applying Causal Savitzky-Golay filter (window={args.window}, polyorder={args.polyorder})...")
    jv_smooth = smooth_data_causal_savgol(jv_raw, window_length=args.window, polyorder=args.polyorder)
    
    # Plot
    print("Generating plot...")
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    if not isinstance(axes, np.ndarray): axes = [axes]
    
    t = np.arange(len(jv_raw))
    
    for i in range(6):
        ax = axes[i]
        ax.plot(t, jv_raw[:, i], label='Raw', color='lightgray', alpha=0.8)
        ax.plot(t, jv_smooth[:, i], label=f'Causal S-G (w={args.window}, p={args.polyorder})', color='blue', linewidth=1.5)
        ax.set_ylabel(f'JV {i+1} (deg/s)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('Sample Index')
    fig.suptitle(f'Joint Velocity Smoothing Check (Causal Savitzky-Golay)\nFile: {os.path.basename(file_path)} | Window: {args.window}, Polyorder: {args.polyorder}', fontsize=14)
    plt.tight_layout()
    
    # 스크립트가 있는 폴더(scripts/)에 저장
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'velocity_savgol_causal_check.png')
    plt.savefig(out_file)
    plt.close()
    print(f"Plot saved to {os.path.abspath(out_file)}")

if __name__ == '__main__':
    main()
