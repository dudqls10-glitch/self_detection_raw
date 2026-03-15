import numpy as np
import os
from scipy.signal import savgol_coeffs, lfilter
from .loader import load_file as original_load_file

def smooth_data(data, window_size=5, polyorder=2):
    """
    Causal Savitzky-Golay 필터를 사용하여 데이터 스무딩.
    실시간 적용을 가정하여, 현재 값은 과거의 window_size개 샘플을 사용해 추정합니다.
    """
    if window_size <= 1:
        return data
    
    # Savitzky-Golay requires window_size to be odd and > polyorder
    if window_size % 2 == 0:
        window_size += 1
    
    if window_size <= polyorder:
        window_size = polyorder + 1 + (polyorder % 2)

    # 필터 윈도우의 '끝' 지점에서 값을 추정하는 계수를 계산하여 인과성을 만족시킴
    try:
        coeffs = savgol_coeffs(window_size, polyorder, pos=window_size - 1)
    except ValueError:
        # Fallback if parameters are invalid
        return data

    smoothed = np.zeros_like(data)
    
    for i in range(data.shape[1]):
        col = data[:, i]
        # 초기값 0으로 인한 과도응답을 줄이기 위해 앞부분을 첫 번째 값으로 패딩
        pad_width = window_size - 1
        padded = np.pad(col, (pad_width, 0), mode='edge')
        
        # lfilter 적용 (FIR 필터)
        filtered = lfilter(coeffs, 1.0, padded)
        
        # 패딩 제거
        smoothed[:, i] = filtered[pad_width:]
        
    return smoothed

def load_multiple_files_v(filepaths, use_vel=True, vel_window=10):
    """
    여러 파일을 로드하고 속도 데이터 정제 후 합칩니다.
    """
    X_list = []
    Y_list = []
    
    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"[WARN] File not found: {fp}")
            continue
            
        data = original_load_file(fp)
        X, Y = extract_features_v(data, use_vel=use_vel, vel_window=vel_window)
        X_list.append(X)
        Y_list.append(Y)
        
    if not X_list:
        return np.array([]), np.array([])
        
    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)

def extract_features_v(data, use_vel=True, vel_window=10):
    """
    데이터에서 Feature(X)와 Target(Y)을 추출합니다.
    use_vel=True일 경우 속도 데이터에 스무딩을 적용합니다.
    """
    # data columns:
    # 0: timestamp
    # 1~6: joint pos (deg)
    # 7~12: joint vel (deg/s)
    # 13~20: prox
    # 21~28: raw (target)
    # 29~36: tof

    # 1. Joint Position -> Sin/Cos Encoding
    joint_pos_deg = data[:, 1:7]
    joint_pos_rad = np.deg2rad(joint_pos_deg)
    sin_j = np.sin(joint_pos_rad)
    cos_j = np.cos(joint_pos_rad)
    
    features = [sin_j, cos_j]
    
    # 2. Joint Velocity (Optional with Smoothing)
    if use_vel:
        joint_vel = data[:, 7:13]
        # 속도 데이터 정제 (Causal Savitzky-Golay)
        # polyorder는 기본값 2 사용
        joint_vel_smooth = smooth_data(joint_vel, window_size=vel_window, polyorder=2)
        features.append(joint_vel_smooth)
        
    X = np.concatenate(features, axis=1) # (N, 12) or (N, 18)
    
    # 3. Target (Raw)
    Y = data[:, 21:29]
    
    return X, Y