#!/usr/bin/env python3
"""
Velocity 포함 모델(ModelBV) 추론 및 시각화 스크립트

train_mlp_v.py로 학습된 모델을 불러와서,
테스트 데이터에 대한 예측 결과(Target vs Pred)와 잔차(Residual)를 그래프로 그립니다.

실행:
    python scripts/infer_plot_v.py --model model/mlp_vel_MMDD_HHMMSS/model.pt --data dataset/1dataset_50_25.txt
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 창 띄우기 방지 (서버/터미널 환경용)
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 패키지 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from self_detection_raw.data.loader import load_file
from self_detection_raw.data.loader_v import extract_features_v
from self_detection_raw.models.mlp_b_v import ModelBV

# ---------------------------------------------------------------------------
# 기본 설정값 (스크립트 상단에서 수정 가능)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"
DEFAULT_INPUT_FILE = "8dataset_50_25.txt"  # DATA_DIR 내 파일 이름. None이면 최신 파일 자동 검색
# ---------------------------------------------------------------------------

def find_latest_model():
    """Find the latest model.pt in the scripts/model directory."""
    # Assuming this script is in scripts/
    script_dir = Path(__file__).parent
    model_dir = script_dir / 'model'
    
    if not model_dir.exists():
        return None
        
    files = list(model_dir.glob('**/model.pt'))
    if not files:
        return None
        
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0])

def find_latest_data(base_dir=None):
    """Find the latest data file in the dataset directory."""
    if base_dir and os.path.exists(base_dir):
        dataset_dir = Path(base_dir)
    else:
        # Assuming this script is in scripts/ and dataset is in dataset/
        script_dir = Path(__file__).parent
        # Try looking in ../dataset relative to script
        dataset_dir = script_dir.parent / 'dataset'
    
    if not dataset_dir.exists():
        # Fallback: try looking in current dir/dataset
        dataset_dir = Path('dataset')
    
    if not dataset_dir.exists():
        return None
        
    files = list(dataset_dir.glob('*.txt'))
    if not files:
        return None
        
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0])

def main():
    parser = argparse.ArgumentParser(description='Inference and Plot for Velocity Model')
    parser.add_argument('--model', type=str, default=None, help='Path to model.pt (default: latest in model/)')
    parser.add_argument('--data', type=str, default=None, help='Path to data file (.txt) (default: latest in dataset/)')
    parser.add_argument('--norm-path', type=str, default=None, help='Path to normalization params (.json). If None, tries to load from checkpoint.')
    parser.add_argument('--output-plot', type=str, default=None, help='Path to save the output plot. If None, saves in model directory.')
    parser.add_argument('--vel-window', type=int, default=10, help='Velocity smoothing window (default: 10, overridden by checkpoint if available)')
    parser.add_argument('--use-hw-baseline', dest='use_hardware_baseline', action='store_true', help='Use hardware baseline (4e+7) for compensation (default)')
    parser.add_argument('--use-mean-baseline', dest='use_hardware_baseline', action='store_false', help='Use training data mean as baseline')
    parser.set_defaults(use_hardware_baseline=True)
    args = parser.parse_args()

    print("Starting inference script...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 모델 및 설정 로드
    if args.model is None:
        args.model = find_latest_model()
        if args.model is None:
            print("Error: No model found in scripts/model/ and --model not specified.")
            return
        print(f"Auto-selected latest model: {args.model}")

    if not os.path.exists(args.model):
        print(f"Error: Model file not found {args.model}")
        return

    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # 2. 데이터 파일 자동 선택
    if args.data is None:
        # 상단 변수에 지정된 파일이 있으면 우선 사용
        if DEFAULT_INPUT_FILE:
            if os.path.isabs(DEFAULT_INPUT_FILE):
                candidate = DEFAULT_INPUT_FILE
            else:
                candidate = os.path.join(DEFAULT_DATA_DIR, DEFAULT_INPUT_FILE)
            
            if os.path.exists(candidate):
                args.data = candidate
        
        # 여전히 없으면 자동 검색
        if args.data is None:
            args.data = find_latest_data(DEFAULT_DATA_DIR)
            
        if args.data is None:
            print("Error: No data file found in dataset/ and --data not specified.")
            return
        print(f"Selected data: {args.data}")

    # 학습 당시 설정 확인 (저장된 경우)
    train_args = checkpoint.get('args', {})
    vel_window = train_args.get('vel_window', args.vel_window)
    print(f"Using Velocity Window: {vel_window}")

    # Normalization 파라미터 로드
    norm_params = None
    if args.norm_path and os.path.exists(args.norm_path):
        print(f"Loading normalization params from specified file: {args.norm_path}")
        with open(args.norm_path, 'r') as f:
            norm_params = json.load(f)
    elif 'normalization' in checkpoint:
        print("Normalization params loaded from checkpoint.")
        norm_params = checkpoint.get('normalization')
    else:
        print("Warning: Normalization params not found in checkpoint or --norm-path. Looking for default json file...")
        # json 파일 찾아보기
        model_dir = os.path.dirname(args.model)
        norm_path = os.path.join(model_dir, 'norm_params.json')
        if os.path.exists(norm_path):
            print(f"Found and loaded default norm file: {norm_path}")
            with open(norm_path, 'r') as f:
                norm_params = json.load(f)
        else:
            print("Error: Normalization params not found in checkpoint or json.")
            return
        
    if norm_params is None:
        print("Error: Normalization params could not be loaded. Please specify --norm-path or ensure it's in the checkpoint/model directory.")
        return

    X_mean = np.array(norm_params['X_mean'])
    X_std = np.array(norm_params['X_std'])
    Y_mean = np.array(norm_params['Y_mean'])
    Y_std = np.array(norm_params['Y_std'])

    if not os.path.exists(args.data):
        # dataset 폴더에서 찾아보기
        dataset_dir = os.path.join(os.path.dirname(__file__), '../dataset')
        alt_path = os.path.join(dataset_dir, args.data)
        if os.path.exists(alt_path):
            args.data = alt_path
        else:
            print(f"Error: Data file not found {args.data}")
            return

    print(f"Loading data from {args.data}...")
    raw_data = load_file(args.data)
    
    # Feature Extraction (Velocity Smoothing 포함)
    # extract_features_v returns X, Y
    X_raw, Y_raw = extract_features_v(raw_data, use_vel=True, vel_window=vel_window)
    
    # Normalize X
    X_norm = (X_raw - X_mean) / X_std
    
    # Tensor conversion
    X_tensor = torch.FloatTensor(X_norm).to(device)

    # 3. 모델 초기화 및 추론
    # 입력 차원 확인 (18 예상)
    in_dim = X_raw.shape[1]
    out_dim = Y_raw.shape[1]
    
    model = ModelBV(in_dim=in_dim, out_dim=out_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Running inference...")
    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy()

    # Denormalize Prediction
    pred = pred_norm * Y_std + Y_mean
    
    # Residual
    residual = Y_raw - pred

    # Compensated calculation
    HARDWARE_BASELINE = 4.0e+07
    if args.use_hardware_baseline:
        baseline = np.full(Y_raw.shape[1], HARDWARE_BASELINE, dtype=np.float32)
    else:
        baseline = Y_mean
    compensated = residual + baseline

    # 4. 시각화
    print("Plotting results...")
    # 8개 채널에 대해 4x2 Plot
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    t = np.arange(len(Y_raw))
    
    for i in range(8):
        ax = axes[i]
        ax_comp = ax.twinx()

        # left axis: raw and prediction
        raw_line = ax.plot(t, Y_raw[:, i], label='raw', color='C0', alpha=0.3, zorder=1)
        pred_line = ax.plot(t, pred[:, i], label='pred', color='C1', alpha=0.5, zorder=2)

        # right axis: compensated only
        comp_line = ax_comp.plot(t, compensated[:, i], label='compensated', color='C2', alpha=0.4, zorder=3)
        ax_comp.tick_params(axis='y', colors='C2')
        ax_comp.spines['right'].set_color('C2')
        
        resid_std = np.std(residual[:, i])
        comp_std = np.std(compensated[:, i])
        ax.set_title(f'raw{i+1} | ResStd: {resid_std:.0f} | CompStd: {comp_std:.0f}', fontsize=10)
        if i == 0:
            lines = raw_line + pred_line + comp_line
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Inference results on {os.path.basename(args.data)}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(os.path.dirname(args.model), 'inference_plot_v.png')
    if args.output_plot:
        out_path = args.output_plot
    else:
        out_path = os.path.join(os.path.dirname(args.model), 'inference_plot_v.png')
    
    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving plot to {out_path}...")
    plt.savefig(out_path)
    plt.close()

    if os.path.exists(out_path):
        print("\n" + "="*60)
        print(f"SUCCESS: Plot generated successfully!")
        print(f"Location: {os.path.abspath(out_path)}")
        print("="*60 + "\n")
    else:
        print(f"Error: Failed to save plot to {out_path}")

    # 간단한 통계 출력
    print("Compensated stats (mean, std) for each channel:")
    for i in range(8):
        print(f" raw{i+1}: {compensated[:, i].mean():.3f}, {compensated[:, i].std():.3f}")

if __name__ == '__main__':
    main()
