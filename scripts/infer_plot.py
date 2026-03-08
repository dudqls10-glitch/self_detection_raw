#!/usr/bin/env python3
"""
간단한 오프라인 추론 + 플롯 스크립트

스크립트 최상단의 설정을 바꾸어 사용하고자 하는 모델, 입력 파일,
저장 위치를 지정합니다.

DATA_DIR: 데이터가 들어있는 폴더
INPUT_FILE: DATA_DIR에 있는 추론할 단일 로그 파일
MODEL_PATH: 학습된 model.pt 경로 또는 모델이 들어있는 폴더 경로
    (폴더를 지정하면 내부에서 최신 .pt 파일을 자동 선택)
NORM_PATH: 정규화 파라미터(.json) 경로 (None이면 체크포인트 안에서 불러옴)
OUTPUT_PLOT: 그림을 저장할 PNG 파일 경로 (비워두면 모델명_번호.png 자동 생성)

실행:
    python scripts/infer_plot.py

플롯은 8개의 서브플롯(raw1..raw8)으로 이루어지며 각 서브플롯에
원래 센서값, 모델 예측값, 잔차(차이)를 표시합니다.
"""

# Add package root to path when script is executed directly
import sys, os
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from self_detection_raw.data.loader import load_file, extract_features
from self_detection_raw.data.stats import load_norm_params
from self_detection_raw.models.mlp_b import ModelB

# ---------------------------------------------------------------------------
# 기본 설정값 (커맨드라인에서 덮어쓸 수 있음)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"
DEFAULT_INPUT_FILE = "1dataset_50_25.txt"      # DATA_DIR 내 파일 이름
DEFAULT_MODEL_PATH = ""  # 빈 문자열이면 자동 검색
DEFAULT_NORM_PATH = None                         # None이면 체크포인트 안에서 로드
DEFAULT_OUTPUT_PLOT = ""  # 빈 문자열이면 자동 생성
DEFAULT_USE_VEL = False
DEFAULT_USE_HARDWARE_BASELINE = True  # True이면 하드웨어 기준값(4e7)을, False이면 y_mean 사용
# ---------------------------------------------------------------------------


def load_model(model_path, norm_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # read configuration if available to reconstruct architecture
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'config.json')
    cfg = {}
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        except Exception:
            pass

    # determine if checkpoint belongs to the tcn-residual model
    sd = checkpoint.get('model_state_dict', {})
    is_tcn = any(k.startswith('main.') for k in sd.keys()) and 'TCN_HIDDEN' in cfg

    if is_tcn:
        from self_detection_raw.models.mlp_tcn_residual import MLP_TCN_ResidualModel
        # infer dims from config or state dict
        in_dim = cfg.get('IN_DIM', 12)
        out_dim = cfg.get('OUT_DIM', 8)
        trunk_hidden = cfg.get('HIDDEN', 128)
        head_hidden = cfg.get('HEAD_HIDDEN', 64)
        tcn_hidden = cfg.get('TCN_HIDDEN', 64)
        tcn_kernel = cfg.get('TCN_KERNEL', 3)
        tcn_dilations = cfg.get('TCN_DILATIONS', [1, 2, 4, 8])
        dropout = cfg.get('DROPOUT', 0.1)

        model = MLP_TCN_ResidualModel(
            in_dim=in_dim,
            out_dim=out_dim,
            trunk_hidden=trunk_hidden,
            head_hidden=head_hidden,
            tcn_hidden=tcn_hidden,
            tcn_kernel=tcn_kernel,
            tcn_dilations=tcn_dilations,
            dropout=dropout,
        ).to(device)
    else:
        # fallback to original ModelB
        model_args = checkpoint.get('args', {})
        # determine input dimension from state dict if possible
        in_dim = None
        # trunk.0.weight is shape (hidden, in_dim)
        key = 'trunk.0.weight'
        if key in sd:
            in_dim = sd[key].shape[1]
        if in_dim is None:
            in_dim = model_args.get('in_dim', 12)

        model = ModelB(
            in_dim=in_dim,
            trunk_hidden=model_args.get('hidden', 128),
            head_hidden=model_args.get('head_hidden', 64),
            out_dim=8,
            dropout=model_args.get('dropout', 0.1)
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # normalization
    if norm_path and os.path.exists(norm_path):
        x_mean, x_std, y_mean, y_std, std_floor = load_norm_params(norm_path)
    elif 'normalization' in checkpoint:
        norm = checkpoint['normalization']
        x_mean = np.array(norm['X_mean'], dtype=np.float32)
        x_std = np.array(norm['X_std'], dtype=np.float32)
        y_mean = np.array(norm['Y_mean'], dtype=np.float32)
        y_std = np.array(norm['Y_std'], dtype=np.float32)
        std_floor = 1e-2
    else:
        # try default file beside model
        model_dir = os.path.dirname(model_path)
        default_norm = os.path.join(model_dir, 'norm_params.json')
        if os.path.exists(default_norm):
            x_mean, x_std, y_mean, y_std, std_floor = load_norm_params(default_norm)
        else:
            # fallback to no normalization (warn user)
            print("[WARNING] Normalization parameters not found; using identity (no norm).\n"
                  "Results may be incorrect. Provide --norm or put norm_params.json next to model.")
            # infer dimension from model or default to 12
            in_dim = model.trunk[0].weight.shape[1]
            out_dim = model.out_dim
            x_mean = np.zeros(in_dim, dtype=np.float32)
            x_std = np.ones(in_dim, dtype=np.float32)
            y_mean = np.zeros(out_dim, dtype=np.float32)
            y_std = np.ones(out_dim, dtype=np.float32)
            std_floor = 1e-2

    # indicate if the architecture was tcn-residual
    return model, x_mean, x_std, y_mean, y_std, is_tcn


def infer_and_plot(
    data_dir,
    input_file,
    model_path,
    norm_path,
    output_plot,
    use_vel,
    use_hardware_baseline,
):
    # 경로 정리
    data_path = os.path.join(data_dir, input_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"입력 파일이 없습니다: {data_path}")

    # if model path not specified treat it as unspecified; later we will search in default location
    mdl_path = model_path

    # when no path provided, try reading last_run.json from same place where training stored it
    if not mdl_path:
        try:
            cfg_path = os.path.join(os.path.dirname(__file__), 'model', 'last_run.json')
            if os.path.exists(cfg_path):
                import json
                with open(cfg_path, 'r') as f:
                    data = json.load(f)
                    mdl_path = data.get('run_dir')
                    if mdl_path and os.path.isdir(mdl_path):
                        # search inside this directory for newest .pt
                        candidates = []
                        for root, dirs, files in os.walk(mdl_path):
                            for fn in files:
                                if fn.endswith('.pt'):
                                    candidates.append(os.path.join(root, fn))
                        if candidates:
                            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                            mdl_path = candidates[0]
                            print(f"[INFO] loaded last run model: {mdl_path}")
                        else:
                            mdl_path = None
        except Exception:
            mdl_path = None

    # if the given path is a directory, search for .pt files inside it
    if mdl_path and os.path.isdir(mdl_path):
        candidates = []
        for root, dirs, files in os.walk(mdl_path):
            for fn in files:
                if fn.endswith('.pt'):
                    candidates.append(os.path.join(root, fn))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            mdl_path = candidates[0]
            print(f"[INFO] using model from directory {model_path}: {mdl_path}")
        else:
            raise FileNotFoundError(f"모델 폴더 '{model_path}' 안에 .pt 파일이 없습니다.")

    # if still unset or not exists, look in default scripts/model folder
    if not mdl_path or not os.path.exists(mdl_path):
        base = os.path.join(os.path.dirname(__file__), 'model')
        candidates = []
        if os.path.isdir(base):
            for root, dirs, files in os.walk(base):
                for fn in files:
                    if fn.endswith('.pt'):
                        candidates.append(os.path.join(root, fn))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            mdl_path = candidates[0]
            print(f"[INFO] auto-detected model: {mdl_path}")
        else:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다. MODEL_PATH를 설정하세요.")
    model, x_mean, x_std, y_mean, y_std, is_tcn = load_model(mdl_path, norm_path)

    # determine output plot path if not set
    plot_path = output_plot
    if not plot_path:
        # base name from model filename (without extension)
        model_base = Path(mdl_path).stem
        # leading digit(s) from input filename
        num = ''
        m = re.match(r'^(\d+)', os.path.basename(input_file))
        if m:
            num = '_' + m.group(1)
        plot_path = os.path.join(os.path.dirname(mdl_path), f"{model_base}{num}.png")
        print(f"[INFO] output plot path set to {plot_path}")

    # 데이터 읽기/전처리
    data = load_file(data_path)  # (N,37)
    X, Y = extract_features(data, use_vel=use_vel)
    X_norm = (X - x_mean) / x_std

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_norm.astype(np.float32))
        # move input to model device
        X_tensor = X_tensor.to(next(model.parameters()).device)
        # if TCN model expect sequence dim
        if is_tcn and X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)
        out = model(X_tensor)
        # model may return tuple (y_hat, y_res)
        if isinstance(out, tuple):
            pred_norm = out[0].cpu().numpy()
        else:
            pred_norm = out.cpu().numpy()
    pred = pred_norm * y_std + y_mean
    residual = Y - pred

    # compensated 계산 (realtime_infer과 동일한 방식)
    HARDWARE_BASELINE = 4.0e+07
    if use_hardware_baseline:
        baseline = np.full(Y.shape[1], HARDWARE_BASELINE, dtype=np.float32)
    else:
        baseline = y_mean
    compensated = residual + baseline

    # 플롯 그리기
    n_channels = Y.shape[1]
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    t = np.arange(Y.shape[0])
    for i in range(n_channels):
        ax = axes[i]
        ax.set_title(f'raw{i+1}')
        # draw raw, pred, compensated on same axis with layering and transparency
        ax.plot(t, Y[:, i], label='raw', color='C0', alpha=0.3, zorder=1)
        ax.plot(t, pred[:, i], label='pred', color='C1', alpha=0.5, zorder=2)
        ax.plot(t, compensated[:, i], label='compensated', color='C2', alpha=0.3, zorder=3)
        if i == 0:
            ax.legend(loc='upper right')
    fig.suptitle(f'Inference results on {input_file}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir = Path(plot_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # 간단한 통계 출력
    print("Compensated stats (mean, std) for each channel:")
    for i in range(n_channels):
        print(f" raw{i+1}: {compensated[:, i].mean():.3f}, {compensated[:, i].std():.3f}")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Offline inference and plotting')
    p.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='dataset directory')
    p.add_argument('--input-file', default=DEFAULT_INPUT_FILE, help='file name to infer')
    p.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='path to model.pt')
    p.add_argument('--norm-path', default=DEFAULT_NORM_PATH, help='normalization file (.json)')
    p.add_argument('--output-plot', default=DEFAULT_OUTPUT_PLOT, help='path to output png')
    p.add_argument('--use-vel', action='store_true', help='include velocity features')
    p.add_argument('--use-hw-baseline', action='store_true', help='apply hardware baseline')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    infer_and_plot(
        data_dir=args.data_dir,
        input_file=args.input_file,
        model_path=args.model_path,
        norm_path=args.norm_path,
        output_plot=args.output_plot,
        use_vel=args.use_vel,
        use_hardware_baseline=args.use_hw_baseline,
    )
