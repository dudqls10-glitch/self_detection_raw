#!/usr/bin/env python3
"""
MLP 학습 스크립트 (Velocity 포함 버전)

기존 train_mlp.py를 기반으로 하되,
1. Joint Velocity를 입력으로 사용합니다 (USE_VEL = True).
2. Velocity 데이터의 노이즈를 줄이기 위해 Smoothing을 적용합니다 (VEL_WINDOW).

실행:
    python scripts/train_mlp_v.py
"""

import os
import sys
import random
from pathlib import Path
from datetime import datetime
import argparse

# make sure package root is on sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 새로 만든 모듈 임포트 (_v)
from self_detection_raw.data.loader_v import load_multiple_files_v
from self_detection_raw.models.mlp_b_v import ModelBV

from self_detection_raw.train.train import SelfDetectionDataset, train_epoch, evaluate
from self_detection_raw.utils.io import ensure_dir
from self_detection_raw.utils.metrics import format_metrics_report

# ---------------------------------------------------------------------------
# 기본 설정값
# ---------------------------------------------------------------------------
DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"

TRAIN_FILES = [
    "2dataset_50_25.txt",
    "3dataset_100_25.txt",
    "4dataset_100_25.txt",
    "5dataset_50_50.txt",
    "6dataset_50_50.txt",
    "7dataset_100_50.txt",
    "8dataset_100_50.txt",
]

VAL_FILES = [
    "1dataset_50_25.txt",
]

script_root = Path(__file__).parent.absolute()
OUT_DIR = str(script_root / "model")
NAME_PREFIX = "mlp_vel"

# 학습 하이퍼파라미터
USE_VEL = True       # 속도 사용 활성화
VEL_WINDOW = 10      # 속도 데이터 스무딩 윈도우 크기 (클수록 부드러워짐)

X_NORM = True
Y_NORM = True
STD_FLOOR = 1e-2
EPOCHS = 300
BATCH = 256
LR = 1e-3
WD = 1e-4
HIDDEN = 128
HEAD_HIDDEN = 64
DROPOUT = 0.1
SEED = 42
NUM_WORKERS = 4
# ---------------------------------------------------------------------------

def full_paths(names, base_dir):
    paths = []
    for fn in names:
        p = Path(fn)
        if not p.is_absolute():
            paths.append(str(Path(base_dir) / fn))
        else:
            paths.append(str(p))
    return paths

def main():
    parser = argparse.ArgumentParser(description='Train MLP with Smoothed Velocity')
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--train-files', default=','.join(TRAIN_FILES))
    parser.add_argument('--val-files', default=','.join(VAL_FILES))
    parser.add_argument('--out-dir', default=OUT_DIR)
    parser.add_argument('--name-prefix', default=NAME_PREFIX)
    
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--wd', type=float, default=WD)
    parser.add_argument('--hidden', type=int, default=HIDDEN)
    parser.add_argument('--head-hidden', type=int, default=HEAD_HIDDEN)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--seed', type=int, default=SEED)
    
    # Velocity 관련 파라미터
    parser.add_argument('--vel-window', type=int, default=VEL_WINDOW, help='Smoothing window size for velocity')
    
    args = parser.parse_args()

    # 변수 설정
    data_dir = args.data_dir
    train_files = args.train_files.split(',') if args.train_files else []
    val_files = args.val_files.split(',') if args.val_files else []
    out_dir = args.out_dir
    name_prefix = args.name_prefix
    vel_window = args.vel_window

    # 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_paths = full_paths(train_files, data_dir)
    val_paths = full_paths(val_files, data_dir)

    # 데이터 로드 (loader_v 사용)
    print(f"\n데이터 로딩 중... (Velocity Smoothing Window: {vel_window})")
    # USE_VEL=True 강제
    X_train, Y_train = load_multiple_files_v(train_paths, use_vel=True, vel_window=vel_window)
    
    if val_paths:
        X_val, Y_val = load_multiple_files_v(val_paths, use_vel=True, vel_window=vel_window)
    else:
        X_val, Y_val = np.array([]).reshape(0, X_train.shape[1]), np.array([]).reshape(0, Y_train.shape[1])

    print(f"Train Input Shape: {X_train.shape} (Should be N x 18)")

    # Dataset 생성
    train_ds = SelfDetectionDataset(
        X_train, Y_train,
        std_floor=STD_FLOOR,
    )
    val_ds = SelfDetectionDataset(
        X_val, Y_val,
        X_mean=train_ds.X_mean,
        X_std=train_ds.X_std,
        Y_mean=train_ds.Y_mean,
        Y_std=train_ds.Y_std,
        std_floor=STD_FLOOR,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # 모델 생성 (ModelBV 사용)
    # 입력 차원: sin(6) + cos(6) + vel(6) = 18
    model = ModelBV(
        in_dim=X_train.shape[1], 
        trunk_hidden=args.hidden,
        head_hidden=args.head_hidden,
        out_dim=Y_train.shape[1],
        dropout=args.dropout,
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.SmoothL1Loss()

    try:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=True)
    except Exception:
        scheduler = None

    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    run_dir = ensure_dir(os.path.join(out_dir, f"{name_prefix}_{timestamp}"))

    best_val_std = float('inf')
    history = {'train_loss': [], 'val_avg_std': []}

    print(f"\n학습 시작...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, val_ds, device) if len(val_paths) > 0 else {}
        
        if scheduler and 'avg_std' in metrics:
            scheduler.step(metrics['avg_std'])

        history['train_loss'].append(loss)
        if 'avg_std' in metrics:
            history['val_avg_std'].append(metrics['avg_std'])

        if 'avg_std' in metrics and metrics['avg_std'] < best_val_std:
            best_val_std = metrics['avg_std']
            norm_params = train_ds.get_norm_params()
            fname = "model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': metrics,
                'normalization': norm_params,
                'args': vars(args), # 나중에 추론할 때 파라미터 확인용
            }, Path(run_dir) / fname)
            print(f"Saved best model to {Path(run_dir)/fname} (STD: {best_val_std:.2f})")

        if epoch % 10 == 0 or epoch == 1:
            msg = f"Epoch {epoch}/{args.epochs} | Train loss: {loss:.6f}"
            if 'avg_std' in metrics:
                msg += f" | Val std: {metrics['avg_std']:.2f}"
            print(msg)

    # 설정 저장
    from self_detection_raw.utils.io import save_json
    cfg = vars(args)
    save_json(cfg, str(Path(run_dir) / 'config.json'))

    print(f"Finished training, best val std: {best_val_std}")
    print(f"Run directory: {run_dir}")

if __name__ == '__main__':
    main()