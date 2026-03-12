#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Detection Compensation Training Script V3

학습 방법: train_self_detection.py 방식 (단순 MSE, 랜덤 분할)
모델 구조: V3 Per-Channel Head (더 깊고 넓은 Head)

입력: sin(pos) + cos(pos) + vel = 18차원 (sin/cos 변환)
출력: proximity 센서 값 예측 (14채널, 5번 센서 임시 제외)
Train/Val 분할: 파일 단위 시간 순서 유지 (temporal split)

사용법:
  python train_self_detection_v3.py --data_dir /path/to/csv --epochs 200

저자: Claude
날짜: 2026-01-29
"""

import os
import glob
import argparse
import random
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# 설정 및 상수
# ============================================================
BASE_VALUE = 4e7  # Proximity 센서 baseline (장애물 없을 때 값)

# 컬럼 이름 정의
JOINT_POS_COLS = [f"joint_pos_{i}" for i in range(6)]
JOINT_VEL_COLS = [f"joint_vel_{i}" for i in range(6)]
INPUT_COLS = JOINT_POS_COLS + JOINT_VEL_COLS  # 12차원 입력

# Proximity 센서 컬럼 - 순서 중요!
# ★ 임시: 5번 센서(prox_51, prox_52) 제외 ★
PROX_COLS = [
    'prox_11', 'prox_12', 'prox_21', 'prox_22',
    'prox_31', 'prox_32', 'prox_41', 'prox_42',
    'prox_51', 'prox_52',
    'prox_61', 'prox_62',
    'prox_71', 'prox_72', 'prox_81', 'prox_82'
]

# 채널 수
NUM_CHANNELS = 16


# ============================================================
# 데이터 전처리 함수
# ============================================================
def load_and_preprocess_csv(file_path: str, filter_zero_values: bool = True) -> Optional[pd.DataFrame]:
    """CSV 파일 로드 및 전처리

    Args:
        file_path: CSV 파일 경로
        filter_zero_values: True이면 센서값이 0인 행 제거 (센서 오류)
    """
    df = pd.read_csv(file_path)

    required_cols = INPUT_COLS + PROX_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Warning: Missing columns in {file_path}: {missing_cols[:5]}...")
        return None

    # 센서 이상치 필터링: 0값 제거 (센서 오류)
    if filter_zero_values:
        original_len = len(df)
        mask = np.ones(len(df), dtype=bool)

        for col in PROX_COLS:
            mask &= (df[col].values > 0)

        df = df[mask].reset_index(drop=True)
        filtered_count = original_len - len(df)

        if filtered_count > 0:
            print(f"  Filtered {filtered_count} zero-value rows ({100*filtered_count/original_len:.2f}%) from {os.path.basename(file_path)}")

    return df


def load_all_files(data_dir: str) -> List[Tuple[str, pd.DataFrame]]:
    """모든 Self_Detection CSV 파일을 개별적으로 로드 (파일 단위 유지)"""
    pattern = os.path.join(data_dir, "*Self_Detection*.csv")
    files = sorted(glob.glob(pattern))

    # 하위 디렉토리도 검색
    pattern_sub = os.path.join(data_dir, "**/*Self_Detection*.csv")
    files_sub = sorted(glob.glob(pattern_sub, recursive=True))
    files = sorted(set(files + files_sub))

    if len(files) == 0:
        raise RuntimeError(f"No CSV files found matching: {pattern}")

    print(f"Found {len(files)} CSV files")

    file_dfs = []
    for f in files:
        df = load_and_preprocess_csv(f)
        if df is not None:
            file_dfs.append((os.path.basename(f), df))
            print(f"  Loaded: {os.path.basename(f)} ({len(df)} samples)")

    total_samples = sum(len(df) for _, df in file_dfs)
    print(f"\nTotal samples: {total_samples}")

    return file_dfs


def split_files_temporal(file_dfs: List[Tuple[str, pd.DataFrame]],
                         val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    파일 단위 시간 순서 유지 Train/Val 분할

    각 파일 내에서 앞부분 train, 뒷부분 val (시간 순서 유지)
    """
    train_dfs = []
    val_dfs = []

    for fname, df in file_dfs:
        n = len(df)
        n_train = int(n * (1 - val_ratio))

        train_dfs.append(df.iloc[:n_train].copy())
        val_dfs.append(df.iloc[n_train:].copy())

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = pd.concat(val_dfs, ignore_index=True)

    return df_train, df_val


# ============================================================
# sin/cos 피처 변환 함수
# ============================================================
def transform_joint_features(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """
    관절 각도를 sin/cos로 변환하여 주기성 반영

    Args:
        pos: (N, 6) 관절 위치 (라디안)
        vel: (N, 6) 관절 속도

    Returns:
        (N, 18) 변환된 피처 [sin(pos), cos(pos), vel]
    """
    sin_pos = np.sin(pos)
    cos_pos = np.cos(pos)
    return np.concatenate([sin_pos, cos_pos, vel], axis=1)


# ============================================================
# PyTorch Dataset
# ============================================================
class SelfDetectionDatasetV3(Dataset):
    """
    Self-Detection Dataset V3

    입력: 관절 위치 + 속도 (12차원)
    출력: proximity 센서 값
    정규화: (Y - BASE_VALUE) / 1e6
    """

    def __init__(self, df: pd.DataFrame,
                 X_mean: Optional[np.ndarray] = None,
                 X_std: Optional[np.ndarray] = None):
        # 입력 데이터: sin/cos 변환 적용 (18차원)
        pos_data = df[JOINT_POS_COLS].to_numpy(dtype=np.float32)
        vel_data = df[JOINT_VEL_COLS].to_numpy(dtype=np.float32)
        self.X = transform_joint_features(pos_data, vel_data)
        self.input_dim = self.X.shape[1]  # 18 = sin(6) + cos(6) + vel(6)

        # 출력 데이터 (센서 값)
        self.Y_raw = df[PROX_COLS].to_numpy(dtype=np.float32)

        # 입력 정규화
        if X_mean is None:
            self.X_mean = np.mean(self.X, axis=0)
            self.X_std = np.std(self.X, axis=0) + 1e-8
        else:
            self.X_mean = X_mean
            self.X_std = X_std

        self.X_normalized = (self.X - self.X_mean) / self.X_std

        # 출력 정규화: (값 - baseline) / scale
        self.Y_baseline = BASE_VALUE
        self.Y_scale = 1e6
        self.Y = (self.Y_raw - self.Y_baseline) / self.Y_scale

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_normalized[idx]),
            torch.from_numpy(self.Y[idx].astype(np.float32))
        )

    def get_normalization_params(self):
        """모델 저장 시 정규화 파라미터도 저장"""
        return {
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'Y_baseline': self.Y_baseline,
            'Y_scale': self.Y_scale,
            'input_dim': self.input_dim,
        }


# ============================================================
# MLP 모델
# ============================================================
class SelfDetectionMLPV3(nn.Module):
    """
    Self-Detection MLP V3

    특징:
      - Layer normalization
      - GELU activation
      - Xavier 초기화
    """

    def __init__(self, in_dim=12, hidden_dims=(512, 256, 256, 128),
                 out_dim=14, dropout=0.1):
        super().__init__()

        layers = [
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        ]

        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], out_dim)
        self.out_dim = out_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        return self.output_layer(features)


class SelfDetectionMLPPerChannelV3(nn.Module):
    """
    채널별 독립 Head를 가진 MLP V3

    각 센서 채널마다 별도의 예측 head를 사용하여
    채널 간 간섭 최소화
    """

    def __init__(self, in_dim=12, trunk_dims=(512, 256),
                 head_dims=(128, 64, 32), out_dim=14, dropout=0.1):
        super().__init__()

        # Shared trunk
        trunk_layers = []
        d = in_dim
        for h in trunk_dims:
            trunk_layers.extend([
                nn.Linear(d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            d = h
        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_out_dim = trunk_dims[-1]

        # Per-channel heads (128->64->32->1)
        self.heads = nn.ModuleList()
        for _ in range(out_dim):
            head_layers = []
            d = self.trunk_out_dim
            for h in head_dims:
                head_layers.extend([
                    nn.Linear(d, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                ])
                d = h
            head_layers.append(nn.Linear(d, 1))
            self.heads.append(nn.Sequential(*head_layers))

        self.out_dim = out_dim

    def forward(self, x):
        z = self.trunk(x)
        outputs = [head(z) for head in self.heads]
        return torch.cat(outputs, dim=1)


# ============================================================
# 평가 메트릭
# ============================================================
@torch.no_grad()
def evaluate_metrics(model, dataloader, device, dataset):
    """검증 데이터에 대한 평가 메트릭 계산"""
    model.eval()

    all_preds = []
    all_targets = []

    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        all_preds.append(pred.cpu())
        all_targets.append(Y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # 원래 스케일로 복원
    preds_original = preds * dataset.Y_scale + dataset.Y_baseline
    targets_original = targets * dataset.Y_scale + dataset.Y_baseline

    # 오차 계산 (원래 스케일)
    errors = np.abs(preds_original - targets_original)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)

    within_1000 = np.mean(errors <= 1000)
    within_2000 = np.mean(errors <= 2000)
    within_5000 = np.mean(errors <= 5000)
    within_10000 = np.mean(errors <= 10000)

    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'within_1000': within_1000,
        'within_2000': within_2000,
        'within_5000': within_5000,
        'within_10000': within_10000,
    }


@torch.no_grad()
def evaluate_per_channel(model, dataloader, device, dataset):
    """채널별 평가 메트릭"""
    model.eval()

    all_preds = []
    all_targets = []

    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        all_preds.append(pred.cpu())
        all_targets.append(Y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # 원래 스케일로 복원
    preds_original = preds * dataset.Y_scale + dataset.Y_baseline
    targets_original = targets * dataset.Y_scale + dataset.Y_baseline

    results = []
    for ch, col in enumerate(PROX_COLS):
        ch_pred = preds_original[:, ch]
        ch_target = targets_original[:, ch]
        ch_error = np.abs(ch_pred - ch_target)

        mae = np.mean(ch_error)
        rmse = np.sqrt(np.mean(ch_error ** 2))
        max_err = np.max(ch_error)
        within_2000 = np.mean(ch_error <= 2000)

        results.append({
            'channel': col,
            'mae': mae,
            'rmse': rmse,
            'max_error': max_err,
            'within_2000': within_2000,
        })

    return results


# ============================================================
# 학습 함수
# ============================================================
def train_model(model, train_loader, val_loader, train_dataset, val_dataset,
                device, args):
    """모델 학습 (MSE Loss)"""

    criterion = nn.HuberLoss(delta=0.002)  # delta=0.002 ≈ 원래 스케일 2000 (Y_scale=1e6)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_rmse = float('inf')
    best_epoch = 0
    early_stop_patience = 20
    no_improve_count = 0
    history = {
        'train_loss': [], 'val_rmse': [], 'val_mae': [],
        'val_within_2000': [], 'learning_rate': []
    }

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0

        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Validation
        val_metrics = evaluate_metrics(model, val_loader, device, val_dataset)

        # 스케줄러 업데이트
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['rmse'])
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_within_2000'].append(val_metrics['within_2000'])
        history['learning_rate'].append(current_lr)

        # Best model 저장
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch
            no_improve_count = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_metrics['rmse'],
                'normalization': train_dataset.get_normalization_params(),
                'args': vars(args),
            }, os.path.join(args.out_dir, 'best_model.pt'))
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                break

        # 로그 출력
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss: {train_loss:.6f} | "
                  f"Val RMSE: {val_metrics['rmse']:.1f} | "
                  f"Val MAE: {val_metrics['mae']:.1f} | "
                  f"W2000: {val_metrics['within_2000']:.3f} | "
                  f"LR: {current_lr:.2e}")

    print(f"\nBest model at epoch {best_epoch+1} with Val RMSE: {best_val_rmse:.1f}")

    return history


# ============================================================
# 시각화
# ============================================================
def plot_training_history(history, out_dir):
    """학습 히스토리 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0,0].plot(history['train_loss'])
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].grid(True)

    # Validation RMSE
    axes[0,1].plot(history['val_rmse'], label='Val RMSE', color='orange')
    axes[0,1].axhline(y=2000, color='r', linestyle='--', label='Target (2000)')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('RMSE')
    axes[0,1].set_title('Validation RMSE')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # Within 2000 ratio
    axes[1,0].plot(history['val_within_2000'], label='Within 2000', color='green')
    axes[1,0].axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Ratio')
    axes[1,0].set_title('Within 2000 Ratio')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # Learning rate
    axes[1,1].plot(history['learning_rate'], color='purple')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].set_title('Learning Rate Schedule')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history_v3.png'), dpi=150)
    plt.close()


def plot_channel_results(channel_results, out_dir):
    """채널별 결과 시각화"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    channels = [r['channel'] for r in channel_results]
    rmse_values = [r['rmse'] for r in channel_results]
    within_2000_values = [r['within_2000'] for r in channel_results]

    x = np.arange(len(channels))

    # RMSE per channel
    colors = ['green' if r < 2000 else 'orange' if r < 5000 else 'red' for r in rmse_values]
    axes[0].bar(x, rmse_values, color=colors, alpha=0.7)
    axes[0].axhline(y=2000, color='r', linestyle='--', label='Target (2000)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels, rotation=45)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE per Channel')
    axes[0].legend()
    axes[0].grid(True, axis='y')

    # Within 2000 ratio per channel
    axes[1].bar(x, within_2000_values, color='steelblue', alpha=0.7)
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels, rotation=45)
    axes[1].set_ylabel('Within 2000 Ratio')
    axes[1].set_title('Within 2000 Ratio per Channel')
    axes[1].legend()
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'channel_results_v3.png'), dpi=150)
    plt.close()


# ============================================================
# 메인 함수
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Self-Detection Training V3')

    # 데이터 경로
    parser.add_argument('--data_dir', type=str,
                        default='/home/sj/self_detection_ml/data/',
                        help='CSV files directory')
    parser.add_argument('--out_dir', type=str, default='./self_detection_model_v3',
                        help='Model save directory')

    # 학습 파라미터
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # 모델 구조
    parser.add_argument('--model_type', type=str, default='per_channel',
                        choices=['simple', 'per_channel'],
                        help='Model architecture')
    parser.add_argument('--hidden_dims', type=str, default='512,256,256,128')
    parser.add_argument('--dropout', type=float, default=0.1)

    # 기타
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 출력 디렉토리 생성
    os.makedirs(args.out_dir, exist_ok=True)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==========================================================
    # 데이터 로드 (파일 단위 유지)
    # ==========================================================
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)

    file_dfs = load_all_files(args.data_dir)

    # ==========================================================
    # Train/Val 분할 (파일 단위 시간 순서 유지)
    # ==========================================================
    print("\n" + "="*60)
    print("Splitting data (temporal, time-order preserved)...")
    print("="*60)

    df_train, df_val = split_files_temporal(file_dfs, args.val_ratio)

    print(f"Train samples: {len(df_train)}")
    print(f"Val samples: {len(df_val)}")

    # ==========================================================
    # Dataset & DataLoader
    # ==========================================================
    train_dataset = SelfDetectionDatasetV3(df_train)

    # Validation에 train의 정규화 파라미터 사용
    val_dataset = SelfDetectionDatasetV3(
        df_val,
        X_mean=train_dataset.X_mean,
        X_std=train_dataset.X_std,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # ==========================================================
    # 모델 생성
    # ==========================================================
    input_dim = train_dataset.input_dim  # 18 = sin(6) + cos(6) + vel(6)
    print(f"\nInput dimension: {input_dim} (sin/cos transformed)")

    if args.model_type == 'per_channel':
        model = SelfDetectionMLPPerChannelV3(
            in_dim=input_dim,
            trunk_dims=(512, 256),
            head_dims=(128, 64, 32),
            out_dim=NUM_CHANNELS,
            dropout=args.dropout
        ).to(device)
    else:
        hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))
        model = SelfDetectionMLPV3(
            in_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=NUM_CHANNELS,
            dropout=args.dropout
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model type: {args.model_type}")
    print(f"Model parameters: {num_params:,}")

    # ==========================================================
    # 학습
    # ==========================================================
    print("\n" + "="*60)
    print("Training (joint state -> sensor value prediction)...")
    print("="*60)

    history = train_model(
        model, train_loader, val_loader,
        train_dataset, val_dataset, device, args
    )

    # ==========================================================
    # Best 모델 로드 및 최종 평가
    # ==========================================================
    print("\n" + "="*60)
    print("Final Evaluation...")
    print("="*60)

    checkpoint = torch.load(os.path.join(args.out_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 전체 메트릭
    val_metrics = evaluate_metrics(model, val_loader, device, val_dataset)
    print(f"\n[Overall Metrics]")
    print(f"  RMSE: {val_metrics['rmse']:.1f}")
    print(f"  MAE: {val_metrics['mae']:.1f}")
    print(f"  Max Error: {val_metrics['max_error']:.1f}")
    print(f"  Within 1000: {val_metrics['within_1000']:.3f}")
    print(f"  Within 2000: {val_metrics['within_2000']:.3f}")
    print(f"  Within 5000: {val_metrics['within_5000']:.3f}")
    print(f"  Within 10000: {val_metrics['within_10000']:.3f}")

    # 채널별 메트릭
    channel_results = evaluate_per_channel(model, val_loader, device, val_dataset)

    print(f"\n[Per-Channel Metrics]")
    print(f"{'Channel':<12} {'RMSE':>10} {'MAE':>10} {'W2000':>10}")
    print("-" * 45)
    for r in channel_results:
        print(f"{r['channel']:<12} {r['rmse']:>10.1f} {r['mae']:>10.1f} {r['within_2000']:>10.3f}")

    # Worst 채널 출력
    worst_channels = sorted(channel_results, key=lambda x: -x['rmse'])[:5]
    print(f"\n[Worst 5 Channels by RMSE]")
    for r in worst_channels:
        print(f"  {r['channel']}: RMSE={r['rmse']:.1f}, W2000={r['within_2000']:.3f}")

    # 시각화
    plot_training_history(history, args.out_dir)
    plot_channel_results(channel_results, args.out_dir)

    # 정규화 파라미터 저장 (추론용)
    norm_params = train_dataset.get_normalization_params()
    np.savez(os.path.join(args.out_dir, 'normalization_params.npz'), **norm_params)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved: {os.path.join(args.out_dir, 'best_model.pt')}")
    print(f"Normalization params saved: {os.path.join(args.out_dir, 'normalization_params.npz')}")


if __name__ == '__main__':
    main()
