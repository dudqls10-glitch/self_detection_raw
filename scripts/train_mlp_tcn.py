#!/usr/bin/env python3
"""
MLP + TCN residual 학습 헬퍼 스크립트

`train_mlp.py`와 유사하지만

- 모델으로 `MLP_TCN_ResidualModel`을 사용하고
  (메인 MLP + TCN residual stream)
- 추가 하이퍼파라미터 (`tcn_hidden`, `tcn_kernel`, `tcn_dilations`)
- `--no-residual` 옵션으로 잔차를 비활성화 가능

실행 예:
    python scripts/train_mlp_tcn.py --epochs 200

상단 변수만 바꾸거나 커맨드라인 인수로 덮어써서 사용합니다.
"""

# ensure package modules import correctly when executed from scripts/
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import random
from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from self_detection_raw.data.loader import load_multiple_files
from self_detection_raw.train.train import SelfDetectionDataset, evaluate
from self_detection_raw.models.mlp_tcn_residual import MLP_TCN_ResidualModel
from self_detection_raw.utils.io import ensure_dir
from self_detection_raw.utils.metrics import compute_channel_metrics

# ---------------------------------------------------------------------------
# 기본 설정값 (커맨드라인에서 덮어쓸 수 있음)
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
VAL_FILES = ["1dataset_50_25.txt"]

script_root = Path(__file__).parent.absolute()
OUT_DIR = str(script_root / "model")
NAME_PREFIX = "mlp_tcn"

# 기본 하이퍼파라미터
USE_VEL = False
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

# TCN-specific defaults
TCN_HIDDEN = 64
TCN_KERNEL = 3
TCN_DILATIONS = [1, 2, 4, 8]
# ---------------------------------------------------------------------------


def full_paths(names, data_dir):
    paths = []
    for fn in names:
        p = Path(fn)
        if not p.is_absolute():
            paths.append(str(Path(data_dir) / fn))
        else:
            paths.append(str(p))
    return paths


def train_epoch_tcn(model, dataloader, criterion, optimizer, device, use_residual):
    model.train()
    total_loss = 0.0
    n_batches = 0
    from tqdm import tqdm

    for X, Y in tqdm(dataloader, desc='Training', leave=False):
        X = X.to(device)
        Y = Y.to(device)
        # add time dimension for TCN model (B,D) -> (B,1,D)
        if X.dim() == 2:
            X = X.unsqueeze(1)

        optimizer.zero_grad()
        out = model(X, use_residual=use_residual)
        pred = out[0] if isinstance(out, tuple) else out
        loss = criterion(pred, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate_tcn(model, dataloader, dataset, device, use_residual):
    """Evaluate model, similar to train.train.evaluate but with TCN support."""
    model.eval()

    all_preds = []
    all_targets = []

    for X, Y in dataloader:
        X = X.to(device)
        # add time dimension if necessary
        if X.dim() == 2:
            X = X.unsqueeze(1)
        out = model(X, use_residual=use_residual)
        pred_norm = out[0] if isinstance(out, tuple) else out
        all_preds.append(pred_norm.cpu().numpy())
        all_targets.append(Y.cpu().numpy())

    preds_norm = np.concatenate(all_preds, axis=0)
    targets_norm = np.concatenate(all_targets, axis=0)

    # Denormalize
    preds = preds_norm * dataset.Y_std + dataset.Y_mean
    targets = targets_norm * dataset.Y_std + dataset.Y_mean

    # Compute residuals
    residuals = targets - preds

    # Metrics
    metrics = compute_channel_metrics(targets, residuals)

    # Overall metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    avg_std = np.mean(metrics['residual_std'])

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'avg_std': float(avg_std),
        'channel_metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Train MLP+TCN residual model')
    parser.add_argument('--data-dir', default=DATA_DIR, help='directory containing data files')
    parser.add_argument('--train-files', default=','.join(TRAIN_FILES),
                        help='comma-separated list of training file names')
    parser.add_argument('--val-files', default=','.join(VAL_FILES),
                        help='comma-separated list of validation file names')
    parser.add_argument('--out-dir', default=OUT_DIR, help='output directory to save model and params')
    parser.add_argument('--name-prefix', default=NAME_PREFIX, help='model name prefix')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--wd', type=float, default=WD)
    parser.add_argument('--hidden', type=int, default=HIDDEN)
    parser.add_argument('--head-hidden', type=int, default=HEAD_HIDDEN)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--tcn-hidden', type=int, default=TCN_HIDDEN)
    parser.add_argument('--tcn-kernel', type=int, default=TCN_KERNEL)
    parser.add_argument('--tcn-dilations', default=','.join(str(d) for d in TCN_DILATIONS),
                        help='comma-separated dilation list')
    parser.add_argument('--no-residual', action='store_true', help='disable residual stream')
    args = parser.parse_args()

    # apply args
    data_dir = args.data_dir
    train_files = args.train_files.split(',') if args.train_files else []
    val_files = args.val_files.split(',') if args.val_files else []
    out_dir = args.out_dir
    name_prefix = args.name_prefix
    epochs = args.epochs
    batch_size = args.batch
    lr = args.lr
    wd = args.wd
    hidden = args.hidden
    head_hidden = args.head_hidden
    dropout = args.dropout
    seed = args.seed
    use_residual = not args.no_residual
    tcn_hidden = args.tcn_hidden
    tcn_kernel = args.tcn_kernel
    tcn_dilations = [int(x) for x in args.tcn_dilations.split(',') if x]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_paths = full_paths(train_files, data_dir)
    val_paths = full_paths(val_files, data_dir)

    if not train_paths:
        raise RuntimeError("TRAIN_FILES 목록이 비어 있습니다.")

    print(f"Training files ({len(train_paths)}):\n  " + "\n  ".join(train_paths))
    if val_paths:
        print(f"Validation files ({len(val_paths)}):\n  " + "\n  ".join(val_paths))

    # load data
    X_train, Y_train = load_multiple_files(train_paths, use_vel=USE_VEL)
    if val_paths:
        X_val, Y_val = load_multiple_files(val_paths, use_vel=USE_VEL)
    else:
        X_val, Y_val = np.array([]).reshape(0, 18), np.array([]).reshape(0, 8)

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
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = MLP_TCN_ResidualModel(
        in_dim=12,
        out_dim=8,
        trunk_hidden=hidden,
        head_hidden=head_hidden,
        tcn_hidden=tcn_hidden,
        tcn_kernel=tcn_kernel,
        tcn_dilations=tcn_dilations,
        dropout=dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
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

    for epoch in range(1, epochs + 1):
        loss = train_epoch_tcn(model, train_loader, criterion, optimizer, device, use_residual)
        metrics = evaluate_tcn(model, val_loader, val_ds, device, use_residual) if len(val_paths) > 0 else {}
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
            }, Path(run_dir) / fname)
            print(f"Saved best model to {Path(run_dir)/fname}")

        if epoch % 10 == 0 or epoch == 1:
            msg = f"Epoch {epoch}/{epochs} | Train loss: {loss:.6f}"
            if 'avg_std' in metrics:
                msg += f" | Val std: {metrics['avg_std']:.2f}"
            print(msg)

    norm_params = train_ds.get_norm_params()
    from self_detection_raw.utils.io import save_json
    from self_detection_raw.data.stats import save_norm_params

    save_norm_params(
        norm_params['X_mean'],
        norm_params['X_std'],
        norm_params['Y_mean'],
        norm_params['Y_std'],
        STD_FLOOR,
        str(Path(run_dir) / 'norm_params.json')
    )
    cfg = {
        'DATA_DIR': data_dir,
        'TRAIN_FILES': train_files,
        'VAL_FILES': val_files,
        'USE_VEL': USE_VEL,
        'X_NORM': X_NORM,
        'Y_NORM': Y_NORM,
        'STD_FLOOR': STD_FLOOR,
        'EPOCHS': epochs,
        'BATCH': batch_size,
        'LR': lr,
        'WD': wd,
        'HIDDEN': hidden,
        'HEAD_HIDDEN': head_hidden,
        'DROPOUT': dropout,
        'SEED': seed,
        'NUM_WORKERS': NUM_WORKERS,
        'TCN_HIDDEN': tcn_hidden,
        'TCN_KERNEL': tcn_kernel,
        'TCN_DILATIONS': tcn_dilations,
        'USE_RESIDUAL': use_residual,
    }
    save_json(cfg, str(Path(run_dir) / 'config.json'))

    with open(Path(run_dir) / 'history.csv', 'w') as f:
        f.write('epoch,train_loss,val_avg_std\n')
        for idx, loss_val in enumerate(history['train_loss']):
            valstd = history['val_avg_std'][idx] if idx < len(history['val_avg_std']) else ''
            f.write(f"{idx+1},{loss_val},{valstd}\n")

    print(f"Finished training, best val std: {best_val_std}")
    print(f"Run directory: {run_dir}")

    try:
        import json
        last_path = Path(out_dir) / 'last_run.json'
        with open(last_path, 'w') as f:
            json.dump({'run_dir': str(run_dir)}, f)
        print(f"Recorded last run in {last_path}")
    except Exception as e:
        print(f"Warning: could not write last_run.json: {e}")


if __name__ == '__main__':
    main()
