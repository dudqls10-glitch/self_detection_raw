#!/usr/bin/env python3
"""
가장 간단한 MLP 학습 헬퍼 스크립트

이 파일을 열어 상단의 변수를 수정하면
 - DATA_DIR: 데이터가 들어 있는 디렉터리
 - TRAIN_FILES, VAL_FILES: 학습/검증에 사용할 파일 이름 목록
 - OUT_DIR: 모델을 저장할 디렉터리

학습 파라미터도 직접 여기서 변경할 수 있습니다.

파일 이름은 `DATA_DIR`를 기준으로 한 상대 경로(또는 절대 경로)로
기입해 주세요.

실행:
    python scripts/train_mlp.py

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

from self_detection_raw.data.loader import load_multiple_files
from self_detection_raw.train.train import SelfDetectionDataset, train_epoch, evaluate
from self_detection_raw.models.mlp_b import ModelB
from self_detection_raw.utils.io import ensure_dir

# ---------------------------------------------------------------------------
# 기본 설정값 (커맨드라인에서 덮어쓸 수 있음)
# ---------------------------------------------------------------------------
DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"

# 파일 이름 목록은 콤마가 아닌 파이썬 리스트로 작성합니다. CLI에서는 쉼표로 전달
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

# 결과를 저장할 위치 (절대 경로). 기본은 스크립트 폴더 하위의 model/ 디렉토리
script_root = Path(__file__).parent.absolute()
OUT_DIR = str(script_root / "model")
# 모델 파일 이름에 들어갈 접두어
NAME_PREFIX = "mlp"

# 학습 하이퍼파라미터
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
# ---------------------------------------------------------------------------


def full_paths(names):
    paths = []
    for fn in names:
        p = Path(fn)
        if not p.is_absolute():
            paths.append(str(Path(DATA_DIR) / fn))
        else:
            paths.append(str(p))
    return paths


def main():
    # 커맨드라인 인자를 파싱
    parser = argparse.ArgumentParser(description='Train simple MLP on multiple dataset files')
    parser.add_argument('--data-dir', default=DATA_DIR, help='directory containing data files')
    parser.add_argument('--train-files', default=','.join(TRAIN_FILES),
                        help='comma-separated list of training file names')
    parser.add_argument('--val-files', default=','.join(VAL_FILES),
                        help='comma-separated list of validation file names')
    parser.add_argument('--out-dir', default=OUT_DIR, help='output directory to save model and params')
    parser.add_argument('--name-prefix', default=NAME_PREFIX, help='model name prefix (e.g. mlp)')
    # allow overriding hyperparams as needed
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--wd', type=float, default=WD)
    parser.add_argument('--hidden', type=int, default=HIDDEN)
    parser.add_argument('--head-hidden', type=int, default=HEAD_HIDDEN)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    # apply parsed values and create local variables
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_paths = full_paths(train_files)
    val_paths = full_paths(val_files)

    if not train_paths:
        raise RuntimeError("TRAIN_FILES 목록이 비어 있습니다.")

    print(f"Training files ({len(train_paths)}):\n  " + "\n  ".join(train_paths))
    if val_paths:
        print(f"Validation files ({len(val_paths)}):\n  " + "\n  ".join(val_paths))

    # 데이터 로드
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

    model = ModelB(
        in_dim=12,
        trunk_hidden=hidden,
        head_hidden=head_hidden,
        out_dim=8,
        dropout=dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.SmoothL1Loss()

    # scheduler not strictly needed but replicate behaviour
    try:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=True)
    except Exception:
        scheduler = None

    # create a timestamped run directory inside out_dir
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    run_dir = ensure_dir(os.path.join(out_dir, f"{name_prefix}_{timestamp}"))

    best_val_std = float('inf')
    history = {'train_loss': [], 'val_avg_std': []}

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, val_ds, device) if len(val_paths) > 0 else {}
        if scheduler and 'avg_std' in metrics:
            scheduler.step(metrics['avg_std'])

        history['train_loss'].append(loss)
        if 'avg_std' in metrics:
            history['val_avg_std'].append(metrics['avg_std'])

        if 'avg_std' in metrics and metrics['avg_std'] < best_val_std:
            best_val_std = metrics['avg_std']
            # save checkpoint with normalization parameters as well
            norm_params = train_ds.get_norm_params()
            # model filename: model_MMDD_mlp.pt
            # save model checkpoint inside run directory
            fname = "model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': metrics,
                'normalization': norm_params,
            }, Path(out_dir) / fname)
            print(f"Saved best model to {Path(run_dir)/fname}")

        if epoch % 10 == 0 or epoch == 1:
            msg = f"Epoch {epoch}/{epochs} | Train loss: {loss:.6f}"
            if 'avg_std' in metrics:
                msg += f" | Val std: {metrics['avg_std']:.2f}"
            print(msg)

    # save normalization parameters regardless of best epoch
    norm_params = train_ds.get_norm_params()
    # reuse utils from package
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
    # save simple config and history
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
    }
    save_json(cfg, str(Path(run_dir) / 'config.json'))

    with open(Path(run_dir) / 'history.csv', 'w') as f:
        f.write('epoch,train_loss,val_avg_std\n')
        for idx, loss_val in enumerate(history['train_loss']):
            valstd = history['val_avg_std'][idx] if idx < len(history['val_avg_std']) else ''
            f.write(f"{idx+1},{loss_val},{valstd}\n")

    print(f"Finished training, best val std: {best_val_std}")
    print(f"Run directory: {run_dir}")

    # remember this run for later inference
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
