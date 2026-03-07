#!/usr/bin/env python3
"""
Training script for self-detection raw baseline compensation (Model B).

CLI usage:
    python -m self_detection_raw.train.train \
      --data_dir /path/to/logs \
      --glob "robot_data_*.txt" \
      --out_dir outputs/run_001 \
      --val_split file --val_ratio 0.2 \
      --use_vel 1 \
      --x_norm 1 --y_norm 1 \
      --std_floor 1e-2 \
      --epochs 300 --batch 256 \
      --lr 1e-3 --wd 1e-4 \
      --hidden 128 --head_hidden 64 --dropout 0.1 \
      --seed 42
"""

import os
import argparse
import random
import json
import csv
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None, leave=True):
        if desc:
            print(desc)
        return iterable

# Import from package
from self_detection_raw.data.loader import (
    load_multiple_files, extract_features, load_file,
    split_files_train_val
)
from self_detection_raw.data.stats import (
    WelfordStats, compute_stats_from_array,
    save_norm_params
)
from self_detection_raw.models.mlp_b import ModelB
from self_detection_raw.utils.io import ensure_dir, save_json, find_files_by_pattern
from self_detection_raw.utils.metrics import compute_channel_metrics, format_metrics_report


class SelfDetectionDataset(Dataset):
    """PyTorch Dataset for self-detection training."""
    
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_mean: np.ndarray = None,
        X_std: np.ndarray = None,
        Y_mean: np.ndarray = None,
        Y_std: np.ndarray = None,
        std_floor: float = 1e-2
    ):
        """
        Args:
            X: (N, 12) input features [sin(j1..j6), cos(j1..j6)] - joint velocities removed
            Y: (N, 8) target raw values
            X_mean, X_std: Normalization params (None = compute from data)
            Y_mean, Y_std: Normalization params (None = compute from data)
            std_floor: Minimum std value
        """
        self.X = X.astype(np.float32)
        self.Y_raw = Y.astype(np.float32)
        self.std_floor = std_floor
        
        # Compute or use provided normalization
        if X_mean is None:
            X_mean, X_std = compute_stats_from_array(self.X, std_floor)
        if Y_mean is None:
            Y_mean, Y_std = compute_stats_from_array(self.Y_raw, std_floor)
        
        self.X_mean = X_mean.astype(np.float32)
        self.X_std = X_std.astype(np.float32)
        self.Y_mean = Y_mean.astype(np.float32)
        self.Y_std = Y_std.astype(np.float32)
        
        # Normalize
        self.X_norm = (self.X - self.X_mean) / self.X_std
        self.Y_norm = (self.Y_raw - self.Y_mean) / self.Y_std
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_norm[idx]),
            torch.from_numpy(self.Y_norm[idx])
        )
    
    def get_norm_params(self):
        """Get normalization parameters."""
        return {
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'Y_mean': self.Y_mean,
            'Y_std': self.Y_std,
        }




@torch.no_grad()
def evaluate(model, dataloader, dataset, device):
    """Evaluate model and compute metrics."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for X, Y in dataloader:
        X = X.to(device)
        pred_norm = model(X)
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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X, Y in tqdm(dataloader, desc='Training', leave=False):
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def find_default_data_dir():
    """Find default data directory with robot_data files."""
    import os
    possible_dirs = [
        '/home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts',
        '/home/son_rb/rb_ws/src/self_detection',
        os.path.join(os.path.dirname(__file__), '../../..', 'robotory_rb10_ros2', 'scripts'),
        os.path.join(os.path.dirname(__file__), '../../..', 'self_detection'),
    ]
    
    for data_dir in possible_dirs:
        if os.path.exists(data_dir):
            pattern = os.path.join(data_dir, 'robot_data_*.txt')
            from glob import glob
            files = glob(pattern)
            if files:
                return data_dir
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Train self-detection baseline model (Model B)')
    
    # Find default data directory
    default_data_dir = find_default_data_dir()
    
    # Data
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help=f'Directory containing robot_data_*.txt files (default: {default_data_dir})')
    parser.add_argument('--glob', type=str, default=None,
                        help='File pattern to match (e.g., "robot_data_*.txt"). If not specified, interactive selection will be shown.')
    
    # Output directory (default: outputs/run_YYYYMMDD_HHMMSS)
    default_out_dir = f"outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument('--out_dir', type=str, default=default_out_dir,
                        help=f'Output directory for model and results (default: {default_out_dir})')
    
    # Split
    parser.add_argument('--val_split', type=str, default='file', choices=['file', 'random'],
                        help='Split mode: file=file-level, random=random files')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation ratio')
    
    # Features
    parser.add_argument('--use_vel', type=int, default=0,
                        help='Use joint velocities (DEPRECATED: always 0, joint velocities removed from input)')
    
    # Normalization
    parser.add_argument('--x_norm', type=int, default=1,
                        help='Normalize input features (1) or not (0)')
    parser.add_argument('--y_norm', type=int, default=1,
                        help='Normalize output targets (1) or not (0)')
    parser.add_argument('--std_floor', type=float, default=1e-2,
                        help='Minimum std value for normalization')
    
    # Training
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay')
    
    # Model
    parser.add_argument('--hidden', type=int, default=128,
                        help='Trunk hidden dimension')
    parser.add_argument('--head_hidden', type=int, default=64,
                        help='Head hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader num_workers')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = ensure_dir(args.out_dir)
    
    # Validate data directory
    if args.data_dir is None:
        raise ValueError(
            "No data directory found. Please specify --data_dir or place robot_data_*.txt files in:\n"
            "  - /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts\n"
            "  - /home/son_rb/rb_ws/src/self_detection"
        )
    
    # Find all available files first
    all_files = find_files_by_pattern(args.data_dir, 'robot_data_*.txt')
    if not all_files:
        all_files = find_files_by_pattern(args.data_dir, '*.txt')
    
    if not all_files:
        raise ValueError(
            f"No data files found in {args.data_dir}\n"
            f"Please check the directory or specify --data_dir"
        )
    
    # Find files by pattern
    filepaths = find_files_by_pattern(args.data_dir, args.glob) if args.glob else []
    
    # If no files found or glob not specified, show interactive selection
    if not filepaths:
        print("\n" + "=" * 60)
        print("[INFO] No files matched the pattern or --glob not specified.")
        print("[INFO] Please select data files interactively:")
        print("=" * 60)
        print(f"\nAvailable data files in {args.data_dir}:")
        for i, f in enumerate(all_files):
            filename = os.path.basename(f)
            file_size = os.path.getsize(f) / (1024 * 1024)  # MB
            print(f"  [{i}] {filename} ({file_size:.2f} MB)")
        print(f"  [{len(all_files)}] Use all files")
        print(f"  [{len(all_files) + 1}] Cancel (exit)")
        
        while True:
            try:
                choice_str = input(f"\nSelect files [0-{len(all_files)}] or comma-separated indices (e.g., 0,1,2) (default: {len(all_files)} for all): ").strip()
                if choice_str == '':
                    choice_str = str(len(all_files))
                
                # Parse comma-separated indices
                if ',' in choice_str:
                    indices = [int(x.strip()) for x in choice_str.split(',')]
                else:
                    indices = [int(choice_str)]
                
                # Check if "use all" selected
                if len(all_files) in indices:
                    filepaths = all_files
                    print(f"[INFO] Selected all {len(filepaths)} files")
                    break
                
                # Check if "cancel" selected
                if (len(all_files) + 1) in indices:
                    print("[INFO] Training cancelled by user.")
                    return
                
                # Validate indices
                valid_indices = [idx for idx in indices if 0 <= idx < len(all_files)]
                if len(valid_indices) != len(indices):
                    print(f"[ERROR] Invalid indices. Please enter numbers between 0 and {len(all_files)}.")
                    continue
                
                filepaths = [all_files[idx] for idx in valid_indices]
                print(f"[INFO] Selected {len(filepaths)} file(s):")
                for f in filepaths:
                    print(f"  - {os.path.basename(f)}")
                print("=" * 60 + "\n")
                break
                
            except ValueError:
                print(f"[ERROR] Invalid input. Please enter numbers between 0 and {len(all_files) + 1}, or comma-separated indices.")
            except KeyboardInterrupt:
                print("\n[INFO] Training cancelled by user (Ctrl+C).")
                return
    else:
        print(f"Found {len(filepaths)} file(s) matching pattern '{args.glob}'")
    
    # Split files
    train_files, val_files = split_files_train_val(
        filepaths, args.val_ratio, args.val_split, args.seed
    )
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    if len(train_files) == 0:
        raise ValueError(
            "Train 파일이 없습니다. 더 많은 데이터 파일을 추가하거나, "
            "--val_ratio를 줄이거나, --val_split within 옵션을 사용하세요."
        )
    
    # Load data
    print("Loading training data...")
    X_train, Y_train = load_multiple_files(train_files, use_vel=bool(args.use_vel))
    print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
    
    if len(val_files) > 0:
        print("Loading validation data...")
        X_val, Y_val = load_multiple_files(val_files, use_vel=bool(args.use_vel))
        print(f"Val: X={X_val.shape}, Y={Y_val.shape}")
    else:
        print("Validation 파일이 없습니다. Train 데이터의 일부를 validation으로 사용합니다.")
        # Train 데이터에서 일부를 val로 사용
        n_val = max(1, int(len(X_train) * args.val_ratio))
        indices = np.arange(len(X_train))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_val = X_train[val_indices]
        Y_val = Y_train[val_indices]
        X_train = X_train[train_indices]
        Y_train = Y_train[train_indices]
        print(f"After split - Train: X={X_train.shape}, Y={Y_train.shape}")
        print(f"After split - Val: X={X_val.shape}, Y={Y_val.shape}")
    
    # Create datasets
    train_dataset = SelfDetectionDataset(
        X_train, Y_train,
        std_floor=args.std_floor
    )
    
    # Use train normalization for val
    val_dataset = SelfDetectionDataset(
        X_val, Y_val,
        X_mean=train_dataset.X_mean,
        X_std=train_dataset.X_std,
        Y_mean=train_dataset.Y_mean,
        Y_std=train_dataset.Y_std,
        std_floor=args.std_floor
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    model = ModelB(
        in_dim=12,  # sin(j1..j6) + cos(j1..j6) = 12 (joint velocities removed)
        trunk_hidden=args.hidden,
        head_hidden=args.head_hidden,
        out_dim=8,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.SmoothL1Loss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # verbose parameter not available in all PyTorch versions
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=True)
    except TypeError:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25)
    
    # Training loop
    best_val_std = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [],
        'val_avg_std': [],
        'val_mae': [],
        'val_rmse': [],
    }
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, val_dataset, device)
        
        # Scheduler step
        scheduler.step(val_metrics['avg_std'])
        
        # History
        history['train_loss'].append(train_loss)
        history['val_avg_std'].append(val_metrics['avg_std'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        
        # Save best model
        if val_metrics['avg_std'] < best_val_std:
            best_val_std = val_metrics['avg_std']
            best_epoch = epoch
            
            # Get normalization params from train dataset
            norm_params = train_dataset.get_norm_params()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
                'normalization': norm_params,  # Save normalization params in checkpoint
            }, out_dir / 'model.pt')
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val STD: {val_metrics['avg_std']:.2f} | "
                  f"Val MAE: {val_metrics['mae']:.2f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    print(f"\nBest model at epoch {best_epoch} (Val STD: {best_val_std:.2f})")
    
    # Load best model for final evaluation
    checkpoint = torch.load(out_dir / 'model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    final_metrics = evaluate(model, val_loader, val_dataset, device)
    
    # Save normalization params
    norm_params = train_dataset.get_norm_params()
    save_norm_params(
        norm_params['X_mean'],
        norm_params['X_std'],
        norm_params['Y_mean'],
        norm_params['Y_std'],
        args.std_floor,
        str(out_dir / 'norm_params.json')
    )
    
    # Save config
    save_json(vars(args), str(out_dir / 'config.json'))
    
    # Save report
    channel_names = [f"raw{i}" for i in range(1, 9)]
    report = {
        'best_epoch': best_epoch,
        'best_val_std': float(best_val_std),
        'final_metrics': {
            'mae': final_metrics['mae'],
            'rmse': final_metrics['rmse'],
            'avg_std': final_metrics['avg_std'],
        },
        'channel_metrics': {
            'raw_std': final_metrics['channel_metrics']['raw_std'].tolist(),
            'residual_std': final_metrics['channel_metrics']['residual_std'].tolist(),
            'improvement': final_metrics['channel_metrics']['improvement'].tolist(),
        },
    }
    save_json(report, str(out_dir / 'report.json'))
    
    # Save history
    with open(out_dir / 'history.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_avg_std', 'val_mae', 'val_rmse'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                history['train_loss'][i],
                history['val_avg_std'][i],
                history['val_mae'][i],
                history['val_rmse'][i],
            ])
    
    # Print final report
    print("\n" + "="*60)
    print("Final Report")
    print("="*60)
    print(format_metrics_report(final_metrics['channel_metrics'], channel_names))
    print(f"\nModel saved to: {out_dir / 'model.pt'}")
    print(f"Normalization params saved to: {out_dir / 'norm_params.json'}")


if __name__ == '__main__':
    main()

