#!/usr/bin/env python3
"""
Evaluation script for self-detection model.

CLI usage:
    python -m self_detection_raw.train.eval \
      --model outputs/run_001/model.pt \
      --norm outputs/run_001/norm_params.json \
      --data_dir /path/to/logs \
      --glob "robot_data_*.txt" \
      --split file --val_ratio 0.2
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from self_detection_raw.data.loader import (
    load_multiple_files, split_files_train_val, find_files_by_pattern
)
from self_detection_raw.data.stats import load_norm_params
from self_detection_raw.models.mlp_b import ModelB
from self_detection_raw.utils.io import load_json
from self_detection_raw.utils.metrics import compute_channel_metrics, format_metrics_report


class EvalDataset:
    """Simple dataset for evaluation."""
    
    def __init__(self, X, Y, X_mean, X_std, Y_mean, Y_std):
        self.X = (X - X_mean) / X_std
        self.Y_raw = Y
        self.Y_mean = Y_mean
        self.Y_std = Y_std
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.float32))


@torch.no_grad()
def evaluate_model(model, X, Y, dataset, device):
    """Evaluate model."""
    model.eval()
    
    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    pred_norm = model(X_tensor).cpu().numpy()
    
    # Denormalize
    pred = pred_norm * dataset.Y_std + dataset.Y_mean
    targets = Y
    
    # Residuals
    residuals = targets - pred
    
    # Metrics
    metrics = compute_channel_metrics(targets, residuals)
    
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
    parser = argparse.ArgumentParser(description='Evaluate self-detection model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--norm', type=str, required=True,
                        help='Path to normalization params (.json)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing robot_data_*.txt files')
    parser.add_argument('--glob', type=str, default='robot_data_*.txt',
                        help='File pattern to match')
    parser.add_argument('--split', type=str, default='file', choices=['file', 'random'],
                        help='Split mode')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation ratio')
    parser.add_argument('--use_vel', type=int, default=1,
                        help='Use joint velocities')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model_args = checkpoint.get('args', {})
    
    model = ModelB(
        in_dim=18,
        trunk_hidden=model_args.get('hidden', 128),
        head_hidden=model_args.get('head_hidden', 64),
        out_dim=8,
        dropout=model_args.get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from: {args.model}")
    
    # Load normalization params
    x_mean, x_std, y_mean, y_std, std_floor = load_norm_params(args.norm)
    print(f"Loaded normalization params from: {args.norm}")
    
    # Find files
    filepaths = find_files_by_pattern(args.data_dir, args.glob)
    if not filepaths:
        raise ValueError(f"No files found matching pattern: {args.glob} in {args.data_dir}")
    
    print(f"Found {len(filepaths)} files")
    
    # Split (same as training)
    train_files, val_files = split_files_train_val(
        filepaths, args.val_ratio, args.split, args.seed
    )
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    # Load validation data
    print("Loading validation data...")
    X_val, Y_val = load_multiple_files(val_files, use_vel=bool(args.use_vel))
    print(f"Val: X={X_val.shape}, Y={Y_val.shape}")
    
    # Create dataset
    val_dataset = EvalDataset(X_val, Y_val, x_mean, x_std, y_mean, y_std)
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_model(model, X_val, Y_val, val_dataset, device)
    
    # Print report
    channel_names = [f"raw{i}" for i in range(1, 9)]
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"Average STD: {metrics['avg_std']:.2f}")
    print("\n" + format_metrics_report(metrics['channel_metrics'], channel_names))


if __name__ == '__main__':
    main()

