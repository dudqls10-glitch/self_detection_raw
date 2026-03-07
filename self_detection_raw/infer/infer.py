#!/usr/bin/env python3
"""
Offline inference script for generating residuals.

CLI usage:
    python -m self_detection_raw.infer.infer \
      --model outputs/run_001/model.pt \
      --norm outputs/run_001/norm_params.json \
      --input robot_data_20260130_173052.txt \
      --output residual_20260130_173052.csv
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from self_detection_raw.data.loader import load_file, extract_features
from self_detection_raw.data.stats import load_norm_params
from self_detection_raw.models.mlp_b import ModelB


@torch.no_grad()
def infer_file(model, filepath, x_mean, x_std, y_mean, y_std, use_vel=True):
    """Run inference on a single file."""
    # Load data
    data = load_file(filepath)  # (N, 37)
    
    # Extract features and targets
    X, Y = extract_features(data, use_vel=use_vel)
    
    # Normalize input
    X_norm = (X - x_mean) / x_std
    
    # Model inference
    X_tensor = torch.from_numpy(X_norm.astype(np.float32))
    pred_norm = model(X_tensor).numpy()
    
    # Denormalize output
    pred = pred_norm * y_std + y_mean
    
    # Compute residuals
    residuals = Y - pred
    
    # Extract other data for CSV
    timestamps = data[:, 0]  # timestamp
    j_pos = data[:, 1:7]  # j1..j6
    j_vel = data[:, 7:13] if use_vel else np.zeros((len(data), 6))  # jv1..jv6
    
    return {
        'timestamp': timestamps,
        'raw': Y,
        'pred_raw': pred,
        'residual': residuals,
        'j1': j_pos[:, 0],
        'j2': j_pos[:, 1],
        'j3': j_pos[:, 2],
        'j4': j_pos[:, 3],
        'j5': j_pos[:, 4],
        'j6': j_pos[:, 5],
        'jv1': j_vel[:, 0],
        'jv2': j_vel[:, 1],
        'jv3': j_vel[:, 2],
        'jv4': j_vel[:, 3],
        'jv5': j_vel[:, 4],
        'jv6': j_vel[:, 5],
    }


def save_results_csv(results, output_path):
    """Save inference results to CSV."""
    n_samples = len(results['timestamp'])
    
    # Column names
    columns = ['timestamp']
    for i in range(1, 9):
        columns.append(f'raw{i}')
    for i in range(1, 9):
        columns.append(f'pred_raw{i}')
    for i in range(1, 9):
        columns.append(f'residual{i}')
    columns.extend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
    columns.extend(['jv1', 'jv2', 'jv3', 'jv4', 'jv5', 'jv6'])
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
        for i in range(n_samples):
            row = [results['timestamp'][i]]
            # raw1..raw8
            row.extend(results['raw'][i].tolist())
            # pred_raw1..pred_raw8
            row.extend(results['pred_raw'][i].tolist())
            # residual1..residual8
            row.extend(results['residual'][i].tolist())
            # j1..j6
            row.extend(results[f'j{j}'][i] for j in range(1, 7))
            # jv1..jv6
            row.extend(results[f'jv{j}'][i] for j in range(1, 7))
            writer.writerow(row)
    
    print(f"Saved {n_samples} samples to {output_path}")


def find_latest_model():
    """Find latest model checkpoint in outputs directory."""
    import os
    from pathlib import Path
    
    outputs_dir = Path(__file__).parent.parent.parent / 'outputs'
    if not outputs_dir.exists():
        return None
    
    model_files = list(outputs_dir.glob('**/model.pt'))
    if not model_files:
        return None
    
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(model_files[0])


def find_default_input_file():
    """Find default input robot_data file."""
    import os
    import glob
    
    possible_dirs = [
        '/home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts',
        '/home/son_rb/rb_ws/src/self_detection',
    ]
    
    for data_dir in possible_dirs:
        if os.path.exists(data_dir):
            pattern = os.path.join(data_dir, 'robot_data_*.txt')
            files = glob.glob(pattern)
            if files:
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0]
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Run offline inference and generate residuals')
    
    default_model = find_latest_model()
    default_input = find_default_input_file()
    
    parser.add_argument('--model', type=str, default=default_model,
                        help=f'Path to model checkpoint (.pt) (default: {default_model})')
    parser.add_argument('--norm', type=str, default=None,
                        help='Path to normalization params (.json). If not provided, will try to load from checkpoint.')
    parser.add_argument('--input', type=str, default=default_input,
                        help=f'Input robot_data_*.txt file (default: {default_input})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (default: residual_<input_filename>.csv)')
    parser.add_argument('--use_vel', type=int, default=1,
                        help='Use joint velocities')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model is None:
        raise ValueError("No model found. Please run training first or specify --model")
    
    if args.input is None:
        raise ValueError("No input file found. Please specify --input")
    
    # Default output filename
    if args.output is None:
        import os
        input_basename = os.path.basename(args.input).replace('.txt', '')
        args.output = f'residual_{input_basename}.csv'
    
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
    if args.norm and os.path.exists(args.norm):
        x_mean, x_std, y_mean, y_std, std_floor = load_norm_params(args.norm)
        print(f"Loaded normalization params from: {args.norm}")
    elif 'normalization' in checkpoint:
        norm_params = checkpoint['normalization']
        x_mean = np.array(norm_params['X_mean'], dtype=np.float32)
        x_std = np.array(norm_params['X_std'], dtype=np.float32)
        y_mean = np.array(norm_params['Y_mean'], dtype=np.float32)
        y_std = np.array(norm_params['Y_std'], dtype=np.float32)
        std_floor = 1e-2
        print(f"Loaded normalization params from checkpoint")
    else:
        # Try to find norm_params.json in model directory
        model_dir = os.path.dirname(args.model)
        default_norm_path = os.path.join(model_dir, 'norm_params.json')
        if os.path.exists(default_norm_path):
            x_mean, x_std, y_mean, y_std, std_floor = load_norm_params(default_norm_path)
            print(f"Loaded normalization params from: {default_norm_path}")
        else:
            raise ValueError("Normalization params not found. Please provide --norm or re-run training.")
    
    # Run inference
    print(f"Processing: {args.input}")
    results = infer_file(model, args.input, x_mean, x_std, y_mean, y_std, bool(args.use_vel))
    
    # Save results
    save_results_csv(results, args.output)
    
    # Print summary
    residuals = results['residual']
    print(f"\nResidual statistics:")
    print(f"  Mean: {np.mean(residuals, axis=0)}")
    print(f"  STD: {np.std(residuals, axis=0)}")
    print(f"  Max: {np.max(np.abs(residuals), axis=0)}")


if __name__ == '__main__':
    main()

