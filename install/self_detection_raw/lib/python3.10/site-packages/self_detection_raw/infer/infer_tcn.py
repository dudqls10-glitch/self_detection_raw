#!/usr/bin/env python3
"""Offline inference for Method B (MLP main + TCN residual)."""

import argparse
import csv
import os
from collections import deque

import numpy as np
import torch

from self_detection_raw.data.loader import extract_features, load_file
from self_detection_raw.data.stats import load_norm_params
from self_detection_raw.models.mlp_tcn_residual import MLP_TCN_ResidualModel


@torch.no_grad()
def infer_file(model, filepath, x_mean, x_std, y_mean, y_std, seq_len=32, warmup_zero_pad=True):
    data = load_file(filepath)
    x, y = extract_features(data, use_vel=False)
    x_norm = (x - x_mean) / x_std

    seq_buf = deque(maxlen=seq_len)
    preds = []

    for i in range(len(x_norm)):
        seq_buf.append(x_norm[i])

        if len(seq_buf) < seq_len and not warmup_zero_pad:
            preds.append(np.zeros_like(y_mean, dtype=np.float32))
            continue

        if len(seq_buf) < seq_len:
            seq = np.zeros((seq_len, x_norm.shape[1]), dtype=np.float32)
            b = np.array(list(seq_buf), dtype=np.float32)
            seq[-len(b):] = b
        else:
            seq = np.array(list(seq_buf), dtype=np.float32)

        x_seq_t = torch.from_numpy(seq).unsqueeze(0)
        y_hat_norm, _ = model(x_seq_t, use_residual=True)
        pred = y_hat_norm.numpy().squeeze(0) * y_std + y_mean
        preds.append(pred.astype(np.float32))

    pred_arr = np.stack(preds, axis=0)
    residuals = y - pred_arr

    return {
        "timestamp": data[:, 0],
        "raw": y,
        "pred_raw": pred_arr,
        "residual": residuals,
        "j": data[:, 1:7],
        "jv": data[:, 7:13],
    }


def save_results_csv(results, output_path):
    n = len(results["timestamp"])

    cols = ["timestamp"]
    cols += [f"raw{i}" for i in range(1, 9)]
    cols += [f"pred_raw{i}" for i in range(1, 9)]
    cols += [f"residual{i}" for i in range(1, 9)]
    cols += [f"j{i}" for i in range(1, 7)]
    cols += [f"jv{i}" for i in range(1, 7)]

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            row = [results["timestamp"][i]]
            row.extend(results["raw"][i].tolist())
            row.extend(results["pred_raw"][i].tolist())
            row.extend(results["residual"][i].tolist())
            row.extend(results["j"][i].tolist())
            row.extend(results["jv"][i].tolist())
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Offline infer for MLP+TCN residual model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--warmup_zero_pad", type=int, default=1)
    args = parser.parse_args()

    if args.output is None:
        base = os.path.basename(args.input).replace(".txt", "")
        args.output = f"residual_tcn_{base}.csv"

    device = torch.device("cpu")

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    dilations = model_args.get("tcn_dilations", "1,2,4,8")
    if isinstance(dilations, str):
        dilations = tuple(int(x.strip()) for x in dilations.split(",") if x.strip())
    else:
        dilations = tuple(int(x) for x in dilations)

    model = MLP_TCN_ResidualModel(
        in_dim=int(model_args.get("in_dim", 12)),
        out_dim=int(model_args.get("out_dim", 8)),
        trunk_hidden=int(model_args.get("hidden", 128)),
        head_hidden=int(model_args.get("head_hidden", 64)),
        tcn_hidden=int(model_args.get("tcn_hidden", 64)),
        tcn_kernel=int(model_args.get("tcn_kernel", 3)),
        tcn_dilations=dilations,
        dropout=float(model_args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.norm and os.path.exists(args.norm):
        x_mean, x_std, y_mean, y_std, _ = load_norm_params(args.norm)
    elif "normalization" in checkpoint:
        norm = checkpoint["normalization"]
        x_mean = np.array(norm["X_mean"], dtype=np.float32)
        x_std = np.array(norm["X_std"], dtype=np.float32)
        y_mean = np.array(norm["Y_mean"], dtype=np.float32)
        y_std = np.array(norm["Y_std"], dtype=np.float32)
    else:
        norm_path = os.path.join(os.path.dirname(args.model), "norm_params.json")
        x_mean, x_std, y_mean, y_std, _ = load_norm_params(norm_path)

    results = infer_file(
        model,
        args.input,
        x_mean,
        x_std,
        y_mean,
        y_std,
        seq_len=args.seq_len,
        warmup_zero_pad=bool(args.warmup_zero_pad),
    )
    save_results_csv(results, args.output)

    residuals = results["residual"]
    print(f"Saved: {args.output}")
    print(f"Residual mean: {np.mean(residuals, axis=0)}")
    print(f"Residual std : {np.std(residuals, axis=0)}")


if __name__ == "__main__":
    main()
