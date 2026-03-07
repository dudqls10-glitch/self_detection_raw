#!/usr/bin/env python3
"""
Real-time compensation node for Method B (MLP + TCN residual).
"""

import glob
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
import torch
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState, Range

from self_detection_raw.data.stats import load_norm_params
from self_detection_raw.models.mlp_tcn_residual import MLP_TCN_ResidualModel

N_JOINTS = 6
N_SENSORS = 8
HARDWARE_BASELINE = 4.0e7


class RealtimeInferTCNNode(Node):
    def __init__(self):
        super().__init__("realtime_infer_tcn")

        self.declare_parameter("model_file", "")
        self.declare_parameter("norm_file", "")
        self.declare_parameter("use_hardware_baseline", True)
        self.declare_parameter("log_rate", 100.0)
        self.declare_parameter("seq_len", 32)
        self.declare_parameter("warmup_zero_pad", True)

        model_file = self.get_parameter("model_file").value
        norm_file = self.get_parameter("norm_file").value
        self.use_hardware_baseline = bool(self.get_parameter("use_hardware_baseline").value)
        self.log_rate = float(self.get_parameter("log_rate").value)
        self.seq_len = int(self.get_parameter("seq_len").value)
        self.warmup_zero_pad = bool(self.get_parameter("warmup_zero_pad").value)

        self.cb_group = ReentrantCallbackGroup()

        self.raw_data = np.zeros(N_SENSORS, dtype=np.float32)
        self.joint_positions = None
        self.joint_velocities = None
        self.raw_received = False
        self.joint_received = False

        self.model = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x_seq_buffer = deque(maxlen=self.seq_len)

        if not model_file:
            model_file = self._find_latest_model()
            if not model_file:
                self.get_logger().error("No TCN model found. Train first or pass model_file.")
                self._model_load_failed = True
                return
            self.get_logger().info(f"Auto-selected latest TCN model: {model_file}")

        self._load_model(model_file, norm_file)
        if self.model is None:
            self._model_load_failed = True
            return

        for i in range(N_SENSORS):
            self.create_subscription(
                Range,
                f"/raw_distance{i+1}",
                lambda msg, idx=i: self.raw_callback(msg, idx),
                10,
                callback_group=self.cb_group,
            )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10,
            callback_group=self.cb_group,
        )

        self.comp_pubs = []
        self.pred_pubs = []
        self.res_pubs = []
        for i in range(N_SENSORS):
            # Keep topic names consistent with realtime_infer.py (no _tcn suffix).
            self.comp_pubs.append(self.create_publisher(Range, f"/compensated_raw_distance{i+1}", 10))
            self.pred_pubs.append(self.create_publisher(Range, f"/predicted_raw_distance{i+1}", 10))
            self.res_pubs.append(self.create_publisher(Range, f"/residual_raw_distance{i+1}", 10))

        log_dir = os.path.expanduser("~/rb10_Proximity/logs")
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(model_file))[0] if model_file else "unknown"
        self.log_path = os.path.join(log_dir, f"compensated_raw_tcn_{model_name}_{now}.txt")
        self.log_file = open(self.log_path, "w")
        header = (
            "# timestamp "
            + " ".join([f"j{i+1}" for i in range(N_JOINTS)]) + " "
            + " ".join([f"jv{i+1}" for i in range(N_JOINTS)]) + " "
            + " ".join([f"raw{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"pred{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"res{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"comp{i+1}" for i in range(N_SENSORS)])
            + "\n"
        )
        self.log_file.write(header)
        self.log_file.flush()

        self.timer = self.create_timer(1.0 / self.log_rate, self.timer_callback, callback_group=self.cb_group)

        self.get_logger().info("=" * 60)
        self.get_logger().info("Realtime Infer TCN (MLP main + TCN residual)")
        self.get_logger().info(f"Model: {model_file}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"seq_len: {self.seq_len}, warmup_zero_pad: {self.warmup_zero_pad}")
        self.get_logger().info(f"Log: {self.log_path}")
        self.get_logger().info("=" * 60)

    def _find_latest_model(self):
        files = find_available_models()
        return files[0] if files else None

    def _load_model(self, model_file: str, norm_file: str = None):
        if not os.path.isabs(model_file):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            for path in [
                os.path.join(base_dir, "train", "outputs", model_file),
                os.path.join(base_dir, "outputs", model_file),
                os.path.join(base_dir, model_file),
                model_file,
            ]:
                if os.path.exists(path):
                    model_file = path
                    break

        if not os.path.exists(model_file):
            self.get_logger().error(f"Model not found: {model_file}")
            return

        try:
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        except Exception as e:
            self.get_logger().error(f"Failed to load checkpoint: {e}")
            return

        if not is_tcn_model(model_file):
            self.get_logger().error("Selected checkpoint is not Method B (MLP+TCN) model.")
            return

        model_args = checkpoint.get("args", {})

        dilations = model_args.get("tcn_dilations", "1,2,4,8")
        if isinstance(dilations, str):
            dilations = tuple(int(x.strip()) for x in dilations.split(",") if x.strip())
        else:
            dilations = tuple(int(x) for x in dilations)

        ckpt_seq_len = int(model_args.get("seq_len", self.seq_len))
        if ckpt_seq_len != self.seq_len:
            self.get_logger().warn(
                f"seq_len mismatch (param={self.seq_len}, ckpt={ckpt_seq_len}). Using ckpt value."
            )
            self.seq_len = ckpt_seq_len
            self.x_seq_buffer = deque(maxlen=self.seq_len)

        self.model = MLP_TCN_ResidualModel(
            in_dim=int(model_args.get("in_dim", 12)),
            out_dim=int(model_args.get("out_dim", 8)),
            trunk_hidden=int(model_args.get("hidden", 128)),
            head_hidden=int(model_args.get("head_hidden", 64)),
            tcn_hidden=int(model_args.get("tcn_hidden", 64)),
            tcn_kernel=int(model_args.get("tcn_kernel", 3)),
            tcn_dilations=dilations,
            dropout=float(model_args.get("dropout", 0.1)),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        if norm_file and os.path.exists(norm_file):
            try:
                self.x_mean, self.x_std, self.y_mean, self.y_std, _ = load_norm_params(norm_file)
            except Exception as e:
                self.get_logger().warn(f"Failed to load norm file: {e}. Trying checkpoint.")
                norm_file = None

        if norm_file is None or not os.path.exists(norm_file):
            if "normalization" in checkpoint:
                norm = checkpoint["normalization"]
                self.x_mean = np.array(norm["X_mean"], dtype=np.float32)
                self.x_std = np.array(norm["X_std"], dtype=np.float32)
                self.y_mean = np.array(norm["Y_mean"], dtype=np.float32)
                self.y_std = np.array(norm["Y_std"], dtype=np.float32)
            else:
                model_dir = os.path.dirname(model_file)
                norm_path = os.path.join(model_dir, "norm_params.json")
                if not os.path.exists(norm_path):
                    self.get_logger().error("Normalization params not found.")
                    self.model = None
                    return
                self.x_mean, self.x_std, self.y_mean, self.y_std, _ = load_norm_params(norm_path)

    def raw_callback(self, msg, idx):
        self.raw_data[idx] = msg.range
        self.raw_received = True

    def joint_callback(self, msg):
        expected_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        name_to_idx = {name: idx for idx, name in enumerate(msg.name) if name in expected_names}

        joint_positions_rad = []
        joint_velocities_rad = []
        for name in expected_names:
            if name in name_to_idx:
                idx = name_to_idx[name]
                joint_positions_rad.append(msg.position[idx])
                if len(msg.velocity) > idx:
                    joint_velocities_rad.append(msg.velocity[idx])
                else:
                    joint_velocities_rad.append(0.0)
            else:
                joint_positions_rad.append(0.0)
                joint_velocities_rad.append(0.0)

        self.joint_positions = np.rad2deg(np.array(joint_positions_rad, dtype=np.float32))
        self.joint_velocities = np.rad2deg(np.array(joint_velocities_rad, dtype=np.float32))
        self.joint_received = True

    def _build_input_seq(self):
        if len(self.x_seq_buffer) >= self.seq_len:
            return np.array(list(self.x_seq_buffer), dtype=np.float32)

        if not self.warmup_zero_pad:
            return None

        seq = np.zeros((self.seq_len, len(self.x_mean)), dtype=np.float32)
        if len(self.x_seq_buffer) > 0:
            buf = np.array(list(self.x_seq_buffer), dtype=np.float32)
            seq[-len(buf):] = buf
        return seq

    def timer_callback(self):
        if not self.raw_received or not self.joint_received:
            return

        j_pos_rad = np.deg2rad(self.joint_positions)
        x = np.concatenate([np.sin(j_pos_rad), np.cos(j_pos_rad)], axis=0).astype(np.float32)
        x_norm = (x - self.x_mean) / self.x_std
        self.x_seq_buffer.append(x_norm.copy())

        x_seq = self._build_input_seq()
        if x_seq is None:
            return

        with torch.no_grad():
            x_seq_tensor = torch.from_numpy(x_seq).unsqueeze(0).to(self.device)
            y_hat_norm, _ = self.model(x_seq_tensor, use_residual=True)
            pred_norm = y_hat_norm.cpu().numpy().squeeze(0)

        y_pred = pred_norm * self.y_std + self.y_mean
        y_actual = self.raw_data.copy()
        y_residual = y_actual - y_pred

        if self.use_hardware_baseline:
            baseline = np.full(N_SENSORS, HARDWARE_BASELINE, dtype=np.float32)
        else:
            baseline = self.y_mean

        compensated = y_actual - y_pred + baseline

        now = self.get_clock().now()
        t = now.nanoseconds / 1e9
        line = (
            f"{t:.9f} "
            + " ".join(f"{v:.6f}" for v in self.joint_positions) + " "
            + " ".join(f"{v:.6f}" for v in self.joint_velocities) + " "
            + " ".join(f"{v:.6f}" for v in y_actual) + " "
            + " ".join(f"{v:.6f}" for v in y_pred) + " "
            + " ".join(f"{v:.6f}" for v in y_residual) + " "
            + " ".join(f"{v:.6f}" for v in compensated)
            + "\n"
        )
        self.log_file.write(line)
        self.log_file.flush()

        for i in range(N_SENSORS):
            msg_comp = Range()
            msg_comp.header.stamp = now.to_msg()
            msg_comp.range = float(compensated[i])
            msg_comp.radiation_type = Range.ULTRASOUND
            msg_comp.field_of_view = 0.1
            msg_comp.min_range = 0.0
            msg_comp.max_range = 100000.0
            self.comp_pubs[i].publish(msg_comp)

            msg_pred = Range()
            msg_pred.header.stamp = now.to_msg()
            msg_pred.range = float(y_pred[i])
            msg_pred.radiation_type = Range.ULTRASOUND
            msg_pred.field_of_view = 0.1
            msg_pred.min_range = 0.0
            msg_pred.max_range = 100000.0
            self.pred_pubs[i].publish(msg_pred)

            msg_res = Range()
            msg_res.header.stamp = now.to_msg()
            msg_res.range = float(y_residual[i])
            msg_res.radiation_type = Range.ULTRASOUND
            msg_res.field_of_view = 0.1
            msg_res.min_range = -100000.0
            msg_res.max_range = 100000.0
            self.res_pubs[i].publish(msg_res)

    def destroy_node(self):
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
        super().destroy_node()


def is_tcn_model(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", {})

        has_main = any(key.startswith("main.") for key in state_dict.keys())
        has_residual = any(key.startswith("residual.") for key in state_dict.keys())
        has_gru = any("res_gru" in key for key in state_dict.keys())
        has_trunk_only = any(key.startswith("trunk.") for key in state_dict.keys())

        if has_main and has_residual and not has_gru:
            return True

        args = checkpoint.get("args", {})
        if args.get("model_type") == "mlp_tcn_residual":
            return True
        if "tcn_hidden" in args and not has_gru:
            return True

        if checkpoint.get("model_type") == "mlp_tcn_residual":
            return True

        if has_trunk_only:
            return False
        return False
    except Exception:
        return False


def find_available_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    possible_dirs = [
        os.path.join(base_dir, "self_detection_raw", "train", "train", "outputs"),
        os.path.join(base_dir, "self_detection_raw", "train", "outputs"),
        os.path.join(base_dir, "train", "outputs"),
        os.path.join(base_dir, "outputs"),
    ]

    model_files = []
    for d in possible_dirs:
        if os.path.exists(d):
            for run_dir in sorted(glob.glob(os.path.join(d, "run_*")), reverse=True):
                model_path = os.path.join(run_dir, "model.pt")
                if os.path.exists(model_path) and is_tcn_model(model_path):
                    model_files.append(model_path)

    model_files = list(set(model_files))
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return model_files


def main(args=None):
    rclpy.init(args=args)

    temp_node = Node("_temp_param_reader")
    temp_node.declare_parameter("model_file", "")
    model_file_from_param = temp_node.get_parameter("model_file").value
    temp_node.destroy_node()
    rclpy.shutdown()

    if not model_file_from_param or not model_file_from_param.strip():
        model_files = find_available_models()
        if model_files:
            model_file_from_param = model_files[0]
            print("\n" + "=" * 60)
            print(f"[INFO] Auto-selected latest TCN model: {model_file_from_param}")
            print("=" * 60 + "\n")
            if args is None:
                args = []
            if "--ros-args" in args:
                args = args + ["-p", f"model_file:={model_file_from_param}"]
            else:
                args = args + ["--ros-args", "-p", f"model_file:={model_file_from_param}"]
        else:
            print("[ERROR] No TCN model found. Train first or pass model_file.")
            sys.exit(1)

    rclpy.init(args=args)
    node = RealtimeInferTCNNode()

    if hasattr(node, "_model_load_failed"):
        node.destroy_node()
        rclpy.shutdown()
        return

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
