#!/usr/bin/env python3
"""
Real-time Self Detection Compensation Node (V4: Hybrid Residual Architecture)
Based on infer_visualize.py compensation logic and realtime_monitor_mlp.py topic structure.

Model: HybridResidualModel (Main Stream MLP + Residual Stream GRU)
Input: sin/cos(joint_angles) (12 dim) - joint velocities removed
Output: raw1~raw8 baseline prediction (8 dim)

Compensation: compensated = actual - predicted + baseline
Baseline: 4e+7 (HARDWARE_BASELINE from main.c)
"""

import os
import sys
import glob
import numpy as np
from datetime import datetime
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState, Range

import torch

from self_detection_raw.models.mlp_gru_v4 import HybridResidualModel
from self_detection_raw.data.stats import load_norm_params

# =========================
# Constants
# =========================
N_JOINTS = 6
N_SENSORS = 8
HARDWARE_BASELINE = 4.0e+07  # From main.c: cvf.MeanData_Q1[0] = 4.00E+07


class RealtimeInferV4Node(Node):
    """Real-time self-detection compensation node (V4: Hybrid Residual Architecture)."""

    def __init__(self):
        super().__init__('realtime_infer_v4')

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('model_file', '')
        self.declare_parameter('norm_file', '')
        self.declare_parameter('use_vel', False)  # DEPRECATED: always False, joint velocities removed
        self.declare_parameter('use_hardware_baseline', True)
        self.declare_parameter('log_rate', 100.0)
        self.declare_parameter('seq_len', 10)  # Sequence length for residual stream

        model_file = self.get_parameter('model_file').value
        norm_file = self.get_parameter('norm_file').value
        self.use_vel = False  # Always False - joint velocities removed from input
        self.use_hardware_baseline = bool(self.get_parameter('use_hardware_baseline').value)
        log_rate = float(self.get_parameter('log_rate').value)
        self.seq_len = int(self.get_parameter('seq_len').value)

        self.cb_group = ReentrantCallbackGroup()

        # -------------------------
        # Data buffers
        # -------------------------
        self.raw_data = np.zeros(N_SENSORS, dtype=np.float32)
        self.joint_positions = None
        self.joint_velocities = None
        self.raw_received = False
        self.joint_received = False
        
        # Sequence buffer for residual stream (stores recent N frames)
        self.x_seq_buffer = deque(maxlen=self.seq_len)

        # -------------------------
        # Model and normalization
        # -------------------------
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and normalization
        if not model_file:
            # Try to find latest model automatically
            model_file = self._find_latest_model()
            if not model_file:
                self.get_logger().error("=" * 60)
                self.get_logger().error("ERROR: No V4 (HybridResidualModel) models found!")
                self.get_logger().error("")
                self.get_logger().error("Please train a V4 model first using:")
                self.get_logger().error("  python -m self_detection_raw.train.train1 --data_dir <path> --glob 'robot_data_*.txt'")
                self.get_logger().error("")
                self.get_logger().error("Or specify a V4 model file explicitly:")
                self.get_logger().error("  ros2 launch self_detection_raw realtime_infer_v4.launch.py model_file:=train/outputs/run_XXX/model.pt")
                self.get_logger().error("=" * 60)
                self._model_load_failed = True
                return
            else:
                self.get_logger().info(f"Auto-selected latest V4 model: {os.path.basename(os.path.dirname(model_file))}")

        self._load_model(model_file, norm_file)

        if self.model is None:
            self.get_logger().error("=" * 60)
            self.get_logger().error("ERROR: Model load failed. Node aborted.")
            self.get_logger().error("=" * 60)
            self._model_load_failed = True
            return

        # -------------------------
        # Subscribers
        # -------------------------
        for i in range(N_SENSORS):
            self.create_subscription(
                Range,
                f'/raw_distance{i+1}',
                lambda msg, idx=i: self.raw_callback(msg, idx),
                10,
                callback_group=self.cb_group
            )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10,
            callback_group=self.cb_group
        )

        # -------------------------
        # Publishers
        # -------------------------
        self.comp_pubs = []
        self.pred_pubs = []
        for i in range(N_SENSORS):
            # V4 (MLP+GRU) 토픽 이름: 기존 MLP와 구분
            self.comp_pubs.append(
                self.create_publisher(
                    Range,
                    f'/compensated_raw_distance_gru{i+1}',
                    10
                )
            )
            self.pred_pubs.append(
                self.create_publisher(
                    Range,
                    f'/predicted_raw_distance_gru{i+1}',
                    10
                )
            )

        # -------------------------
        # Logging
        # -------------------------
        log_dir = os.path.expanduser('~/rb10_Proximity/logs')
        os.makedirs(log_dir, exist_ok=True)

        # Extract model name for log filename
        model_name = os.path.splitext(os.path.basename(model_file))[0] if model_file else "unknown"
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'compensated_raw_v4_{model_name}_{now}.txt'
        self.log_path = os.path.join(log_dir, filename)
        self.log_file = open(self.log_path, 'w')

        header = (
            "# timestamp "
            + " ".join([f"j{i+1}" for i in range(N_JOINTS)]) + " "
            + " ".join([f"jv{i+1}" for i in range(N_JOINTS)]) + " "
            + " ".join([f"raw{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"comp{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"pred{i+1}" for i in range(N_SENSORS)]) + "\n"
        )
        self.log_file.write(header)
        self.log_file.flush()

        # -------------------------
        # Timer
        # -------------------------
        self.timer = self.create_timer(
            1.0 / log_rate,
            self.timer_callback,
            callback_group=self.cb_group
        )

        # Get model directory name for display
        model_dir = os.path.dirname(model_file)
        model_name = os.path.basename(model_dir) if model_dir else os.path.basename(model_file)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Real-time Self Detection Compensation (V4: Hybrid Residual)")
        self.get_logger().info(f"Selected Model: {model_name}")
        self.get_logger().info(f"Model Path: {model_file}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Sequence Length: {self.seq_len}")
        self.get_logger().info(f"Baseline: {HARDWARE_BASELINE:.2e} (hardware)" if self.use_hardware_baseline else f"Baseline: training data mean")
        self.get_logger().info(f"Log: {self.log_path}")
        self.get_logger().info("=" * 60)

    # ======================================================
    # Model loading
    # ======================================================
    def _find_latest_model(self):
        """Find the latest model file in outputs directories."""
        model_files = find_available_models()
        return model_files[0] if model_files else None
    
    def _load_model(self, model_file: str, norm_file: str = None):
        """Load model checkpoint and normalization parameters."""
        # Find model file
        if not os.path.isabs(model_file):
            # Try relative paths
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            possible_paths = [
                os.path.join(base_dir, 'train', 'outputs', model_file),
                os.path.join(base_dir, 'outputs', model_file),
                os.path.join(base_dir, model_file),
                model_file,
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_file = path
                    break

        if not os.path.exists(model_file):
            self.get_logger().error(f"Model not found: {model_file}")
            return

        # Load checkpoint
        try:
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        except Exception as e:
            self.get_logger().error(f"Failed to load checkpoint: {e}")
            return

        # Verify this is a V4 model
        if not is_v4_model(model_file):
            self.get_logger().error("=" * 60)
            self.get_logger().error(f"ERROR: Selected model is not a V4 (HybridResidualModel) model!")
            self.get_logger().error(f"Model file: {model_file}")
            self.get_logger().error("Please select a model trained with train1.py (V4 architecture)")
            self.get_logger().error("=" * 60)
            return

        # Load model architecture
        model_args = checkpoint.get('args', {})
        
        # Get model parameters from checkpoint
        in_dim = model_args.get('in_dim', 12)  # Default to 12 if not in args
        if in_dim is None:
            # Try to infer from model state dict
            # Check first layer weight shape
            first_layer_key = None
            for key in checkpoint['model_state_dict'].keys():
                if 'main_stream.0.weight' in key:
                    first_layer_key = key
                    break
            if first_layer_key:
                in_dim = checkpoint['model_state_dict'][first_layer_key].shape[1]
            else:
                in_dim = 12  # Default
        
        seq_len = model_args.get('seq_len', self.seq_len)
        main_hidden = model_args.get('main_hidden', 128)
        res_hidden = model_args.get('res_hidden', 64)
        out_dim = model_args.get('out_dim', 8)
        dropout = model_args.get('dropout', 0.1)
        res_scale = model_args.get('res_scale', 0.1)
        res_clip = model_args.get('res_clip', None)
        
        # Update seq_len if different from parameter
        if seq_len != self.seq_len:
            self.get_logger().warn(f"Model seq_len ({seq_len}) differs from parameter ({self.seq_len}). Using model seq_len.")
            self.seq_len = seq_len
            # Resize buffer
            self.x_seq_buffer = deque(maxlen=self.seq_len)
        
        self.model = HybridResidualModel(
            in_dim=in_dim,
            seq_len=seq_len,
            main_hidden=main_hidden,
            res_hidden=res_hidden,
            out_dim=out_dim,
            dropout=dropout,
            res_scale=res_scale,
            res_clip=res_clip
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.get_logger().info(f"Model loaded from: {model_file}")
        self.get_logger().info(f"Model architecture: in_dim={in_dim}, seq_len={seq_len}, main_hidden={main_hidden}, res_hidden={res_hidden}")

        # Load normalization parameters
        if norm_file and os.path.exists(norm_file):
            # Load from specified file
            try:
                self.x_mean, self.x_std, self.y_mean, self.y_std, _ = load_norm_params(norm_file)
                self.get_logger().info(f"Normalization params loaded from: {norm_file}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load norm file: {e}, trying checkpoint...")
                norm_file = None

        if norm_file is None or not os.path.exists(norm_file):
            # Try to load from checkpoint
            if 'normalization' in checkpoint:
                norm_params = checkpoint['normalization']
                self.x_mean = np.array(norm_params['X_mean'], dtype=np.float32)
                self.x_std = np.array(norm_params['X_std'], dtype=np.float32)
                self.y_mean = np.array(norm_params['Y_mean'], dtype=np.float32)
                self.y_std = np.array(norm_params['Y_std'], dtype=np.float32)
                self.get_logger().info("Normalization params loaded from checkpoint")
            else:
                # Try to find norm_params.json in model directory
                model_dir = os.path.dirname(model_file)
                default_norm_path = os.path.join(model_dir, 'norm_params.json')
                if os.path.exists(default_norm_path):
                    try:
                        self.x_mean, self.x_std, self.y_mean, self.y_std, _ = load_norm_params(default_norm_path)
                        self.get_logger().info(f"Normalization params loaded from: {default_norm_path}")
                    except Exception as e:
                        self.get_logger().error(f"Failed to load norm params: {e}")
                        return
                else:
                    self.get_logger().error("Normalization params not found!")
                    return

        self.get_logger().info(f"X_mean shape: {self.x_mean.shape}, Y_mean shape: {self.y_mean.shape}")

    # ======================================================
    # Callbacks
    # ======================================================
    def raw_callback(self, msg, idx):
        """Raw sensor callback."""
        self.raw_data[idx] = msg.range
        self.raw_received = True

    def joint_callback(self, msg):
        """Joint state callback.
        
        IMPORTANT: JointState.position is in RADIANS (ROS2 standard).
        Training data uses DEGREES, so we need to convert.
        Also, JointState order is not guaranteed - must use name mapping.
        """
        # Expected joint names (same order as training data)
        expected_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        # Create name->index mapping
        name_to_idx = {name: idx for idx, name in enumerate(msg.name) if name in expected_names}
        
        # Extract joint positions in correct order (convert radian -> degree)
        # ROS2 JointState.position is in radians, but training data uses degrees
        joint_positions_rad = []
        joint_velocities_rad = []
        
        for name in expected_names:
            if name in name_to_idx:
                idx = name_to_idx[name]
                joint_positions_rad.append(msg.position[idx])
                if self.use_vel and len(msg.velocity) > idx:
                    joint_velocities_rad.append(msg.velocity[idx])
                else:
                    joint_velocities_rad.append(0.0)
            else:
                self.get_logger().warn(f"Joint '{name}' not found in JointState. Available: {msg.name}")
                joint_positions_rad.append(0.0)
                joint_velocities_rad.append(0.0)
        
        # Convert radian -> degree (to match training data format)
        self.joint_positions = np.rad2deg(np.array(joint_positions_rad, dtype=np.float32))
        self.joint_velocities = np.rad2deg(np.array(joint_velocities_rad, dtype=np.float32))
        
        self.joint_received = True

    # ======================================================
    # Inference + compensation
    # ======================================================
    def timer_callback(self):
        """Timer callback for inference and compensation."""
        if not self.raw_received or not self.joint_received:
            return

        # Extract features (same as extract_features in loader.py)
        # Joint positions: degrees -> radians -> sin/cos
        # NOTE: self.joint_positions is already in DEGREES (converted from rad in callback)
        j_pos_rad = np.deg2rad(self.joint_positions)
        sin_j = np.sin(j_pos_rad)
        cos_j = np.cos(j_pos_rad)

        # Concatenate: [sin, cos] -> (12,) - joint velocities removed
        x_current = np.concatenate([sin_j, cos_j], axis=0).astype(np.float32)

        # Normalize input
        x_current_norm = (x_current - self.x_mean) / self.x_std
        
        # Add to sequence buffer
        self.x_seq_buffer.append(x_current_norm.copy())
        
        # Check if we have enough frames for residual stream
        if len(self.x_seq_buffer) < self.seq_len:
            # Not enough frames yet, use only main stream (residual = 0)
            x_seq_norm = None
        else:
            # Convert buffer to array for residual stream
            x_seq_norm = np.array(list(self.x_seq_buffer), dtype=np.float32)  # (seq_len, in_dim)

        # Model inference
        with torch.no_grad():
            x_current_tensor = torch.from_numpy(x_current_norm).unsqueeze(0).to(self.device)  # (1, in_dim)
            
            if x_seq_norm is not None:
                x_seq_tensor = torch.from_numpy(x_seq_norm).unsqueeze(0).to(self.device)  # (1, seq_len, in_dim)
            else:
                x_seq_tensor = None
            
            pred_norm = self.model(x_current_tensor, x_seq_tensor).cpu().numpy().squeeze(0)

        # Denormalize output (CRITICAL: must denormalize to get raw scale)
        Y_pred = pred_norm * self.y_std + self.y_mean

        # Get actual raw values
        Y_actual = self.raw_data.copy()

        # Compute compensated values
        # compensated = actual - predicted + baseline
        if self.use_hardware_baseline:
            baseline = np.full(N_SENSORS, HARDWARE_BASELINE, dtype=np.float32)
        else:
            baseline = self.y_mean

        compensated = Y_actual - Y_pred + baseline

        # Logging
        now = self.get_clock().now()
        t = now.nanoseconds / 1e9

        line = (
            f"{t:.9f} "
            + " ".join(f"{x:.6f}" for x in self.joint_positions) + " "
            + " ".join(f"{x:.6f}" for x in self.joint_velocities) + " "
            + " ".join(f"{x:.6f}" for x in Y_actual) + " "
            + " ".join(f"{x:.6f}" for x in compensated) + " "
            + " ".join(f"{x:.6f}" for x in Y_pred)
            + "\n"
        )
        self.log_file.write(line)
        self.log_file.flush()

        # Publish compensated values
        for i in range(N_SENSORS):
            msg = Range()
            msg.header.stamp = now.to_msg()
            msg.range = float(compensated[i])
            msg.radiation_type = Range.ULTRASOUND
            msg.field_of_view = 0.1
            msg.min_range = 0.0
            msg.max_range = 100000.0
            self.comp_pubs[i].publish(msg)
        
        # Publish predicted values
        for i in range(N_SENSORS):
            msg = Range()
            msg.header.stamp = now.to_msg()
            msg.range = float(Y_pred[i])
            msg.radiation_type = Range.ULTRASOUND
            msg.field_of_view = 0.1
            msg.min_range = 0.0
            msg.max_range = 100000.0
            self.pred_pubs[i].publish(msg)

    def destroy_node(self):
        """Cleanup on node destruction."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
        super().destroy_node()


def is_v4_model(checkpoint_path):
    """Check if checkpoint is a V4 (HybridResidualModel) model."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Check model state dict keys for V4 model structure
        state_dict = checkpoint.get('model_state_dict', {})
        
        # V4 model has: main_stream, res_gru, res_proj
        # ModelB has: trunk, heads
        has_main_stream = any('main_stream' in key for key in state_dict.keys())
        has_res_gru = any('res_gru' in key for key in state_dict.keys())
        has_trunk = any('trunk' in key for key in state_dict.keys())
        
        # V4 model should have main_stream and res_gru, but not trunk
        if has_main_stream and has_res_gru and not has_trunk:
            return True
        
        # Also check args for model type hint
        args = checkpoint.get('args', {})
        if 'seq_len' in args or 'main_hidden' in args or 'res_hidden' in args:
            return True
        
        return False
    except Exception as e:
        # If we can't check, assume it's not V4 (safer)
        return False


def find_available_models():
    """Find all available V4 model files in outputs directories.
    
    Priority: train/outputs/ > outputs/
    Only returns V4 (HybridResidualModel) models.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check multiple possible outputs directories (train/outputs/ has priority)
    # Possible locations:
    # 1. self_detection_raw/train/train/outputs/ (when train1.py is run from package root)
    # 2. self_detection_raw/train/outputs/ (when train1.py is run from self_detection_raw/train/)
    # 3. train/outputs/ (when train1.py is run from package root with relative path)
    # 4. outputs/ (package root, for backward compatibility)
    
    possible_dirs = [
        os.path.join(base_dir, 'self_detection_raw', 'train', 'train', 'outputs'),  # train1.py from package root
        os.path.join(base_dir, 'self_detection_raw', 'train', 'outputs'),  # train1.py from self_detection_raw/train/
        os.path.join(base_dir, 'train', 'outputs'),  # train/outputs/ relative path
        os.path.join(base_dir, 'outputs'),  # outputs/ (backward compatibility)
    ]
    
    model_files = []
    all_model_files = []  # For debugging
    
    # Check all possible directories
    for train_outputs_dir in possible_dirs:
        if os.path.exists(train_outputs_dir):
            for run_dir in sorted(glob.glob(os.path.join(train_outputs_dir, 'run_*')), reverse=True):
                model_path = os.path.join(run_dir, 'model.pt')
                if os.path.exists(model_path):
                    all_model_files.append(model_path)
                    if is_v4_model(model_path):
                        model_files.append(model_path)
    
    # Debug: if no V4 models found but other models exist, print info
    if not model_files and all_model_files:
        print("\n" + "=" * 60)
        print("[WARNING] Found model files but none are V4 (HybridResidualModel) models:")
        for mf in all_model_files[:5]:  # Show first 5
            print(f"  - {os.path.relpath(mf, base_dir)}")
        if len(all_model_files) > 5:
            print(f"  ... and {len(all_model_files) - 5} more")
        print("\nPlease train a V4 model using train1.py")
        print("=" * 60 + "\n")
    
    # Remove duplicates and sort by modification time (newest first)
    model_files = list(set(model_files))
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return model_files


def select_model_interactively():
    """Interactively select a model file from available options."""
    model_files = find_available_models()
    
    if not model_files:
        print("\n" + "=" * 60)
        print("[ERROR] No model files found in outputs directory")
        print("[ERROR] Please train a model first or specify --ros-args -p model_file:=/path/to/model.pt")
        print("=" * 60 + "\n")
        return None
    
    print("\n" + "=" * 60)
    print("[INFO] Model file not specified. Please select a model:")
    print("=" * 60)
    print(f"\nAvailable models in outputs directory:")
    for i, f in enumerate(model_files):
        filename = os.path.basename(os.path.dirname(f))  # run_YYYYMMDD_HHMMSS
        file_size = os.path.getsize(f) / (1024 * 1024)  # MB
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        print(f"  [{i}] {filename}/model.pt ({file_size:.2f} MB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"  [{len(model_files)}] Cancel (exit)")
    
    while True:
        try:
            choice = input(f"\nSelect model [0-{len(model_files)}] (default: 0): ").strip()
            if choice == '':
                choice = '0'
            
            choice_idx = int(choice)
            
            if choice_idx == len(model_files):
                print("[INFO] Node launch cancelled by user.")
                return None
            
            if 0 <= choice_idx < len(model_files):
                selected_model = model_files[choice_idx]
                print(f"[INFO] Selected model: {os.path.relpath(selected_model)}")
                print("=" * 60 + "\n")
                return selected_model
            else:
                print(f"[ERROR] Invalid choice. Please enter a number between 0 and {len(model_files)}.")
        except ValueError:
            print(f"[ERROR] Invalid input. Please enter a number between 0 and {len(model_files)}.")
        except KeyboardInterrupt:
            print("\n[INFO] Node launch cancelled by user (Ctrl+C).")
            return None


def parse_ros_args_for_model(args):
    """Parse ROS2 arguments to extract model_file parameter."""
    if args is None:
        return None
    
    model_file = None
    i = 0
    while i < len(args):
        arg = args[i]
        # Direct parameter format: model_file:=path
        if arg.startswith('model_file:='):
            model_file = arg.split(':=', 1)[1]
            break
        # Parameter with -p flag: -p model_file:=path
        elif arg == '-p' and i + 1 < len(args):
            next_arg = args[i + 1]
            if next_arg.startswith('model_file:='):
                model_file = next_arg.split(':=', 1)[1]
                break
        i += 1
    
    return model_file


def main(args=None):
    """Main entry point."""
    # Initialize ROS2 first to read parameters
    rclpy.init(args=args)
    
    # Create a temporary node to read the model_file parameter
    temp_node = Node('_temp_param_reader')
    temp_node.declare_parameter('model_file', '')
    model_file_from_param = temp_node.get_parameter('model_file').value
    temp_node.destroy_node()
    rclpy.shutdown()
    
    # If model file not specified via parameter, auto-select latest model
    if not model_file_from_param or not model_file_from_param.strip():
        model_files = find_available_models()
        if model_files:
            model_file_from_param = model_files[0]
            model_dir = os.path.dirname(model_file_from_param)
            model_name = os.path.basename(model_dir) if model_dir else os.path.basename(model_file_from_param)
            print("\n" + "=" * 60)
            print(f"[INFO] Auto-selected latest V4 model: {model_name}")
            print(f"[INFO] Model path: {model_file_from_param}")
            print("=" * 60 + "\n")
            # Add model_file parameter to ROS2 args
            if args is None:
                args = []
            if '--ros-args' in args:
                args = args + ['-p', f'model_file:={model_file_from_param}']
            else:
                args = args + ['--ros-args', '-p', f'model_file:={model_file_from_param}']
        else:
            print("\n" + "=" * 60)
            print("[ERROR] No V4 (HybridResidualModel) models found!")
            print("")
            print("Please train a V4 model first using:")
            print("  python -m self_detection_raw.train.train1 --data_dir <path> --glob 'robot_data_*.txt'")
            print("")
            print("Or specify a V4 model file explicitly:")
            print("  ros2 launch self_detection_raw realtime_infer_v4.launch.py model_file:=train/outputs/run_XXX/model.pt")
            print("=" * 60 + "\n")
            sys.exit(1)
    else:
        # Model was specified, show which one
        model_dir = os.path.dirname(model_file_from_param)
        model_name = os.path.basename(model_dir) if model_dir else os.path.basename(model_file_from_param)
        print("\n" + "=" * 60)
        print(f"[INFO] Using specified model: {model_name}")
        print(f"[INFO] Model path: {model_file_from_param}")
        print("=" * 60 + "\n")
    
    # Re-initialize ROS2 with the model_file parameter
    rclpy.init(args=args)
    node = RealtimeInferV4Node()

    if hasattr(node, '_model_load_failed'):
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
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()

