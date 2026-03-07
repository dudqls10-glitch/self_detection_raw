#!/usr/bin/env python3
"""
Launch file for real-time self-detection compensation node (V4: Hybrid Residual Architecture).

Usage:
    ros2 launch self_detection_raw realtime_infer_v4.launch.py
    ros2 launch self_detection_raw realtime_infer_v4.launch.py model_file:=train/outputs/run_001/model.pt
    ros2 launch self_detection_raw realtime_infer_v4.launch.py model_file:=train/outputs/run_001/model.pt norm_file:=train/outputs/run_001/norm_params.json
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import os
import glob
from datetime import datetime


def find_available_models(package_dir):
    """Find all available model files in outputs directories."""
    # Check multiple possible outputs directories (train/outputs/ has priority)
    possible_outputs_dirs = [
        os.path.join(package_dir, 'self_detection_raw', 'train', 'outputs'),  # train/outputs/ (priority)
        os.path.join(package_dir, 'outputs'),  # outputs/ (package root)
    ]
    
    all_model_files = []
    for outputs_dir in possible_outputs_dirs:
        if os.path.exists(outputs_dir):
            model_files = glob.glob(os.path.join(outputs_dir, '**', 'model.pt'), recursive=True)
            all_model_files.extend(model_files)
    
    # Remove duplicates and sort by modification time (newest first)
    all_model_files = list(set(all_model_files))
    all_model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return all_model_files


def select_model_interactively(package_dir):
    """Interactively select a model file from available options."""
    model_files = find_available_models(package_dir)
    
    if not model_files:
        print("\n" + "=" * 60)
        print("[ERROR] No model files found in outputs directory")
        print("[ERROR] Please train a model first or specify model_file parameter")
        print("=" * 60 + "\n")
        return None
    
    print("\n" + "=" * 60)
    print("[INFO] Please select a model:")
    print("=" * 60)
    print(f"\nAvailable models:")
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
                print("[INFO] Launch cancelled by user.")
                return None
            
            if 0 <= choice_idx < len(model_files):
                selected_model = model_files[choice_idx]
                print(f"[INFO] Selected model: {os.path.relpath(selected_model, package_dir)}")
                print("=" * 60 + "\n")
                return selected_model
            else:
                print(f"[ERROR] Invalid choice. Please enter a number between 0 and {len(model_files)}.")
        except ValueError:
            print(f"[ERROR] Invalid input. Please enter a number between 0 and {len(model_files)}.")
        except KeyboardInterrupt:
            print("\n[INFO] Launch cancelled by user (Ctrl+C).")
            return None


def generate_launch_description():
    # Get package directory (absolute path)
    package_dir = '/home/son_rb/rb_ws/src/self_detection_raw'
    
    # Paths
    script_path = os.path.join(package_dir, 'self_detection_raw', 'infer', 'realtime_infer_v4.py')
    venv_python = os.path.join(package_dir, 'venv1', 'bin', 'python3')
    
    # Use venv Python if available, otherwise use system Python
    python_exec = venv_python if os.path.exists(venv_python) else 'python3'
    
    # Set PYTHONPATH
    venv_site_packages = os.path.join(package_dir, 'venv1', 'lib', 'python3.10', 'site-packages')
    pythonpath = f"{venv_site_packages}:{package_dir}"
    if 'PYTHONPATH' in os.environ:
        pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"
    
    def launch_setup(context):
        actions = []
        
        # Get launch argument values
        model_file = context.launch_configurations.get('model_file', '')
        norm_file = context.launch_configurations.get('norm_file', '')
        use_vel = context.launch_configurations.get('use_vel', 'false')
        use_hardware_baseline = context.launch_configurations.get('use_hardware_baseline', 'true')
        log_rate = context.launch_configurations.get('log_rate', '100.0')
        seq_len = context.launch_configurations.get('seq_len', '10')
        
        # If model_file is not specified or empty, let the node handle selection
        # (Launch file context may not have stdin available)
        # The node will handle interactive selection if needed
        
        # Build command
        cmd = [
            python_exec,
            script_path,
            '--ros-args',
            '-r', '__node:=realtime_infer_v4',
        ]
        
        # Add parameters (only if not empty)
        if model_file and model_file.strip():
            cmd.extend(['-p', f'model_file:={model_file}'])
        
        if norm_file and norm_file.strip():
            cmd.extend(['-p', f'norm_file:={norm_file}'])
        
        cmd.extend(['-p', f'use_vel:={use_vel}'])
        cmd.extend(['-p', f'use_hardware_baseline:={use_hardware_baseline}'])
        cmd.extend(['-p', f'log_rate:={log_rate}'])
        cmd.extend(['-p', f'seq_len:={seq_len}'])
        
        # Set environment
        actions.append(SetEnvironmentVariable('PYTHONPATH', pythonpath))
        
        # Execute node
        actions.append(ExecuteProcess(
            cmd=cmd,
            output='screen',
        ))
        
        return actions
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'model_file',
            default_value='',
            description='Path to model checkpoint (.pt file). If empty, will show interactive selection'
        ),
        DeclareLaunchArgument(
            'norm_file',
            default_value='',
            description='Path to normalization params (.json). If empty, will try to load from checkpoint or model directory.'
        ),
        DeclareLaunchArgument(
            'use_vel',
            default_value='false',
            description='Use joint velocities (true/false) - DEPRECATED: always false'
        ),
        DeclareLaunchArgument(
            'use_hardware_baseline',
            default_value='true',
            description='Use hardware baseline (4e+7) for compensation (true/false)'
        ),
        DeclareLaunchArgument(
            'log_rate',
            default_value='100.0',
            description='Logging rate in Hz'
        ),
        DeclareLaunchArgument(
            'seq_len',
            default_value='10',
            description='Sequence length for residual stream (should match training)'
        ),
        
        # Use OpaqueFunction to conditionally add parameters
        OpaqueFunction(function=launch_setup),
    ])

