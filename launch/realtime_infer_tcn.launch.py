#!/usr/bin/env python3
"""Launch for realtime TCN infer node."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable


def generate_launch_description():
    package_dir = '/home/son_rb/rb_ws/src/self_detection_raw'

    script_path = os.path.join(package_dir, 'self_detection_raw', 'infer', 'realtime_infer_tcn.py')
    venv_python = os.path.join(package_dir, 'venv1', 'bin', 'python3')
    python_exec = venv_python if os.path.exists(venv_python) else 'python3'

    venv_site_packages = os.path.join(package_dir, 'venv1', 'lib', 'python3.10', 'site-packages')
    pythonpath = f"{venv_site_packages}:{package_dir}"
    if 'PYTHONPATH' in os.environ:
        pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"

    def launch_setup(context):
        model_file = context.launch_configurations.get('model_file', '')
        norm_file = context.launch_configurations.get('norm_file', '')
        use_hardware_baseline = context.launch_configurations.get('use_hardware_baseline', 'true')
        log_rate = context.launch_configurations.get('log_rate', '100.0')
        seq_len = context.launch_configurations.get('seq_len', '32')
        warmup_zero_pad = context.launch_configurations.get('warmup_zero_pad', 'true')

        cmd = [
            python_exec,
            script_path,
            '--ros-args',
            '-r', '__node:=realtime_infer_tcn',
        ]

        if model_file and model_file.strip():
            cmd.extend(['-p', f'model_file:={model_file}'])
        if norm_file and norm_file.strip():
            cmd.extend(['-p', f'norm_file:={norm_file}'])

        cmd.extend(['-p', f'use_hardware_baseline:={use_hardware_baseline}'])
        cmd.extend(['-p', f'log_rate:={log_rate}'])
        cmd.extend(['-p', f'seq_len:={seq_len}'])
        cmd.extend(['-p', f'warmup_zero_pad:={warmup_zero_pad}'])

        return [
            SetEnvironmentVariable('PYTHONPATH', pythonpath),
            ExecuteProcess(cmd=cmd, output='screen'),
        ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_file',
            default_value='',
            description='Path to TCN model checkpoint (.pt). If empty, latest model is selected automatically.',
        ),
        DeclareLaunchArgument(
            'norm_file',
            default_value='',
            description='Path to normalization params (.json). If empty, tries checkpoint/model directory.',
        ),
        DeclareLaunchArgument(
            'use_hardware_baseline',
            default_value='true',
            description='Use hardware baseline (4e+7) for compensation.',
        ),
        DeclareLaunchArgument(
            'log_rate',
            default_value='100.0',
            description='Inference/log timer rate in Hz.',
        ),
        DeclareLaunchArgument(
            'seq_len',
            default_value='32',
            description='Sequence length. Should match trained model.',
        ),
        DeclareLaunchArgument(
            'warmup_zero_pad',
            default_value='true',
            description='If true, zero-pad during warmup before seq_len frames are filled.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
