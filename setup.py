from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'self_detection_raw'

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
    ],
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.10.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": ["pandas>=1.3.0", "matplotlib>=3.3.0"],
    },
    python_requires=">=3.10",
    maintainer='song',
    maintainer_email='dudqls10@g.skku.edu',
    description='Self-detection raw baseline compensation using Model B',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        "console_scripts": [
            "self-detection-train=self_detection_raw.train.train:main",
            "self-detection-train-tcn=self_detection_raw.train.train_tcn:main",
            "self-detection-eval=self_detection_raw.train.eval:main",
            "self-detection-infer=self_detection_raw.infer.infer:main",
            "self-detection-infer-tcn=self_detection_raw.infer.infer_tcn:main",
            "realtime_infer=self_detection_raw.infer.realtime_infer:main",
            "realtime_infer_base_corr_explicit=self_detection_raw.infer.realtime_infer_base_corr_explicit:main",
            "realtime_infer_prev_sensor=self_detection_raw.infer.realtime_infer_prev_sensor:main",
            "realtime_infer_v4=self_detection_raw.infer.realtime_infer_v4:main",
            "realtime_infer_tcn=self_detection_raw.infer.realtime_infer_tcn:main",
        ],
    },
    zip_safe=True,
)
