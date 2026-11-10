#!/usr/bin/env python3
"""Check external installations, built workspaces, CUDA and portable inputs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()
    for name in ('ISAAC_SIM_ROOT', 'ROS_SETUP', 'QWEN_MODEL_PATH', 'RACER_CRPO_PYTHON', 'SIONNA_RUNTIME_DIR'):
        value = os.environ.get(name)
        if not value or not Path(value).exists():
            raise SystemExit(f'Missing {name}: configure env.local.sh')
    required = [Path(os.environ['ISAAC_SIM_ROOT']) / 'python.sh',
                Path(os.environ['QWEN_MODEL_PATH']) / 'config.json',
                ROOT / 'final_racer/data/gt_occupied_voxels.txt',
                ROOT / 'final_racer/assets/warehouse_scenes/isaac/warehouse_full_with_industrial_ap.usda',
                ROOT / 'final_racer/assets/sionna_scene/warehouse.xml']
    if not args.skip_build:
        required.extend([ROOT / 'final_racer/ros2_ws/install/setup.bash',
                         ROOT / 'final_racer_training/training_overlay_ws/install/setup.bash'])
    for path in required:
        if not path.is_file():
            raise SystemExit(f'Missing required file: {path}')
    from agentic_crpo.config import load_config, require_reference_input
    config = load_config(ROOT / 'final_racer_training/config/train.yaml')
    assert isinstance(config['actor']['mask_no_link_actions'], bool)
    print('HARD_NO_LINK_MASK', config['actor']['mask_no_link_actions'])
    print('NO_LINK_LOGIT_SCHEDULE', config['actor']['no_link_logit_schedule'])
    assert config['actor']['no_link_logit_penalty'] == 0
    try:
        require_reference_input(config)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    import torch
    import vllm
    import stable_baselines3
    if not torch.cuda.is_available():
        raise SystemExit('Training Python cannot access CUDA')
    for name in ('torch', 'vllm', 'transformers', 'stable-baselines3', 'gymnasium'):
        print(name, importlib.metadata.version(name))
    print('GPU', torch.cuda.get_device_name(0))
    # Sionna runs alongside ROS and must use the system/ROS Python ABI.
    env = os.environ.copy()
    env['PYTHONPATH'] = os.environ['SIONNA_RUNTIME_DIR']
    subprocess.run(['/usr/bin/python3', '-c',
        'import sionna.rt, mitsuba, drjit; print("SIONNA_IMPORT_OK", sionna.rt.__file__)'],
        env=env, check=True)
    subprocess.run(['bash', '-c',
        'set +u; source "$ROS_SETUP"; exec /usr/bin/python3 -c "$1"',
        'racer-ros-check', 'import rclpy; print("ROS_IMPORT_OK")'],
        env=os.environ.copy(), check=True)
    subprocess.run([str(Path(os.environ['ISAAC_SIM_ROOT']) / 'python.sh'),
                    str(ROOT / 'scripts/check_scene.py')], check=True)
    print('PORTABLE_ENVIRONMENT_OK', ROOT)


if __name__ == '__main__':
    main()
