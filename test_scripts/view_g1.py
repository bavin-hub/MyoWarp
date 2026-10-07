"""Open the MJLab-configured Unitree G1 with a ground plane."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assets.robots.unitree_g1.g1_constants import get_g1_robot_cfg

from _view_mujoco_scene import view_scene


if __name__ == "__main__":
    view_scene(get_g1_robot_cfg, "G1")
