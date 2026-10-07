"""Open the MJLab-configured Unitree H1-2 with a ground plane."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assets.robots.unitree_h1_2.h1_2_constants import get_h1_2_robot_cfg

from _view_mujoco_scene import view_scene


if __name__ == "__main__":
    view_scene(get_h1_2_robot_cfg, "H1-2")
