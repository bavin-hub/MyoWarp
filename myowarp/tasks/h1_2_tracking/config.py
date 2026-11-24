"""Intact H1-2 configuration; reuses the unchanged MJLab tracking/PPO recipe."""

from dataclasses import dataclass
from pathlib import Path

from myowarp.tasks.h1_with_osl_tracking.config import (
    H1WithOslTrackingCfg,
)
from myowarp.tasks.h1_with_osl_tracking.config import (
    make_rsl_rl_cfg as _base_agent_cfg,
)

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class H1TrackingCfg(H1WithOslTrackingCfg):
    model_path: str = str(ROOT / "assets/robots/unitree_h1_2/xmls/h1_2.xml")
    motion_path: str = str(ROOT / "assets/reference_clips/walk1_subject1_h1_2.npz")


def make_rsl_rl_cfg(**kwargs):
    cfg = _base_agent_cfg(**kwargs)
    cfg["experiment_name"] = "h1_2_tracking_standalone"
    return cfg


def config_from_dict(values):
    """Restore nested configuration as dataclasses rather than plain dicts."""
    from myowarp.tasks.h1_with_osl_tracking.config import (
        MotionCommandCfg,
        TrackingDomainRandomizationCfg,
        TrackingObservationNoiseCfg,
    )

    values = dict(values)
    for key, cls in (
        ("observation_noise", TrackingObservationNoiseCfg),
        ("domain_randomization", TrackingDomainRandomizationCfg),
        ("motion", MotionCommandCfg),
    ):
        if key in values:
            values[key] = cls(**values[key])
    return H1TrackingCfg(**values)
