"""MJLab configuration for the H1-2 with an OSL right leg."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets


MODEL_XML = Path(__file__).resolve().parent / "h1_with_osl.xml"


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    assets: dict[str, bytes] = {}
    update_assets(assets, MODEL_XML.parent / "assets", spec.meshdir)
    spec.assets = assets
    return spec


def use_h1_with_osl_spec(h1_cfg: EntityCfg) -> EntityCfg:
    """Retarget a fresh H1-2 config while preserving its actuator settings."""
    if h1_cfg.articulation is None:
        raise ValueError("H1-2 configuration has no articulation settings")

    actuators = []
    for actuator in h1_cfg.articulation.actuators:
        if actuator.target_names_expr == (".*_knee.*",):
            # Drive the OSL knee with the H1 knee position-controller settings.
            actuator = replace(
                actuator,
                target_names_expr=("left_knee_joint", "osl_knee_angle_r"),
            )
        elif ".*_ankle_pitch.*" in actuator.target_names_expr:
            # The OSL has one ankle DOF; use the H1 ankle-pitch PD settings.
            actuator = replace(
                actuator,
                target_names_expr=actuator.target_names_expr + ("osl_ankle_angle_r",),
            )
        actuators.append(actuator)

    h1_cfg.spec_fn = get_spec
    h1_cfg.articulation = replace(h1_cfg.articulation, actuators=tuple(actuators))
    h1_cfg.init_state = replace(
        h1_cfg.init_state,
        joint_pos={
            **h1_cfg.init_state.joint_pos,
            # Match the removed H1 right-knee and ankle-pitch home pose.
            "osl_knee_angle_r": 0.5,
            # The OSL ankle axis is opposite to the H1 ankle-pitch axis.
            "osl_ankle_angle_r": 0.3,
        },
    )
    return h1_cfg
