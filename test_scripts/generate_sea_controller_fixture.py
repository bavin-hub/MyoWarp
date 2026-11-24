"""Development fixture generation from the original scalar controller, never used at runtime."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(
    description="Regenerate recorded scalar CMU-controller parity fixture; development only."
)
parser.add_argument("--cmu-source", type=Path, default=ROOT.parent / "cmu_leg/src")
args = parser.parse_args()
sys.path.insert(0, str(args.cmu_source.resolve()))
from cmu_leg.control.low_level_control import LowLevelControl
from cmu_leg.structures.commands import TorqueDesired

T, N = 1600, 3
rng = np.random.default_rng(254)
time = np.arange(T) * 0.001
q = np.zeros((T, N, 5))
v = np.zeros_like(q)
for i in range(N):
    q[:, i, 0] = 0.7 + 0.95 * np.sin(2 * np.pi * (0.8 + i * 0.2) * time)
    q[:, i, 1] = q[:, i, 0] + 0.03 * np.sin(11 * time + i)
    q[:, i, 2] = 0.6 * np.sin(5 * time + i)
    q[:, i, 4] = 0.5 * np.sin(5 * time + i)
    q[:, i, 3] = q[:, i, 4] + 0.04 * np.sin(7 * time + i)
    v[:, i] = np.gradient(q[:, i], 0.001, axis=0)
u = rng.uniform(-160, 160, (T // 10, N, 2)).repeat(10, 0)
enabled = np.ones((T, N), bool)
actuate = enabled.copy()
standing = np.zeros_like(enabled)
enabled[1100:1130, 1] = False
actuate[1250:1300, 0] = False
standing[600:850, 2] = True
# World 2 starts as already-enabled mid-walk (ProSim reset).
cs = [LowLevelControl() for _ in range(N)]
cs[2]._prev_control_enabled = True
cs[2]._post_enable_ticks = 1000
out = np.zeros((T, N, 2))
vel = out.copy()
conditioned = out.copy()
total = out.copy()
integral = np.zeros((N, 2))
# Same kv/limits as the current combined model, read from actual XML.
import mujoco

m = mujoco.MjModel.from_xml_path(
    str(ROOT / "assets/robots/myofullbody_cmu_leg/myofullbody_cmu_leg.xml")
)
kv = np.array([m.actuator(n).gainprm[0] for n in ("knee_vel", "ankle_vel")])
limits = np.array([m.actuator(n).forcerange[1] for n in ("knee_act", "ankle_act")])
ki = kv * 2 * np.pi * 16.8
for t in range(T):
    for i, c in enumerate(cs):
        qq, vv = q[t, i], v[t, i]
        s = SimpleNamespace(
            knee_angle=qq[0],
            knee_motor_angle=qq[1],
            ankle_angle=qq[2],
            ankle_motor_angle=qq[3],
            ankle_spring_angle=qq[4],
            knee_velocity=vv[0],
            knee_motor_velocity=vv[1],
            ankle_velocity=vv[2],
            ankle_motor_velocity=vv[3],
            ankle_spring_velocity=vv[4],
            knee_torque_measured=1666 * (qq[1] - qq[0]),
            ankle_torque_measured=993.8613 * (qq[3] - qq[4]),
            standing=standing[t, i],
        )
        o = c.compute(
            s, TorqueDesired(*u[t, i]), bool(enabled[t, i]), 0.001, bool(actuate[t, i])
        )
        out[t, i] = o.knee_motor_torque_ff, o.ankle_motor_torque_ff
        vel[t, i] = o.knee_velocity_commanded, o.ankle_velocity_commanded
        conditioned[t, i] = o.knee_torque_commanded, o.ankle_torque_commanded
        if enabled[t, i] and actuate[t, i]:
            integral[i] = np.clip(
                integral[i] + ki * (vel[t, i] - vv[[1, 3]]) * 0.001, -limits, limits
            )
            total[t, i] = np.clip(out[t, i] + integral[i], -limits, limits)
        else:
            integral[i] = 0
p = ROOT / "tests/fixtures"
p.mkdir(exist_ok=True)
np.savez_compressed(
    p / "sea_controller_oracle.npz",
    q=q,
    v=v,
    desired=u,
    enabled=enabled,
    actuate=actuate,
    standing=standing,
    raw_ff=out,
    velocity=vel,
    conditioned=conditioned,
    total_ff=total,
)
print(p / "sea_controller_oracle.npz")
