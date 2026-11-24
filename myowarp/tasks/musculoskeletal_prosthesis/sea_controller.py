"""Local batched port of cmu_leg LLC and prosim's simulated drive integral.

No sibling-repository imports. Filter design/model lookup occur at construction;
all tick state and arithmetic remain on the supplied device. See CONTROLLER_NOTICE.md.
"""

import json
from pathlib import Path

import torch
from scipy.signal import butter

from .gpu_controller import GPUMotorCommand


class SEATorqueController:
    """Independent per-world state. Float64 protects second torque differences.

    `compute` exposes hardware enable/actuate/standing for parity tests. `update`
    runs the walking-training mode: enabled, actuating, not standing. Mid-motion
    reset treats the hardware enable dwell as already completed, as in ProSim.
    The physics motor DOFs use output-side Nm/rad/s, with no extra gear factors.
    """

    def __init__(
        self, *, sim, num_envs, device, dtype=torch.float64, startup_ready=True
    ):
        self.num_envs, self.device, self.dtype = num_envs, torch.device(device), dtype
        self.startup_ready = startup_ready
        self.parameters = json.loads(
            Path(__file__).with_name("sea_parameters.json").read_text()
        )
        self.common = self.parameters["common"]
        self.knee, self.ankle = self.parameters["knee"], self.parameters["ankle"]
        model = sim.cpu_model

        def tensor(values):
            return torch.tensor(values, device=self.device, dtype=dtype)

        def index(names, attr):
            return torch.tensor(
                [getattr(model.joint(n), attr)[0] for n in names], device=self.device
            )

        self.qids = index(
            ("knee", "knee_motor", "ankle", "ankle_motor", "ankle_drive"), "qposadr"
        )
        self.vids = index(
            ("knee", "knee_motor", "ankle", "ankle_motor", "ankle_drive"), "dofadr"
        )
        self.ks = tensor([self.knee["stiffness"], self.ankle["stiffness"]])
        self.kp = tensor([self.knee["kp"], self.ankle["kp"]])
        self.bm = tensor([self.knee["motor_damping"], self.ankle["motor_damping"]])
        self.jm = tensor([self.knee["motor_inertia"], self.ankle["motor_inertia"]])
        # Ankle source intentionally does NOT divide feedforward by motor_eff.
        self.efficiency = tensor([self.knee["motor_eff"], 1.0])
        self.velmax = tensor([self.knee["vel_max"], self.ankle["vel_max"]])
        self.taumax = tensor([self.knee["torque_max"], self.ankle["torque_max"]])
        ffids = [model.actuator(n).id for n in ("knee_act", "ankle_act")]
        velids = [model.actuator(n).id for n in ("knee_vel", "ankle_vel")]
        self.drive_limit = tensor(model.actuator_forcerange[ffids, 1].copy())
        self.drive_ki = tensor(model.actuator_gainprm[velids, 0].copy()) * (
            2 * torch.pi * tensor(self.parameters["drive_integral_corner_hz"])
        )
        # Source uses 25 Hz/order2 defaults for conditioning. Six other filters
        # include the ankle spring velocity and per-joint output channels.
        cutoffs = [
            25,
            25,
            25,
            self.ankle["torque_ctrl_spring_vel_lowpass_cutoff"],
            self.knee["vel_sat_lowpass_cutoff"],
            self.ankle["vel_sat_lowpass_cutoff"],
            self.knee["vel_sat_lowpass_cutoff"],
            self.ankle["vel_sat_lowpass_cutoff"],
        ]
        self.coeffs = []
        for hz in cutoffs:
            b0, b1, b2, a0, a1, a2 = butter(
                2, hz, fs=self.common["sample_rate"], output="sos"
            )[0]
            self.coeffs.append(tuple(float(v / a0) for v in (b0, b1, b2, a1, a2)))
        self.z1 = torch.zeros((num_envs, 8), device=self.device, dtype=dtype)
        self.z2 = torch.zeros_like(self.z1)
        self.previous_tau = torch.zeros((num_envs, 2), device=self.device, dtype=dtype)
        self.previous_tau_dot = torch.zeros_like(self.previous_tau)
        self.drive_integral = torch.zeros_like(self.previous_tau)
        self.prev_active = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.prev_enabled = torch.zeros_like(self.prev_active)
        self.ticks = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.enabled = torch.ones_like(self.prev_active)
        self.actuate = torch.ones_like(self.prev_active)
        self.standing = torch.zeros_like(self.prev_active)
        self.commanded_torque = torch.zeros_like(self.previous_tau)
        self.raw_feedforward = torch.zeros_like(self.previous_tau)
        self.velocity_command = torch.zeros_like(self.previous_tau)
        self.reset(torch.arange(num_envs, device=self.device))

    def reset(self, env_ids):
        for t in (
            self.z1,
            self.z2,
            self.previous_tau,
            self.previous_tau_dot,
            self.drive_integral,
            self.commanded_torque,
            self.raw_feedforward,
            self.velocity_command,
        ):
            t[env_ids] = 0
        self.prev_active[env_ids] = False
        self.prev_enabled[env_ids] = self.startup_ready
        self.ticks[env_ids] = (
            self.common["startup_mute_ticks"] if self.startup_ready else 0
        )
        self.enabled[env_ids], self.actuate[env_ids], self.standing[env_ids] = (
            True,
            True,
            False,
        )

    def _lowpass(self, x, channel, active):
        b0, b1, b2, a1, a2 = self.coeffs[channel]
        y = b0 * x + self.z1[:, channel]
        z1 = b1 * x - a1 * y + self.z2[:, channel]
        z2 = b2 * x - a2 * y
        self.z1[:, channel] = torch.where(active, z1, self.z1[:, channel])
        self.z2[:, channel] = torch.where(active, z2, self.z2[:, channel])
        return y

    @staticmethod
    def _stop(position, velocity, limit, ext, cfg):
        e = ext * cfg["c_jointstop"] * (position - limit)
        ev = -ext * velocity / cfg["w_max_jointstop"]
        return torch.where((e > 0) & (ev < 1), -ext * e * (1 - ev), 0.0)

    @staticmethod
    def _clip(x, lo, hi):
        # Match min(max(x,lo),hi), including lo>hi soft-stop cases.
        return torch.minimum(torch.maximum(x, lo), hi)

    @torch.no_grad()
    def compute(
        self,
        state,
        desired_joint_torque,
        dt,
        *,
        enabled=None,
        actuate=None,
        standing=None,
    ):
        if not 0 < dt:
            raise ValueError("Controller dt must be positive")
        enabled = self.enabled if enabled is None else enabled
        actuate = self.actuate if actuate is None else actuate
        standing = self.standing if standing is None else standing
        active = enabled & actuate
        edge = active & ~self.prev_active
        enable_edge = enabled & ~self.prev_enabled
        self.prev_active.copy_(active)
        self.prev_enabled.copy_(enabled)
        self.ticks.copy_(torch.where(enable_edge, 0, self.ticks) + 1)
        for t in (self.z1, self.z2, self.previous_tau, self.previous_tau_dot):
            t.masked_fill_(edge[:, None], 0)
        q = state.qpos[:, self.qids].to(self.dtype)
        v = state.qvel[:, self.vids].to(self.dtype)
        desired = desired_joint_torque.to(self.dtype)
        k, a = self.knee, self.ankle
        # Conditioning preserves the source's knee motor/joint asymmetry.
        flex = self._stop(q[:, 0], v[:, 0], k["angle_max_soft"], k["flex_ext"], k)
        ext = self._stop(q[:, 1], v[:, 1], k["angle_min_soft"], k["extend_ext"], k)
        hi = torch.where(q[:, 0] > k["angle_max_soft"], flex, k["torque_max"])
        lo = torch.where(
            (q[:, 1] < k["angle_min_soft"]) & ~standing, ext, -k["torque_max"]
        )
        tau_k = (
            self._lowpass(self._clip(desired[:, 0] + flex + ext, lo, hi), 0, active)
            + self.common["damping_gain"] * v[:, 0]
        )
        av = self._lowpass(v[:, 3], 1, active)
        af = self._stop(q[:, 3], av, a["angle_max_soft"], a["flex_ext"], a)
        ae = self._stop(q[:, 3], av, a["angle_min_soft"], a["extend_ext"], a)
        hi = torch.where(q[:, 3] > a["angle_max_soft"], af, a["torque_max"])
        lo = torch.where(q[:, 3] < a["angle_min_soft"], ae, -a["torque_max"])
        tau_a = self._lowpass(self._clip(desired[:, 1], lo, hi), 2, active)
        tau = torch.stack((tau_k, tau_a), 1)
        derivative_dt = min(max(dt, self.common["dt_min"]), self.common["dt_max"])
        tau_dot = (tau - self.previous_tau) / derivative_dt
        tau_ddot = (tau_dot - self.previous_tau_dot) / derivative_dt
        self.previous_tau.copy_(torch.where(active[:, None], tau, self.previous_tau))
        self.previous_tau_dot.copy_(
            torch.where(active[:, None], tau_dot, self.previous_tau_dot)
        )
        measured = self.ks * torch.stack((q[:, 1] - q[:, 0], q[:, 3] - q[:, 4]), 1)
        spring_v = self._lowpass(v[:, 4], 3, active)
        velocity = (
            torch.stack((v[:, 0], spring_v), 1)
            + tau_dot / self.ks
            + self.kp * (tau - measured)
        )
        ff = (
            tau + self.bm / self.ks * tau_dot + self.jm / self.ks * tau_ddot
        ) / self.efficiency
        # Knee hard flexion gate reads joint; all other hard gates read motor.
        flexed = torch.stack((q[:, 0] > k["angle_max"], q[:, 3] > a["angle_max"]), 1)
        extended = torch.stack((q[:, 1] < k["angle_min"], q[:, 3] < a["angle_min"]), 1)
        vlo, vhi = (
            torch.where(extended, 0.0, -self.velmax),
            torch.where(flexed, 0.0, self.velmax),
        )
        tlo, thi = (
            torch.where(extended, 0.0, -self.taumax),
            torch.where(flexed, 0.0, self.taumax),
        )
        velocity = self._clip(velocity, vlo, vhi)
        ff = self._clip(ff, tlo, thi)
        velocity = torch.stack(
            (
                self._lowpass(velocity[:, 0], 4, active),
                self._lowpass(velocity[:, 1], 5, active),
            ),
            1,
        )
        ff = torch.stack(
            (self._lowpass(ff[:, 0], 6, active), self._lowpass(ff[:, 1], 7, active)), 1
        )
        live = active & (self.ticks > self.common["startup_mute_ticks"])
        velocity, ff = (
            torch.where(live[:, None], velocity, 0.0),
            torch.where(live[:, None], ff, 0.0),
        )
        self.commanded_torque.copy_(torch.where(active[:, None], tau, 0.0))
        self.raw_feedforward.copy_(ff)
        self.velocity_command.copy_(velocity)
        # ProSim apply(): clamped drive I, plus feedforward; MuJoCo supplies P.
        # Avoid a temporary CPU index tensor so this tick can be CUDA-captured.
        motor_v = torch.stack((v[:, 1], v[:, 3]), dim=1)
        integral = (
            self.drive_integral + self.drive_ki * (velocity - motor_v) * dt
        ).clamp(-self.drive_limit, self.drive_limit)
        self.drive_integral.copy_(torch.where(active[:, None], integral, 0.0))
        total_ff = torch.where(
            active[:, None],
            (ff + self.drive_integral).clamp(-self.drive_limit, self.drive_limit),
            0.0,
        )
        return GPUMotorCommand(
            total_ff.to(state.ctrl.dtype), velocity.to(state.ctrl.dtype)
        )

    def update(self, state, desired_joint_torque, dt):
        return self.compute(state, desired_joint_torque, dt)


def create_controller(*, sim, num_envs, device):
    """Default CUDA walking controller factory; independent per-world histories."""
    return SEATorqueController(sim=sim, num_envs=num_envs, device=device)
