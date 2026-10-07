"""RSL-RL runner integration for the standalone velocity task."""

from __future__ import annotations

from pathlib import Path

from rsl_rl.runners import OnPolicyRunner


class H1VelocityRunner(OnPolicyRunner):
    """Persist the command curriculum counter alongside RSL-RL state."""

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint_infos = dict(infos or {})
        checkpoint_infos["env_state"] = self.env.training_state()
        super().save(path, checkpoint_infos)

    def load(
        self,
        path: str | Path,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(str(path), load_cfg, strict, map_location)
        infos = infos or {}
        self.env.load_training_state(infos.get("env_state"))
        return infos
