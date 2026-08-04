from __future__ import annotations

__all__ = ["MyoAssistLegCpuEnv", "MyoAssistLegWarpEnv"]


def __getattr__(name: str):
    if name == "MyoAssistLegCpuEnv":
        from myowarp.envs.myoassist_leg_cpu import MyoAssistLegCpuEnv

        return MyoAssistLegCpuEnv
    if name == "MyoAssistLegWarpEnv":
        from myowarp.envs.myoassist_leg_warp import MyoAssistLegWarpEnv

        return MyoAssistLegWarpEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
