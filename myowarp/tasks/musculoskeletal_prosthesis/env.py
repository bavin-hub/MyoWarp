"""CUDA-only public task environment; native numerical oracle lives in cpu_oracle.py."""

from .gpu_env import GPUProsthesisVecEnv

ProsthesisEnv = GPUProsthesisVecEnv

__all__ = ["GPUProsthesisVecEnv", "ProsthesisEnv"]
