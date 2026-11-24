"""PPO defaults copied from H1 tracking without importing its Warp environment."""


def make_rsl_rl_cfg(
    *,
    max_iterations: int = 30_000,
    save_interval: int = 500,
    run_name: str = "",
) -> dict:
    """Return the native RSL-RL configuration used for motion tracking."""
    return {
        "seed": 42,
        "num_steps_per_env": 24,
        "max_iterations": max_iterations,
        "save_interval": save_interval,
        "experiment_name": "h1_with_osl_tracking_standalone",
        "run_name": run_name,
        "logger": "tensorboard",
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
            "optimizer": "adam",
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "share_cnn_encoders": False,
        },
        "multi_gpu": None,
        "check_for_nan": True,
    }
