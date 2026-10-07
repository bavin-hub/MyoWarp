"""Runner for dual-policy full-observation velocity tracking."""

from myowarp.tasks.h1_with_osl_velocity.runner import H1VelocityRunner


class H1DualFullVelocityRunner(H1VelocityRunner):
    """Use the velocity curriculum checkpointing with a dual actor."""

