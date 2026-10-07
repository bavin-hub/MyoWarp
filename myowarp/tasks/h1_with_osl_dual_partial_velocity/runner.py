"""Runner for dual-policy partial-observation velocity tracking."""

from myowarp.tasks.h1_with_osl_dual_full_velocity.runner import (
    H1DualFullVelocityRunner,
)


class H1DualPartialVelocityRunner(H1DualFullVelocityRunner):
    """Use the velocity curriculum checkpointing with partial OSL observations."""

