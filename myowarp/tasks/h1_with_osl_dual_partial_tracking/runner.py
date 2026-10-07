"""Runner for dual-policy partial-observation motion tracking."""

from myowarp.tasks.h1_with_osl_dual_full_tracking.runner import (
    H1DualFullTrackingRunner,
)


class H1DualPartialTrackingRunner(H1DualFullTrackingRunner):
    """Persist tracking state and export the two-input dual actor."""

