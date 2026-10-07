"""Runner for dual-policy full-observation motion tracking."""

from myowarp.tasks.h1_with_osl_tracking.runner import H1TrackingRunner


class H1DualFullTrackingRunner(H1TrackingRunner):
    """Persist adaptive sampling and bundle motion data for the dual actor."""

