# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under Apache-2.0; see LICENSE and NOTICE in this directory.
# Ported from MyoSuite 3.0.0; imports adapted for standalone MyoWarp use.

"""Legacy motion-cache lookup used by the local trajectory reader."""
import os
from pathlib import Path

_MUSCLEMIMIC_CACHE_ENV_VARS = (
    "MUSCLEMIMIC_CONVERTED_AMASS_PATH",
    "CONVERTED_AMASS_PATH",
)


def default_musclemimic_cache_root() -> Path:
    """Return cache root used by MuscleMimic-compatible assets.

    Resolution order mirrors upstream MuscleMimic cache discovery:

    1. ``MUSCLEMIMIC_CONVERTED_AMASS_PATH``
    2. ``CONVERTED_AMASS_PATH``
    3. legacy ``~/.musclemimic/caches/AMASS``
    """
    for env_var in _MUSCLEMIMIC_CACHE_ENV_VARS:
        configured = os.environ.get(env_var)
        if configured:
            return Path(configured).expanduser()
    return Path.home() / ".musclemimic" / "caches" / "AMASS"

