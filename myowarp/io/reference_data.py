from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ReferenceData:
    series_data: dict[str, torch.Tensor]
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.metadata["resampled_data_length"])


def _npz_to_dict(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key].item() for key in loaded.files}


def load_reference_data(path: str | Path, control_framerate: int, device: str = "cpu") -> ReferenceData:
    """Load and resample MyoAssist reference data."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"Only .npz reference files are supported for now: {path}")

    ref = _npz_to_dict(path)
    if "resampled_series_data" not in ref:
        original_sample_rate = ref["metadata"]["sample_rate"]
        for key in list(ref["series_data"].keys()):
            values = np.asarray(ref["series_data"][key], dtype=np.float32)
            original_len = len(values)
            original_x = np.linspace(0, original_len - 1, original_len)
            new_len = int(original_len * control_framerate / original_sample_rate)
            new_x = np.linspace(0, original_len - 1, new_len)
            ref["series_data"][key] = np.interp(new_x, original_x, values).astype(np.float32)
            ref["metadata"]["resampled_data_length"] = new_len
            ref["metadata"]["resampled_sample_rate"] = control_framerate

    series = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in ref["series_data"].items()
    }
    return ReferenceData(series_data=series, metadata=ref["metadata"])
