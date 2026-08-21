"""Method-independent plots for generated JSON experiment results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def plot_result(result: dict[str, Any], destination: str | Path) -> Path:
    """Plot an active experiment result and return the output path."""

    cache = Path(tempfile.gettempdir()) / "continual-learning-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib.pyplot as plt

    experiment = result.get("experiment")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if experiment == "nonstationary_signal":
        figure, axis = plt.subplots(figsize=(9, 5))
        for name, model in result["models"].items():
            curve = model["rolling_mse"]
            axis.plot(
                [point["step"] for point in curve],
                [point["mse"] for point in curve],
                label=name.upper(),
            )
        for point in result["models"]["frozen"]["change_points"]:
            axis.axvline(point, color="grey", alpha=0.25, linewidth=1)
        axis.set_yscale("log")
        axis.set_xlabel("stream step")
        axis.set_ylabel("rolling prequential MSE")
        axis.set_title("Online adaptation on a nonstationary signal")
        axis.legend()
        axis.grid(alpha=0.2)
    else:
        raise ValueError(f"plotting is not implemented for {experiment!r}")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
