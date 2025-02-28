import numpy as np


def bin_lines_by_angle(
    lines: list[tuple[float, float, float, float, float]],
    n_bins: int = 10,
    angle_offset: float = 0,
) -> list[list[tuple[float, float, float, float, float]]]:
    # Extract thetas
    thetas = [l[-1] for l in lines]
    # Get bin indices
    bin_angles = np.arange(0, np.pi + np.pi / n_bins, np.pi / n_bins) + np.deg2rad(
        angle_offset
    )
    bin_indices = np.digitize(thetas, bin_angles, right=False)
    # Sort lines into bins
    lines_binned = [[] for _ in range(n_bins)]
    for i, bin_idx in enumerate(bin_indices):
        lines_binned[bin_idx - 1].append(lines[i])
    return lines_binned
