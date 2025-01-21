import subprocess
import numpy as np
import cv2
from time import sleep, time


def get_remote_directories(server, remote_path):
    """
    Get a list of numbered directories from a remote server.

    Args:
        server (str): SSH server address (e.g., user@hostname).
        remote_path (str): Path to the directory on the remote server.

    Returns:
        set: A set of integers representing the existing directory names.
    """
    try:
        # Run the SSH command to list the directory contents
        command = f"ssh {server} ls {remote_path}"
        output = subprocess.check_output(command, shell=True, text=True)

        # Filter numeric directories and return as a set of integers
        return set(int(d) for d in output.splitlines() if d.isdigit())
    except subprocess.CalledProcessError as e:
        print(f"Error accessing remote directory: {e}")
        return set()


server = "ma_ploen"
data_path = "masterarbeit/data/generation/out"
output_image = "dump/output_visualization.png"
max_value = 16384
grid_size = int(np.ceil(np.sqrt(max_value + 1)))

start_samples = None
start_time = time()

while True:
    # Get the existing directories from the remote server
    existing_dirs = get_remote_directories(server, data_path)

    # Time Tracking
    if start_samples is None:
        start_samples = len(existing_dirs)
    n_samples = len(existing_dirs) - start_samples
    dt = time() - start_time
    sec_per_sample = dt / n_samples if n_samples != 0 else 0

    # Create a grid and mark existing directories
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    grid[:, :] = (0, 0, 255)
    for value in existing_dirs:
        if 0 <= value <= max_value:  # Ensure the value is within the valid range
            row, col = divmod(value, grid_size)
            grid[row, col] = (0, 255, 0)  # Mark the pixel

    # Resize grid for better visualization
    scale_factor = 7  # Scale up for better visualization
    visualization = cv2.resize(
        grid,
        (grid_size * scale_factor, grid_size * scale_factor),
        interpolation=cv2.INTER_NEAREST,
    )
    visualization[::scale_factor] = 0
    visualization[1::scale_factor] = 0
    visualization[:, ::scale_factor] = 0
    visualization[:, 1::scale_factor] = 0

    txt = f"{len(existing_dirs)}/{max_value}, {sec_per_sample:.2f}s/sample"
    cv2.putText(
        visualization,
        txt,
        org=(10, 25),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1.5,
        color=(0, 0, 0),
        thickness=5,
    )
    cv2.putText(
        visualization,
        txt,
        org=(10, 25),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1.5,
        color=(255, 255, 255),
        thickness=1,
    )

    # Save and display the image
    cv2.imwrite(output_image, visualization)
    print(txt, end="\r")
    sleep(20)
