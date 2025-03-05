import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1) Define a small input (5×5), a kernel (3×3), and compute one output pixel. ---
# For clarity, we'll compute the convolution value at output pixel (2,2).
# We do a simple "valid" convolution step for that one pixel (no padding).
input_mat = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 1],
    [2, 3, 4, 5, 6],
    [7, 8, 9, 1, 2],
    [3, 4, 5, 6, 7]
], dtype=float)

kernel_mat = np.array([
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 1,  0, -1]
], dtype=float)

# We pick the center of the input, (2,2), for the demonstration:
i, j = 2, 2  # row=2, col=2

# Extract the 3×3 patch around (i, j) (assuming 'valid' region is inside).
patch = input_mat[i-1:i+2, j-1:j+2]
conv_value = np.sum(patch * kernel_mat)

# Prepare an output matrix (5×5) that is all zeros except the computed pixel.
output_mat = np.zeros_like(input_mat)
output_mat[i, j] = conv_value

# --- 2) Helper function to draw a matrix as squares with optional highlights. ---
def draw_matrix(ax, matrix, x_offset=0, y_offset=0, cell_size=1,
                highlight_cells=None, highlight_color='red', alpha=0.2,
                show_values=True, cmap=None):
    """
    Draws a 2D matrix as a grid of colored squares on Axes 'ax'.
    
    Parameters:
      matrix          : 2D NumPy array
      x_offset, y_offset : Offsets to position the grid in the plot
      cell_size       : Size of each cell (in data coordinates)
      highlight_cells : List of (row, col) tuples to highlight
      highlight_color : Color used to highlight cells
      alpha           : Opacity of the highlight overlay
      show_values     : Whether to place numeric text inside each cell
      cmap            : Optional Matplotlib colormap for coloring cell backgrounds
    """
    nrows, ncols = matrix.shape
    
    # For coloring if a colormap is provided
    norm = None
    if cmap is not None:
        vmin, vmax = matrix.min(), matrix.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Draw each cell as a rectangle
    for r in range(nrows):
        for c in range(ncols):
            # Calculate the lower-left corner of the cell in plot coordinates
            x = x_offset + c * cell_size
            # Invert row index so row 0 is at top visually
            y = y_offset + (nrows - r - 1) * cell_size
            
            # Base rectangle
            rect = patches.Rectangle((x, y), cell_size, cell_size,
                                     linewidth=1, edgecolor='black',
                                     facecolor='white')
            ax.add_patch(rect)
            
            # If a colormap is given, color the cell according to its value
            if cmap is not None and norm is not None:
                color_val = cmap(norm(matrix[r, c]))
                rect.set_facecolor(color_val)
            
            # Highlight specific cells if requested
            if highlight_cells and (r, c) in highlight_cells:
                highlight_rect = patches.Rectangle((x, y), cell_size, cell_size,
                                                   linewidth=2, edgecolor=highlight_color,
                                                   facecolor=highlight_color, alpha=alpha)
                ax.add_patch(highlight_rect)
            
            # Optionally place the numeric value in the center of the cell
            if show_values:
                ax.text(x + cell_size / 2, y + cell_size / 2,
                        f"{matrix[r, c]:.1f}",
                        ha='center', va='center', fontsize=8)
    
    # Adjust axes so the drawn grid fits neatly
    ax.set_xlim(x_offset, x_offset + ncols * cell_size)
    ax.set_ylim(y_offset, y_offset + nrows * cell_size)
    ax.set_aspect('equal')
    ax.axis('off')

# --- 3) Plot everything side by side in a single figure. ---
fig, ax = plt.subplots(figsize=(10, 5))

# Draw the 5×5 input on the left, highlighting the 3×3 patch used for pixel (2,2).
draw_matrix(
    ax,
    input_mat,
    x_offset=0,
    y_offset=0,
    cell_size=1,
    highlight_cells=[(r, c) for r in range(i-1, i+2) for c in range(j-1, j+2)],
    highlight_color='red',
    alpha=0.3,
    show_values=True,
    cmap=plt.cm.Blues
)

# Draw the 3×3 kernel in the middle, fully highlighted.
draw_matrix(
    ax,
    kernel_mat,
    x_offset=7,
    y_offset=1,  # shift it slightly up
    cell_size=1,
    highlight_cells=[(r, c) for r in range(3) for c in range(3)],
    highlight_color='red',
    alpha=0.3,
    show_values=True,
    cmap=plt.cm.Reds
)

# Draw the 5×5 output on the right, highlighting the single pixel (2,2) where the result goes.
draw_matrix(
    ax,
    output_mat,
    x_offset=14,
    y_offset=0,
    cell_size=1,
    highlight_cells=[(i, j)],
    highlight_color='green',
    alpha=0.3,
    show_values=True,
    cmap=plt.cm.Greens
)

# Label the three sections
ax.text(2.5, -1, "Input (5×5)", ha='center', va='center', fontsize=12)
ax.text(8.5,  4.2, "Kernel (3×3)", ha='center', va='center', fontsize=12)
ax.text(16.5, -1, "Output (5×5)", ha='center', va='center', fontsize=12)

# Draw arrows to indicate the flow of convolution
# Arrow from the highlighted patch in the input to the kernel
ax.annotate(
    "",
    xy=(7, 2.5), xycoords='data',    # arrow tip
    xytext=(5, 2.5), textcoords='data',  # arrow start
    arrowprops=dict(arrowstyle="->", lw=2, color='red')
)

# Arrow from the kernel to the highlighted output pixel
ax.annotate(
    "",
    xy=(14, 2.5), xycoords='data',   # arrow tip
    xytext=(10, 2.5), textcoords='data', # arrow start
    arrowprops=dict(arrowstyle="->", lw=2, color='green')
)

ax.set_title("Conceptual Visualization of a Single Convolution Step", fontsize=14)
plt.show()
