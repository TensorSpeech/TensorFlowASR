import math
import os
import re

import matplotlib.pyplot as plt

PLOT_DIR_ENV = "TFASR_PLOT_DIR"


def get_plot_path(title: str, output: str = None) -> str:
    """
    Resolve where a figure should be written.

    An explicit `output` wins; otherwise the file is named after `title` and placed in
    `$TFASR_PLOT_DIR` (default: the current working directory).
    """
    if output is None:
        filename = re.sub(r"[^\w.-]+", "_", title).strip("_") or "figure"
        output = os.path.join(os.getenv(PLOT_DIR_ENV, os.getcwd()), f"{filename}.png")
    directory = os.path.dirname(os.path.abspath(output))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return output


def plotline(data, title="data", figsize=(24, 5), output: str = None, dpi: int = 150) -> str:
    """
    Plot `data` as a line and write it to disk, returning the path.

    Same contract as `plotmesh`: it saves rather than shows, so it is safe to call from tests.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontweight="bold")
    ax.plot(data)
    ax.minorticks_on()
    fig.tight_layout()
    path = get_plot_path(title, output)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plotmesh(data, title="data", scale_ysize=4, invert_yaxis=True, output: str = None, dpi: int = 150) -> str:
    """
    Render `data` as a colour mesh and write it to disk.

    Returns the path written. This deliberately never calls `plt.show()`: on a GUI backend that
    blocks until the window is closed, which hangs any non-interactive caller -- it is what hung
    the test suite. Pass `output` to choose the file, or set `$TFASR_PLOT_DIR`.
    """
    xsize = data.shape[1]
    ysize = data.shape[0]
    gcd = math.gcd(xsize, ysize)
    xsize /= gcd
    ysize /= gcd
    xsize = (xsize * scale_ysize) / ysize
    ysize = scale_ysize
    figsize = [xsize, ysize]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontweight="bold")
    ax.minorticks_on()
    if invert_yaxis:
        ax.invert_yaxis()
    img = ax.pcolormesh(data, cmap="viridis")
    cbar = fig.colorbar(img, ax=ax, format="%.2f", pad=0.01)
    cbar.minorticks_on()
    fig.tight_layout()
    path = get_plot_path(title, output)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)  # closing keeps repeated calls from leaking figures
    return path
