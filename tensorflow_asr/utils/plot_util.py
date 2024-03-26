import math

import matplotlib.pyplot as plt


def plotmesh(data, title="data", scale_ysize=4, invert_yaxis=True):
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
    plt.show()
