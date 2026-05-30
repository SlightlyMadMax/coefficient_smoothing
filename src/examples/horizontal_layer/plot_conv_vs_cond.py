import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
    }
)


def add_subfigure_label(ax, label):
    """Add subfigure label (a), (b), etc. centered above the axes."""
    ax.text(
        0.5,
        1.05,
        f"{label}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        zorder=10,
    )


# -----------------------------
# load data
# -----------------------------
b_conv_melt = np.load("./data/boundary/melting/convection_boundary.npz")["b"]
b_stef_melt = np.load("./data/boundary/melting/stefan_boundary.npz")["b"]

b_conv_freeze = np.load("./data/boundary/freezing/convection_boundary.npz")["b"]
b_stef_freeze = np.load("./data/boundary/freezing/stefan_boundary.npz")["b"]

t = np.arange(0, len(b_stef_melt), 10) / 60

# -----------------------------
# figure
# -----------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.3, 3.15), constrained_layout=True)

# -------- (а) melting ----------
y1m = b_conv_melt[0:1441:10]
y2m = b_stef_melt[0:1441:10]

ax0.plot(t, y1m)
ax0.plot(t, y2m, "--")

ax0.set_xlabel("Time, h")
ax0.set_ylabel("Average interface position, m")
ax0.set_ylim(0, 0.05)

L = 0.003
dy_line = 0.0
dy_label = 0.0008

i1 = int(0.7 * (len(t) - 1))
x1, y1 = t[i1] - 0.05, y1m[i1]
ax0.plot([x1, x1], [y1 - dy_line, y1 - dy_line - L], color="black", linewidth=0.8)
ax0.text(x1 - 0.4, y1 - 0.001 - dy_label - L, "1", va="center", fontsize=10)

i2 = int(0.4 * (len(t) - 1))
x2, y2 = t[i2] - 0.05, y2m[i2]
ax0.plot([x2, x2], [y2 + dy_line, y2 + dy_line + L], color="black", linewidth=0.8)
ax0.text(x2, y2 + dy_line + dy_label + L, "2", ha="center", fontsize=10)

add_subfigure_label(ax0, "a")


# -------- (б) freezing ----------
y1f = b_conv_freeze[0:1441:10]
y2f = b_stef_freeze[0:1441:10]

ax1.plot(t, y1f, linewidth=2)
ax1.plot(t, y2f, "--", linewidth=2)

ax1.set_xlabel("Time, h")
ax1.set_ylabel("Average interface position, m")
ax1.set_ylim(0, 0.05)

i1 = int(0.7 * (len(t) - 1))
x1, y1 = t[i1], y1f[i1]
ax1.plot([x1, x1], [y1 - dy_line, y1 - dy_line - L], color="black", linewidth=0.8)
ax1.text(x1 - 0.4, y1 - 0.001 - dy_label - L, "1", va="center", fontsize=10)

i2 = int(0.4 * (len(t) - 1))
x2, y2 = t[i2] - 0.05, y2f[i2]
ax1.plot([x2, x2], [y2 + dy_line, y2 + dy_line + L], color="black", linewidth=0.8)
ax1.text(x2, y2 + dy_line + dy_label + L, "2", ha="center", fontsize=10)

add_subfigure_label(ax1, "b")


# -----------------------------
plt.savefig("./graphs/boundary_vs_time.tiff", dpi=300)
plt.show()
