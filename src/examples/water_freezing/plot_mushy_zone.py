#!/usr/bin/env python
"""
How wide the smoothed band actually is, in cells, along the front.

Both smoothing parameters are widths in TEMPERATURE: the phase change is spread over
eps_T and the momentum sink over eps_flow, through the same smoothed step in T. What
the grid sees is the width in space, which is that temperature interval divided by the
local gradient at the front - and the gradient is not the same on the two sides, since
k_s / k_l = 3.8 makes the solid half of the band the wider one. Both halves are
measured here.

The regularised step has infinite support, so "the width" only exists relative to a
cut-off. Panel (a) therefore draws three nested bands rather than one, and the number
worth quoting is at the bottom of the printout: the fraction of the latent heat that
is released within one, two and three cells.

The damped band is a different object. It is set by where C (1 - f_l)^2 is large
enough to arrest the flow, which happens far out on the tail where f_l is 1 to within
a millionth, so it does not coincide with any f_l cut-off.

Example
-------
    python -m src.examples.water_freezing.plot_mushy_zone
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfinv

HERE = Path(__file__).resolve().parent
DU = 10.0          # K per unit of the dimensionless temperature
# The solver's smoothed step is 0.5 (1 + erf(dT / (sqrt(2) delta))) and its companion
# delta function is a Gaussian of standard deviation delta, so the parameter called
# eps_T here IS the standard deviation. Writing erf(dT / eps) instead would make every
# width come out a factor of sqrt(2) too small
SQRT2 = np.sqrt(2.0)
CUTS = (0.1, 0.01, 1e-3)

mpl.rcParams.update({
    "font.size": 12, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8,
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "lines.linewidth": 1.4, "figure.dpi": 300,
})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Width of the smoothed band along the phase interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run", type=Path,
        default=HERE / "data" / "star" / "151x151_dt0.02_e0.1_ef0.1_C1e+06",
    )
    p.add_argument(
        "--zoom-at", type=float, nargs=2, default=[0.20, 0.90],
        help="heights y/H of the two magnified windows",
    )
    p.add_argument("--zoom-cells", type=int, default=6,
                   help="half-width of a magnified window, in cells")
    p.add_argument(
        "--v-cut", type=float, default=0.1,
        help="the damped band reaches out to where the speed is this fraction of the "
        "local maximum along the front",
    )
    p.add_argument("--out", type=Path, default=HERE / "graphs" / "mushy_zone.png")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def load(run: Path) -> dict:
    meta = json.load(open(run / "summary.json", encoding="utf-8"))
    cps = sorted(glob.glob(str(run / "checkpoint_*.npz")),
                 key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not cps:
        raise SystemExit(f"No checkpoint in {run}")
    with np.load(cps[-1], allow_pickle=True) as z:
        meta |= {k: z[k] for k in ("u", "v_x", "v_y")}
    return meta


def front_row(u_row: np.ndarray, dx: float) -> tuple[float, int]:
    """Sub-cell x of the last T = 0 crossing, and the index of the cell before it."""
    s = np.where(np.diff(np.sign(u_row)) != 0)[0]
    if not s.size:
        return np.nan, -1
    i = s[-1]
    a, b = u_row[i], u_row[i + 1]
    return (i + a / (a - b)) * dx, i


def grads(u: np.ndarray, j: int, i: int, xi: float,
          dx: float, dy: float) -> tuple[float, float, float]:
    """
    |grad T| at the front on each side, plus the x component on the liquid side.

    Taken along the normal: the interface turns almost horizontal at the kink of the
    S, where the x derivative alone would overstate the width by a factor of two.
    """
    def one_side(idx, i_col):
        gx = abs(float(np.polyfit(idx * dx - xi, u[j, idx] * DU, 2)[1]))
        gy = abs(float(u[j + 1, i_col] - u[j - 1, i_col]) * DU / (2.0 * dy))
        return gx, float(np.hypot(gx, gy))

    gx_l, gn_l = one_side(np.array([i, i - 1, i - 2]), i)
    _, gn_s = one_side(np.array([i + 1, i + 2, i + 3]), i + 1)
    return gn_l, gn_s, gx_l


def damped_edge(speed: np.ndarray, dx: float, threshold: float) -> float:
    """
    x of the outermost point where the speed still exceeds `threshold`.

    Interpolated within the cell: the band is of the order of one cell, so rounding
    the crossing to whole cells would quantise it as coarsely as the quantity itself.
    """
    above = np.where(speed > threshold)[0]
    if not above.size:
        return np.nan
    k = above[-1]
    if k + 1 >= speed.size:
        return k * dx
    a, b = speed[k], speed[k + 1]
    return (k + (a - threshold) / (a - b)) * dx if a != b else k * dx


def analyse(d: dict, v_cut: float) -> dict:
    u, dx, dy = d["u"], d["dx"], d["dy"]
    sp = np.hypot(d["v_x"], d["v_y"])
    n_y, n_x = u.shape

    y, xi_l, g_l, g_s, w_damp, v_ref_l = [], [], [], [], [], []
    for j in range(3, n_y - 3):
        xi, i = front_row(u[j], dx)
        if not np.isfinite(xi) or i < 3 or i + 4 >= n_x:
            continue
        gn_l, gn_s, gx_l = grads(u, j, i, xi, dx, dy)
        if gn_l <= 0 or gn_s <= 0:
            continue
        i_lo = max(0, i - 40)          # the layer on the ice, not the jet at the hot wall
        v_ref = sp[j, i_lo:i + 1].max()
        # Whether any flow reaches the front at all, as opposed to the free stream
        # somewhere out in the row: at the stagnation line between the two cells the
        # fluid next to the ice is at rest and no threshold can locate a damped edge
        v_near = sp[j, max(0, i - 8):i + 1].max()
        xd = damped_edge(sp[j], dx, v_cut * v_ref)
        y.append(j / (n_y - 1))
        xi_l.append(xi)
        g_l.append(gn_l)
        g_s.append(gn_s)
        v_ref_l.append(v_near)
        # Measured along a row, so project onto the normal
        w_damp.append((xi - xd) * gx_l / gn_l)
    r = {k: np.array(v) for k, v in
         dict(y=y, xi=xi_l, g_l=g_l, g_s=g_s, w_damp=w_damp, v_ref=v_ref_l).items()}
    # Half-widths of the band on each side, for every cut-off
    for c in CUTS:
        t_edge = SQRT2 * d["eps_T"] * erfinv(1.0 - 2.0 * c)
        r[f"wl_{c}"] = t_edge / r["g_l"]
        r[f"ws_{c}"] = t_edge / r["g_s"]
    return r


BANDS = {"lower (inversion cell)": lambda f: (f > 0.03) & (f < 0.58),
         "kink of the S": lambda f: (f >= 0.58) & (f <= 0.68),
         "upper (main cell)": lambda f: (f > 0.68) & (f < 0.97),
         "whole front": lambda f: np.ones_like(f, bool)}


def report(d: dict, r: dict) -> None:
    dx, eps = d["dx"], d["eps_T"]
    print(f"{d['n_x']}x{d['n_y']}, eps_T={eps:g} K, eps_flow={d['eps_flow']:g} K, "
          f"C={d['penalty_C']:.0e} 1/s, dx={dx*1e3:.4f} mm\n")

    print("gradient just outside the front, K/m")
    print(f"{'band':<24}{'liquid':>10}{'solid':>10}{'ratio':>8}")
    for tag, sel in BANDS.items():
        m = sel(r["y"])
        print(f"{tag:<24}{r['g_l'][m].mean():>10.1f}{r['g_s'][m].mean():>10.1f}"
              f"{(r['g_l'][m]/r['g_s'][m]).mean():>8.2f}")

    print("\nband width in cells, both halves, against the cut-off on f_l")
    print(f"{'band':<24}" + "".join(f"{('f_l=' + f'{c:g}'):>12}" for c in CUTS)
          + f"{'damped':>10}")
    for tag, sel in BANDS.items():
        m = sel(r["y"])
        line = f"{tag:<24}"
        for c in CUTS:
            line += f"{((r[f'wl_{c}'][m] + r[f'ws_{c}'][m]).mean() / dx):>12.2f}"
        line += f"{np.nanmean(r['w_damp'][m]) / dx:>10.2f}"
        print(line)

    print("\nthe number that does not depend on a cut-off:")
    print("fraction of the latent heat released within N cells of the front")
    print(f"{'band':<24}{'1 cell':>10}{'2 cells':>10}{'3 cells':>10}")
    for tag, sel in BANDS.items():
        m = sel(r["y"])
        # Each half of the strip is converted to a temperature interval with its own
        # gradient, and the Gaussian is integrated over the two halves separately
        line = f"{tag:<24}"
        for nc in (1.0, 2.0, 3.0):
            h = 0.5 * nc * dx
            frac = 0.5 * (erf(h * r["g_l"][m] / (SQRT2 * eps))
                          + erf(h * r["g_s"][m] / (SQRT2 * eps)))
            line += f"{np.mean(frac) * 100:>10.2f}"
        print(line)


def draw_zoom(ax, d: dict, r: dict, y_target: float, half: int) -> None:
    """One window of the mesh around the front, coloured by liquid fraction."""
    u, dx, dy, eps = d["u"], d["dx"], d["dy"], d["eps_T"]
    n_y = u.shape[0]
    j = int(round(y_target * (n_y - 1)))
    k = int(np.argmin(np.abs(r["y"] - y_target)))
    i0 = int(r["xi"][k] / dx)

    sl_x = slice(max(0, i0 - half), i0 + half + 1)
    sl_y = slice(max(0, j - half), j + half + 1)
    sub = u[sl_y, sl_x]
    f_l = 0.5 * (1.0 + erf(sub * DU / (SQRT2 * eps)))

    x = np.arange(sl_x.start, sl_x.stop) * dx
    y = np.arange(sl_y.start, sl_y.stop) * dy
    ext = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]
    ax.imshow(f_l, extent=ext, origin="lower", cmap="Blues_r", vmin=0.0, vmax=1.0,
              interpolation="nearest")
    ax.contour(x, y, sub, levels=[0.0], colors="k", linewidths=1.2)
    for c in (CUTS[1],):
        t = SQRT2 * eps * erfinv(1.0 - 2.0 * c) / DU
        ax.contour(x, y, sub, levels=[-t, t], colors="tab:red", linewidths=0.9,
                   linestyles="--")
    ax.set_xticks(x - dx / 2)
    ax.set_yticks(y - dy / 2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="0.6", lw=0.4, alpha=0.7)
    ax.tick_params(length=0)
    w = (r[f"wl_{CUTS[1]}"][k] + r[f"ws_{CUTS[1]}"][k]) / dx
    ax.set_title(rf"$y/H={y_target:.2f}$:  {w:.2f} cells", fontsize=9, pad=3)
    ax.set_aspect("equal", adjustable="box")


def main(argv=None) -> None:
    a = parse_args(argv)
    d = load(a.run)
    r = analyse(d, a.v_cut)
    dx = d["dx"]
    report(d, r)

    fig = plt.figure(figsize=(6.6, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])
    ax0 = fig.add_subplot(gs[:, 0])
    axz = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

    shades = ["#c6dbef", "#6baed6", "#2171b5"]
    for c, col in zip(reversed(CUTS), shades):
        ax0.fill_betweenx(r["y"], -r[f"wl_{c}"] / dx, r[f"ws_{c}"] / dx,
                          color=col, lw=0, label=rf"$f_l={c:g}\ldots{1-c:g}$")
    ax0.axvline(0.0, color="k", lw=0.9)
    # The threshold on the damped band is a fraction of the local speed, so it says
    # nothing where the two convection cells meet and the fluid is barely moving at
    # all: every threshold then sits far from the front for want of any flow to damp
    ok = r["v_ref"] > 0.25 * np.median(r["v_ref"])
    w = np.where(ok, -r["w_damp"] / dx, np.nan)
    ax0.plot(w, r["y"], color="tab:red", lw=1.1, ls="--",
             label=rf"damped, ${int(a.v_cut*100)}\,\%$ of $v$ gone")
    for s in (-1, 1, -2, 2):
        ax0.axvline(s, color="0.75", lw=0.5, ls=":")
    ax0.set_xlabel("cells from the front")
    ax0.set_ylabel(r"$y/H$")
    ax0.set_xlim(-3.4, 1.6)
    ax0.set_ylim(0.0, 1.0)
    ax0.legend(frameon=False, loc="lower left", handlelength=1.3,
           borderaxespad=0.6)
    ax0.text(-0.09, 1.02, "a", transform=ax0.transAxes, fontsize=12,
             fontweight="bold", ha="center")
    ax0.annotate("liquid", xy=(-3.2, 0.97), fontsize=8, color="0.4")
    ax0.annotate("ice", xy=(1.15, 0.97), fontsize=8, color="0.4")

    for ax, yt in zip(axz, a.zoom_at):
        draw_zoom(ax, d, r, yt, a.zoom_cells)
    axz[0].text(-0.13, 1.16, "b", transform=axz[0].transAxes, fontsize=12,
                fontweight="bold", ha="center")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {a.out}")
    if a.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
