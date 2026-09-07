#!/usr/bin/env python
"""
The two-sided cost of the penalty coefficient.

Choosing C is a compromise. Too small and the solid is not rigid: fluid keeps moving
where it should not. Too large and the damping reaches past the diffuse interface into
the liquid, weakening the very circulation the model is meant to resolve. This plots
both sides against C so the plateau between them can be read off.

Panel (a) is the solid-side failure: the largest residual speed inside the ice, which
should fall as 1/C while the penalty behaves as intended.

Panel (b) is the liquid-side failure: the peak stream function and the Nusselt number
at the hot wall, both far from the interface, normalised to the smallest C. A monotone
decline there is damping leaking out of the solid.

Example
-------
    python -m src.examples.water_freezing.penalty_diagnostics
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.size": 12, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "lines.linewidth": 1.5, "figure.dpi": 300,
})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Residual motion in the solid and damping leakage into the liquid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=HERE / "data" / "star")
    p.add_argument(
        "--pattern", default="151x151_dt0.02_e0.1_ef0.1_C*",
        help="glob selecting the runs of the penalty axis",
    )
    p.add_argument("--out", type=Path, default=None,
                   help="output path; defaults to graphs/penalty_<panel>.png")
    p.add_argument(
        "--panel", choices=["a", "b", "both"], default="both",
        help="a = residual speed in the solid, b = circulation and Nusselt number, "
        "both = the two side by side",
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def add_subfigure_label(ax, label: str) -> None:
    ax.text(0.5, 1.04, label, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", zorder=10)


def collect(data_dir: Path, pattern: str) -> list[dict]:
    out = []
    for d in sorted(glob.glob(str(data_dir / pattern))):
        f = os.path.join(d, "summary.json")
        cps = sorted(glob.glob(os.path.join(d, "checkpoint_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if not os.path.exists(f) or not cps:
            continue
        m = json.load(open(f, encoding="utf-8"))
        with np.load(cps[-1], allow_pickle=True) as z:
            u, sf, vx, vy = z["u"], z["sf"], z["v_x"], z["v_y"]
        solid = u < 0.0
        speed = np.hypot(vx, vy)
        out.append({
            "C": m["penalty_C"],
            "v_solid": float(speed[solid].max()) if solid.any() else np.nan,
            "v_liquid": float(speed[~solid].max()),
            "psi_max": float(np.abs(sf).max()),
            "Nu_hot": m["Nu_hot"],
            "iface": m["mean_interface_x"],
            "ice": m["ice_fraction"],
        })
    out.sort(key=lambda r: r["C"])
    return out


def main(argv=None) -> None:
    a = parse_args(argv)
    rows = collect(a.data_dir, a.pattern)
    if len(rows) < 2:
        raise SystemExit(f"Need at least two runs matching {a.pattern}")

    C = np.array([r["C"] for r in rows])
    v_solid = np.array([r["v_solid"] for r in rows])
    psi = np.array([r["psi_max"] for r in rows])
    nu = np.array([r["Nu_hot"] for r in rows])

    print(f"{'C':>9}{'max|v| solid':>14}{'max|psi|':>12}{'Nu_hot':>10}"
          f"{'iface_x[m]':>13}{'ice_frac':>10}")
    for r in rows:
        print(f"{r['C']:>9.0e}{r['v_solid']:>14.4e}{r['psi_max']:>12.6f}"
              f"{r['Nu_hot']:>10.4f}{r['iface']:>13.6f}{r['ice']:>10.5f}")

    # Fitted slope of the residual speed: an exact -1 means the penalty is doing
    # precisely what a Darcy term should
    slope = np.polyfit(np.log10(C), np.log10(v_solid), 1)[0]
    print(f"\nresidual speed in the solid scales as C^{slope:.3f}")
    print(f"circulation change over the range: "
          f"max|psi| {(psi[-1] / psi[0] - 1) * 100:+.2f} %, "
          f"Nu_hot {(nu[-1] / nu[0] - 1) * 100:+.2f} %")

    def draw_solid(ax, label: str | None) -> None:
        ax.loglog(C, v_solid, "o-", color="tab:red", label=r"max$|v|$ in the ice")
        ax.loglog(C, v_solid[0] * C[0] / C, ":", color="0.5", lw=1.0,
                  label=r"$\propto 1/C$")
        ax.set_xlabel(r"$C$, s$^{-1}$")
        ax.set_ylabel(r"residual $|v|$ in the solid")
        ax.legend(frameon=False, loc="upper right")
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        if label:
            add_subfigure_label(ax, label)

    def draw_liquid(ax, label: str | None) -> None:
        ax.semilogx(C, psi / psi[0], "s-", color="tab:blue", label=r"max$|\psi|$")
        ax.semilogx(C, nu / nu[0], "^-", color="tab:green",
                    label=r"$\mathrm{Nu}$, hot wall")
        ax.set_xlabel(r"$C$, s$^{-1}$")
        ax.set_ylabel("normalised to the smallest $C$")
        ax.legend(frameon=False, loc="lower left")
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        if label:
            add_subfigure_label(ax, label)

    if a.panel == "both":
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.3, 2.9),
                                       constrained_layout=True)
        draw_solid(ax0, "a")
        draw_liquid(ax1, "b")
        default_name = "penalty_diagnostics.png"
    else:
        fig, ax = plt.subplots(figsize=(3.4, 2.9), constrained_layout=True)
        (draw_solid if a.panel == "a" else draw_liquid)(ax, None)
        default_name = f"penalty_panel_{a.panel}.png"

    out = a.out or (HERE / "graphs" / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {out}")
    if a.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
