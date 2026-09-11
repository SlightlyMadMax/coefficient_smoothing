#!/usr/bin/env python
"""
Compare the FVM velocity-pressure solver against this code, over the PIV frame.

The two codes solve the same benchmark with different formulations — finite volumes in
primitive variables against finite differences in stream function and vorticity — so
overlaying their T = T_m contours at the comparison time is a genuine cross-check of
both, independent of the experiment.

The FVM checkpoint stores the field transposed relative to this code: its first index
runs along the hot-to-cold direction. It also stores dimensionless temperature on a
unit-square mesh, so the axes are scaled by the cavity width here.

Example
-------
    python -m src.examples.water_freezing.compare_fvm \
        --fvm C:/Users/ZZZ/Desktop/water_freezing_data/151x151_test \
        --baseline data/cold_start/parametric/151x151_dt0.02_e0.1_ef0.1_C1e+06
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.size": 12, "axes.labelsize": 10, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9,
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "lines.linewidth": 1.6, "figure.dpi": 300,
})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay the FVM and stream-function solutions on the PIV frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fvm", type=Path, required=True,
                   help="FVM run directory holding manifest.json, mesh.npz and a checkpoint")
    p.add_argument("--baseline", type=Path,
                   default=HERE / "data" / "cold_start" / "parametric"
                   / "151x151_dt0.02_e0.1_ef0.1_C1e+06",
                   help="run directory of this code to compare against")
    p.add_argument("--piv", type=Path, default=HERE / "data" / "inputs" / "kowalewski.png",
                   help="PIV frame to use as background; pass 'none' to omit it")
    p.add_argument("--width", type=float, default=0.038, help="cavity width [m]")
    p.add_argument("--out", type=Path, default=HERE / "graphs" / "fvm" / "interface_fvm_vs_sfw.png")
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-flip-y", action="store_true",
                   help="keep the FVM y axis as stored instead of matching orientations "
                   "automatically")
    return p.parse_args(argv)


def contour_zero(field: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Longest theta = 0 contour, in the units of x and y."""
    fig = plt.figure()
    try:
        segs = fig.gca().contour(x, y, field, levels=[0.0]).allsegs[0]
    finally:
        plt.close(fig)
    if not segs:
        return np.array([]), np.array([])
    s = max(segs, key=len)
    return s[:, 0], s[:, 1]


def load_fvm(run: Path, width: float, auto_flip: bool):
    manifest = json.load(open(run / "manifest.json", encoding="utf-8"))
    mesh = np.load(run / "mesh.npz", allow_pickle=True)
    cps = sorted(run.glob("checkpoint_*.npz"),
                 key=lambda f: int(f.stem.split("_")[1]))
    if not cps:
        raise SystemExit(f"No checkpoint in {run}")
    with np.load(cps[-1], allow_pickle=True) as d:
        theta = d["field_theta"]
        t = float(d["t"])

    c0 = 0.5 * (mesh["faces_0"][:-1] + mesh["faces_0"][1:]) * width
    c1 = 0.5 * (mesh["faces_1"][:-1] + mesh["faces_1"][1:]) * width

    # The first index runs hot-to-cold, i.e. along x, so the stored array is the
    # transpose of the [y, x] layout used here.
    field = theta.T
    x, y = c0, c1

    if auto_flip:
        # Both y walls are adiabatic, so the stored y direction is ambiguous. Resolve it
        # by where the ice is thickest: this benchmark grows more ice at the bottom.
        ice_low = float((field[: field.shape[0] // 2] < 0.0).mean())
        ice_high = float((field[field.shape[0] // 2:] < 0.0).mean())
        if ice_high > ice_low:
            field = field[::-1]
    return field, x, y, t, manifest


def load_baseline(run: Path):
    meta = json.load(open(run / "summary.json", encoding="utf-8"))
    cps = sorted(run.glob("checkpoint_*.npz"),
                 key=lambda f: int(f.stem.split("_")[1]))
    with np.load(cps[-1], allow_pickle=True) as d:
        u = d["u"]
        t = float(d["t"])
    n_y, n_x = u.shape
    x = np.linspace(0.0, meta["dx"] * (n_x - 1), n_x)
    y = np.linspace(0.0, meta["dy"] * (n_y - 1), n_y)
    return u, x, y, t, meta


def main(argv=None) -> None:
    a = parse_args(argv)
    f_field, f_x, f_y, f_t, f_meta = load_fvm(a.fvm, a.width, not a.no_flip_y)
    b_field, b_x, b_y, b_t, b_meta = load_baseline(a.baseline)

    xf, yf = contour_zero(f_field, f_x, f_y)
    xb, yb = contour_zero(b_field, b_x, b_y)
    if xf.size == 0 or xb.size == 0:
        raise SystemExit("No T = T_m contour in one of the solutions")

    p = f_meta["params"]
    print(f"FVM      : {p['n']}x{p['n']}, dt={p['dt']:g} s, C={p['darcy_c']:.0e}, "
          f"t={f_t:.1f} s   ice fraction {float((f_field < 0).mean()):.5f}")
    print(f"this code: {b_meta['n_x']}x{b_meta['n_y']}, dt={b_meta['dt']:g} s, "
          f"C={b_meta['penalty_C']:.0e}, t={b_t:.1f} s   "
          f"ice fraction {b_meta['ice_fraction']:.5f}")

    # Deviation between the two interfaces, measured at common heights
    lo, hi = max(yf.min(), yb.min()), min(yf.max(), yb.max())
    yy = np.linspace(lo, hi, 500)
    xi_f = np.interp(yy, *(lambda o: (yf[o], xf[o]))(np.argsort(yf)))
    xi_b = np.interp(yy, *(lambda o: (yb[o], xb[o]))(np.argsort(yb)))
    d = xi_f - xi_b
    print(f"\ninterface difference (FVM minus this code):")
    print(f"  mean {d.mean()*1e3:+.3f} mm   RMS {np.sqrt(np.mean(d**2))*1e3:.3f} mm   "
          f"max |d| {np.abs(d).max()*1e3:.3f} mm")
    print(f"  RMS = {np.sqrt(np.mean(d**2))/a.width*100:.2f} % of the cavity width, "
          f"{np.sqrt(np.mean(d**2))/(b_meta['dx']):.2f} cells")

    fig, ax = plt.subplots(figsize=(3.6, 3.4), constrained_layout=True)
    use_piv = str(a.piv).lower() != "none" and Path(a.piv).exists()
    if use_piv:
        ax.imshow(plt.imread(a.piv), extent=[0.0, a.width, 0.0, a.width])
    ax.plot(xb, yb, color="tab:red",
            label=rf"stream function, $\tau={b_meta['dt']:g}$ s")
    ax.plot(xf, yf, color="yellow" if use_piv else "tab:blue", linestyle="--",
            label=rf"FVM, $\tau={p['dt']:g}$ s")
    ax.set_xlabel(r"$x$, m")
    ax.set_ylabel(r"$y$, m")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, a.width)
    ax.set_ylim(0.0, a.width)
    if use_piv:
        ax.legend(loc="upper left", frameon=True, framealpha=0.85,
                  facecolor="white", edgecolor="none")
    else:
        ax.legend(loc="upper left", frameon=False)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {a.out}")
    if a.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
