#!/usr/bin/env python
"""
Extract the experimental ice-water interface from the PIV frame.

The published comparison overlays the computed interface on the PIV photograph, which
can only be judged by eye. Turning the photographed ice edge into a curve makes the
comparison quantitative: the two interfaces can be plotted as lines and the deviation
reported in millimetres.

In the frame the ice occupies the bright region against the cold wall on the right,
while the water carries dark PIV streaks. The edge is found per row by scanning inward
from the cold wall to the first sustained drop in brightness, which is robust to the
bright particle tracks that speckle the liquid.

Examples
--------
    python -m src.examples.water_freezing.piv_interface --check
    python -m src.examples.water_freezing.piv_interface --compare \
        data/warm_start/refinement_dt0.01/warm_151x151_dt0.01_epsT0.1_C1e+06
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
    "lines.linewidth": 1.5, "figure.dpi": 300,
})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Digitise the ice edge from the PIV frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", type=Path,
                   default=HERE / "data" / "inputs" / "kowalewski.png")
    p.add_argument("--width", type=float, default=0.038, help="cavity width [m]")
    p.add_argument("--height", type=float, default=0.038, help="cavity height [m]")
    p.add_argument(
        "--threshold", type=float, default=0.55,
        help="brightness below which a pixel counts as water",
    )
    p.add_argument(
        "--run-length", type=int, default=6,
        help="consecutive water pixels required to accept the edge; rejects the "
        "isolated dark specks inside the ice",
    )
    p.add_argument(
        "--smooth", type=int, default=9,
        help="running-median window in rows; 0 disables",
    )
    p.add_argument(
        "--trim", type=float, default=0.03,
        help="fraction of the frame height dropped at the top and bottom, where the "
        "cavity walls and the frame border corrupt the brightness profile",
    )
    p.add_argument(
        "--reject", type=float, default=2.0,
        help="reject an edge point deviating from the local trend by more than this "
        "many millimetres; removes the printed 'ICE' label and border artefacts",
    )
    p.add_argument("--out", type=Path,
                   default=HERE / "data" / "inputs" / "piv_interface.npz")
    p.add_argument("--check", action="store_true",
                   help="write a diagnostic overlay so the extraction can be verified")
    p.add_argument("--compare", type=Path, default=None,
                   help="run directory whose computed interface to compare against")
    p.add_argument("--outdir", type=Path, default=HERE / "graphs" / "piv")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def extract(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Return the ice edge as (x, y) in metres, ordered by y."""
    img = plt.imread(args.image)
    grey = img.mean(axis=2) if img.ndim == 3 else img
    n_row, n_col = grey.shape

    r_lo = int(round(args.trim * n_row))
    r_hi = n_row - r_lo

    xs, ys = [], []
    for r in range(r_lo, r_hi):
        row = grey[r]
        water = row < args.threshold
        # Walk left from the cold wall; the edge is where water starts and persists
        edge = None
        for c in range(n_col - 1, args.run_length - 1, -1):
            if water[c - args.run_length:c].all():
                edge = c
                break
        # No sustained water run means the row is all ice or all frame border; an
        # edge pinned against the cold wall is the same failure, not a measurement.
        if edge is None or edge >= n_col - 1 - args.run_length:
            continue
        xs.append(edge / (n_col - 1) * args.width)
        # row 0 is the top of the frame, y increases upward
        ys.append((1.0 - r / (n_row - 1)) * args.height)

    x = np.asarray(xs)
    y = np.asarray(ys)
    order = np.argsort(y)
    x, y = x[order], y[order]

    def running_median(v: np.ndarray, k: int) -> np.ndarray:
        k = max(3, k | 1)
        pad = k // 2
        padded = np.pad(v, pad, mode="edge")
        return np.array([np.median(padded[i:i + k]) for i in range(v.size)])

    # The printed "ICE" label and the frame border produce edge points far inside the
    # ice. They are local, so a wide running median tracks the true edge and the
    # offenders can be dropped against it.
    if args.reject > 0 and x.size > 20:
        trend = running_median(x, max(21, x.size // 15))
        keep = np.abs(x - trend) <= args.reject * 1e-3
        dropped = int((~keep).sum())
        if dropped:
            print(f"  rejected {dropped} outlier point(s) "
                  f"(> {args.reject:g} mm from the local trend)")
        x, y = x[keep], y[keep]

    if args.smooth and args.smooth > 1 and x.size > args.smooth:
        x = running_median(x, args.smooth)
    return x, y


def computed_interface(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = json.load(open(run_dir / "summary.json", encoding="utf-8"))
    cps = sorted(run_dir.glob("checkpoint_*.npz"),
                 key=lambda f: int(f.stem.split("_")[1]))
    with np.load(cps[-1], allow_pickle=True) as d:
        u = d["u"]
    n_y, n_x = u.shape
    xs = np.linspace(0.0, meta["dx"] * (n_x - 1), n_x)
    ys = np.linspace(0.0, meta["dy"] * (n_y - 1), n_y)
    fig = plt.figure()
    try:
        segs = fig.gca().contour(xs, ys, u, levels=[0.0]).allsegs[0]
    finally:
        plt.close(fig)
    if not segs:
        raise SystemExit(f"No T = T_m contour found in {run_dir}")
    seg = max(segs, key=len)
    return seg[:, 0], seg[:, 1], meta


def main(argv=None) -> None:
    args = parse_args(argv)
    x, y = extract(args)
    if x.size == 0:
        raise SystemExit("No interface found; adjust --threshold")

    np.savez_compressed(args.out, x=x, y=y,
                        threshold=args.threshold, source=str(args.image))
    print(f"Extracted {x.size} edge points -> {args.out}")
    print(f"  x range [{x.min()*1e3:.2f}, {x.max()*1e3:.2f}] mm, "
          f"mean {x.mean()*1e3:.2f} mm")

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.check:
        fig, ax = plt.subplots(figsize=(3.4, 3.4), constrained_layout=True)
        ax.imshow(plt.imread(args.image), extent=[0, args.width, 0, args.height])
        ax.plot(x, y, color="red", linewidth=1.2, label="extracted edge")
        ax.set_xlabel(r"$x$, m")
        ax.set_ylabel(r"$y$, m")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", frameon=True, framealpha=0.85,
                  facecolor="white", edgecolor="none")
        out = args.outdir / "piv_extraction_check.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  diagnostic overlay -> {out}")
        if not args.show:
            plt.close(fig)

    if args.compare is not None:
        xc, yc, meta = computed_interface(args.compare)
        # Deviation measured at common heights
        lo = max(y.min(), yc.min())
        hi = min(y.max(), yc.max())
        yy = np.linspace(lo, hi, 400)
        oe = np.argsort(y)
        oc = np.argsort(yc)
        xe = np.interp(yy, y[oe], x[oe])
        xn = np.interp(yy, yc[oc], xc[oc])
        d = xn - xe
        rms = float(np.sqrt(np.mean(d ** 2)))
        print(f"\nComputed vs experimental interface "
              f"({meta['n_x']}x{meta['n_y']}, dt={meta['dt']:g} s):")
        print(f"  mean deviation {d.mean()*1e3:+.3f} mm")
        print(f"  RMS deviation  {rms*1e3:.3f} mm  "
              f"({rms/args.width*100:.2f} % of the cavity width)")
        print(f"  max |deviation| {np.abs(d).max()*1e3:.3f} mm")

        fig, ax = plt.subplots(figsize=(3.4, 3.15), constrained_layout=True)
        ax.plot(xe, yy, color="0.35", linestyle="--",
                label="experiment (PIV frame)")
        ax.plot(xn, yy, color="tab:red",
                label=rf"computed, ${meta['n_x']}\times{meta['n_y']}$")
        ax.set_xlabel(r"$x$, m")
        ax.set_ylabel(r"$y$, m")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.0, args.width)
        ax.set_ylim(0.0, args.height)
        ax.legend(loc="upper left", frameon=False)
        ax.annotate(rf"RMS $= {rms*1e3:.2f}$ mm", xy=(0.04, 0.06),
                    xycoords="axes fraction", fontsize=9)
        out = args.outdir / "piv_vs_computed.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  figure -> {out}")
        if not args.show:
            plt.close(fig)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
