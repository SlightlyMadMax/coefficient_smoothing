#!/usr/bin/env python
"""
Locate the largest stable time step, and find out what actually limits it.

Two candidate mechanisms are separated by construction:

  * the wall vorticity condition (Thom, 1st order, vs Woods/Jensen, 2nd order);
  * the penalty term, which enters the vorticity right-hand side explicitly, so its
    stability budget should scale like 1/C.

If the limit tracks 1/C the penalty is responsible; if it barely moves with C but jumps
between the two boundary conditions, the boundary condition is.

Each case runs in a subprocess so that a divergence cannot take the probe down with it.

Example
-------
    python -m src.examples.water_freezing.dt_limit_probe --nx 151 --horizon 60
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

DEFAULT_DTS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find the stability limit in dt and attribute it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nx", type=int, default=151)
    p.add_argument("--horizon", type=float, default=60.0, help="physical time [s]")
    p.add_argument("--start", choices=["cold", "warm"], default="warm")
    p.add_argument("--bc-orders", type=int, nargs="+", default=[1, 2])
    p.add_argument(
        "--penalty-c", type=float, nargs="+", default=[1e4, 1e5, 1e6],
        help="penalty coefficients to scan",
    )
    p.add_argument("--dts", type=float, nargs="+", default=DEFAULT_DTS)
    p.add_argument("--cold-wall-ramp", type=float, default=0.0)
    p.add_argument(
        "--sf-tolerance", type=float, nargs="+", default=[1e-6],
        help="stream-function solve tolerances to scan; a limit that moves with this "
        "points at cancellation between the predictor and the corrector",
    )
    p.add_argument("--timeout", type=float, default=900.0, help="per case [s]")
    return p.parse_args(argv)


def try_case(nx, dt, bc, c, args, sf_tol: float = 1e-6) -> tuple[bool, str]:
    cmd = [
        sys.executable, "-u", "-m", "src.examples.water_freezing.run",
        "--nx", str(nx), "--ny", str(nx),
        "--dt", str(dt),
        "--end-time", str(args.horizon),
        "--start", args.start,
        "--vorticity-bc-order", str(bc),
        "--penalty-c", str(c),
        "--sf-tolerance", str(sf_tol),
        "--amg-rebuild-every", "10",
        "--quiet", "--no-save-final",
        "--outdir", str(HERE / "data" / "cold_start" / "dt_probe"
                        / f"n{nx}_bc{bc}_C{c:.0e}_tol{sf_tol:.0e}_dt{dt:g}"),
        "--summary-csv", str(HERE / "data" / "cold_start" / "dt_probe" / "summary.csv"),
    ]
    if args.cold_wall_ramp > 0:
        cmd += ["--cold-wall-ramp", str(args.cold_wall_ramp)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=REPO, timeout=args.timeout,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"timeout >{args.timeout:g}s"
    el = time.perf_counter() - t0
    if proc.returncode != 0:
        blob = (proc.stderr or "") + (proc.stdout or "")
        if "FloatingPointError" in blob:
            reason = "diverged"
        else:
            tail = blob.strip().splitlines()
            reason = tail[-1][:60] if tail else f"exit {proc.returncode}"
        return False, f"{reason} ({el:.0f}s)"
    return True, f"stable ({el:.0f}s)"


def main(argv=None) -> None:
    args = parse_args(argv)
    (HERE / "data" / "cold_start" / "dt_probe").mkdir(parents=True, exist_ok=True)
    dts = sorted(args.dts, reverse=True)

    print(f"grid {args.nx}x{args.nx}, start={args.start}, horizon={args.horizon:g} s, "
          f"ramp={args.cold_wall_ramp:g} s", flush=True)
    print("scanning dt from large to small; first stable value is the limit\n", flush=True)

    limits = {}
    for bc in args.bc_orders:
        for tol in args.sf_tolerance:
            for c in args.penalty_c:
                label = f"bc={bc} C={c:.0e} tol={tol:.0e}"
                limit = None
                for dt in dts:
                    ok, detail = try_case(args.nx, dt, bc, c, args, sf_tol=tol)
                    print(f"  {label}  dt={dt:<7g} {'OK  ' if ok else 'FAIL'}  {detail}",
                          flush=True)
                    if ok:
                        limit = dt
                        break
                limits[(bc, c, tol)] = limit
                print(f"  -> {label}: largest stable dt = "
                      f"{limit if limit is not None else 'none in range'}\n", flush=True)

    print("=" * 64)
    print("LARGEST STABLE dt [s]")
    print("=" * 64)
    print("  bc  sf_tol     " + "".join(f"{c:>12.0e}" for c in args.penalty_c))
    for bc in args.bc_orders:
        for tol in args.sf_tolerance:
            row = f"  {bc:<4}{tol:<11.0e}"
            for c in args.penalty_c:
                v = limits[(bc, c, tol)]
                row += f"{(v if v is not None else float('nan')):>12g}"
            print(row)
    print("\nScaling with C points at the explicit penalty term; scaling with the")
    print("solve tolerance points at cancellation between predictor and corrector.")


if __name__ == "__main__":
    main()
