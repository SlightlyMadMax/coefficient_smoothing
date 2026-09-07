#!/usr/bin/env python
"""
Parameter sweep driver for the freezing refinement study.

Each case runs in its own subprocess, so a failure — including a hard crash inside
numba or pyamg — is recorded and the sweep carries on with the remaining cases.

For a warm start the precursor steady state is grid-dependent, so it is generated
per grid before the production run (unless it already exists).

Examples
--------
Grid refinement at fixed dt, warm start, up to t = 2340 s:
    python -m src.examples.water_freezing.sweep --grids 51 101 151 201

Time-step refinement on the production grid:
    python -m src.examples.water_freezing.sweep --grids 151 --dts 0.04 0.02 0.01 0.005

Smoothing width:
    python -m src.examples.water_freezing.sweep --grids 151 --eps-t 0.2 0.1 0.05 0.025

Penalty coefficient:
    python -m src.examples.water_freezing.sweep --grids 151 --penalty-c 1e5 1e6 1e7 1e8

Show what would run without running it:
    python -m src.examples.water_freezing.sweep --grids 51 101 151 201 --dry-run
"""

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep grid / dt / eps / C for the freezing benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[51, 101, 151, 201],
        help="grid sizes as NUMBER OF NODES (square grids); 51 nodes = 50 intervals",
    )
    p.add_argument("--dts", type=float, nargs="+", default=[0.01], help="time steps [s]")
    p.add_argument(
        "--eps-t", type=float, nargs="+", default=[None], help="smoothing half-width [K]"
    )
    p.add_argument(
        "--penalty-c", type=float, nargs="+", default=[None], help="penalty C [1/s]"
    )
    p.add_argument("--end-time", type=float, default=2340.0, help="final time [s]")
    p.add_argument("--start", choices=["cold", "warm"], default="warm")
    p.add_argument(
        "--amg-rebuild-every",
        type=int,
        default=10,
        help="AMG hierarchy reuse interval; 10 is ~1.8x faster than 1 and agrees "
        "with it to ~1e-7 relative",
    )
    p.add_argument(
        "--cold-wall-ramp",
        type=float,
        default=0.0,
        help="ramp the cold wall down over this many seconds instead of stepping it; "
        "a short ramp removes the start-up instability seen on fine grids",
    )
    p.add_argument(
        "--precursor-dt", type=float, default=0.5, help="time step for the warm-start run"
    )
    p.add_argument(
        "--precursor-max-time",
        type=float,
        default=3600.0,
        help="time limit for the warm-start run [s]",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=HERE / "data" / "refinement" / "summary.csv",
        help="every case appends its result here",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-case wall-clock limit [s]; no limit by default",
    )
    p.add_argument("--tag", type=str, default=None, help="label added to every outdir")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a case whose summary.json is already present",
    )
    p.add_argument("--dry-run", action="store_true", help="print commands only")
    return p.parse_args(argv)


def run_case(cmd: list[str], timeout: float | None, dry_run: bool) -> tuple[bool, str]:
    printable = " ".join(cmd[2:])  # drop the interpreter and -m
    print(f"    $ python -m {printable}", flush=True)
    if dry_run:
        return True, "dry-run"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout:g} s"
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit code {proc.returncode}"
        print(f"      FAILED after {dt:.1f} s: {detail}", flush=True)
        return False, detail
    print(f"      ok, {dt:.1f} s", flush=True)
    return True, f"{dt:.1f} s"


def main(argv=None) -> None:
    args = parse_args(argv)
    py = sys.executable

    cases = list(itertools.product(args.grids, args.dts, args.eps_t, args.penalty_c))
    print(f"{len(cases)} case(s), start={args.start}, end_time={args.end_time:g} s\n")

    results = []
    for idx, (n, dt, eps_t, pc) in enumerate(cases, 1):
        label = f"grid={n}x{n} dt={dt:g}"
        if eps_t is not None:
            label += f" eps_T={eps_t:g}"
        if pc is not None:
            label += f" C={pc:.0e}"
        print(f"[{idx}/{len(cases)}] {label}")

        if args.skip_existing:
            probe = list(
                (HERE / "data" / "refinement").glob(
                    f"{args.start}_{n}x{n}_dt{dt:g}_*/summary.json"
                )
            )
            if probe:
                print("      skipped (summary.json exists)\n", flush=True)
                results.append((label, True, "skipped"))
                continue

        # --- warm-start precursor, one per grid -----------------------------
        if args.start == "warm":
            precursor = HERE / "data" / "warm_start" / f"steady_{n}x{n}.npz"
            if precursor.exists():
                print(f"    precursor present: {precursor.name}", flush=True)
            else:
                ok, detail = run_case(
                    [
                        py,
                        "-m",
                        "src.examples.water_freezing.prepare_warm_start",
                        "--nx", str(n),
                        "--ny", str(n),
                        "--dt", str(args.precursor_dt),
                        "--max-time", str(args.precursor_max_time),
                    ],
                    args.timeout,
                    args.dry_run,
                )
                if not ok:
                    print(f"      -> skipping {label}: precursor failed\n", flush=True)
                    results.append((label, False, f"precursor: {detail}"))
                    continue

        # --- production run --------------------------------------------------
        cmd = [
            py,
            "-m",
            "src.examples.water_freezing.run",
            "--nx", str(n),
            "--ny", str(n),
            "--dt", str(dt),
            "--end-time", str(args.end_time),
            "--start", args.start,
            "--amg-rebuild-every", str(args.amg_rebuild_every),
            "--summary-csv", str(args.summary_csv),
            "--log-interval", "300",
        ]
        if eps_t is not None:
            cmd += ["--eps-t", str(eps_t)]
        if pc is not None:
            cmd += ["--penalty-c", str(pc)]
        if args.cold_wall_ramp > 0.0:
            cmd += ["--cold-wall-ramp", str(args.cold_wall_ramp)]
        if args.tag:
            cmd += ["--tag", args.tag]

        ok, detail = run_case(cmd, args.timeout, args.dry_run)
        results.append((label, ok, detail))
        print(flush=True)

    print("=" * 72)
    print("SWEEP SUMMARY")
    print("=" * 72)
    for label, ok, detail in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:<44} {detail}")
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_ok}/{len(results)} case(s) succeeded")
    if not args.dry_run:
        print(f"Per-case metrics: {args.summary_csv}")
    if n_ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
