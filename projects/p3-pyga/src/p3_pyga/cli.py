from __future__ import annotations

import argparse
from pathlib import Path

from p3_pyga.plots import generate_all_plots
from p3_pyga.suite import run_benchmark


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Project 3 — PyGAD benchmarks vs custom P1/P2 implementations",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="Run full comparison suite")
    bench.add_argument("--quick", action="store_true", help="Fewer epochs and configs")
    bench.add_argument("--output", type=Path, default=root, help="Project root for results/")

    plot_cmd = sub.add_parser("plot", help="Generate comparison charts from latest benchmark")
    plot_cmd.add_argument("--output", type=Path, default=root, help="Project root")

    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark(output_dir=args.output, quick=args.quick)
    elif args.command == "plot":
        generate_all_plots(project_root=args.output)


if __name__ == "__main__":
    main()
