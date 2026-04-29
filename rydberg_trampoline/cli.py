"""Command-line entry point.

Usage::

    rydberg-trampoline figures all          # generate every hero figure
    rydberg-trampoline figures decay        # one specific figure
    rydberg-trampoline backends             # list available backends

This is a thin wrapper that defers to the per-figure scripts so they remain
runnable as ``python -m rydberg_trampoline.figures.fig_<name>`` for users
who want full control of the per-figure CLI flags.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from rydberg_trampoline.backends import available_backends


_FIGURES = {
    "decay": "fig_decay_traces",
    "gamma": "fig_gamma_vs_inv_delta",
    "resonance": "fig_resonance_scan",
    "bubbles": "fig_bubble_histogram",
    "imperfection": "fig_imperfection_sensitivity",
}


def _run_figure(stem: str, extra_args: list[str]) -> int:
    module = importlib.import_module(f"rydberg_trampoline.figures.{stem}")
    return int(module.main(extra_args))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rydberg-trampoline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fig_p = sub.add_parser("figures", help="generate hero figures")
    fig_p.add_argument(
        "which",
        choices=["all"] + list(_FIGURES.keys()),
        help="figure name (or 'all')",
    )
    fig_p.add_argument("rest", nargs=argparse.REMAINDER, help="passed to the figure script")

    sub.add_parser("backends", help="list installed backends")

    args = parser.parse_args(argv)

    if args.cmd == "backends":
        for name in available_backends():
            print(name)
        return 0

    if args.cmd == "figures":
        rest = list(args.rest)
        # argparse REMAINDER may include a leading "--"; strip it.
        if rest and rest[0] == "--":
            rest = rest[1:]
        if args.which == "all":
            rc = 0
            for stem in _FIGURES.values():
                rc |= _run_figure(stem, rest)
            return rc
        return _run_figure(_FIGURES[args.which], rest)

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
