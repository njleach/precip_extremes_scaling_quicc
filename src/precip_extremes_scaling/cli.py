"""Command-line interface for precip extreme scaling."""

from __future__ import annotations

import argparse
import json

import numpy as np

from precip_extremes_scaling import scaling


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Estimate precipitation from vertical profiles."
    )
    parser.add_argument(
        "--omega",
        required=True,
        help="JSON array of omega values in Pa/s.",
    )
    parser.add_argument(
        "--temp",
        required=True,
        help="JSON array of temperature values in K.",
    )
    parser.add_argument(
        "--plev",
        required=True,
        help="JSON array of pressure values in Pa, highest pressure first.",
    )
    parser.add_argument(
        "--ps",
        required=True,
        type=float,
        help="Surface pressure in Pa.",
    )
    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    precip = scaling(
        np.asarray(json.loads(args.omega), dtype=float),
        np.asarray(json.loads(args.temp), dtype=float),
        np.asarray(json.loads(args.plev), dtype=float),
        args.ps,
    )
    print(f"{precip:.12g}")
