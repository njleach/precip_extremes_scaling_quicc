"""Benchmark scaling implementations and refresh the README section."""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from precip_extremes_scaling import scaling, scaling_nb

README_PATH = Path("README.md")
README_START = "<!-- benchmarks:start -->"
README_END = "<!-- benchmarks:end -->"

N_SAMPLES = 1000
N_LEVELS = 10
N_ROUNDS = 5
SEED = 0

OMEGA_RANGE = (-0.3, 0.2)
TEMP_RANGE = (220.0, 270.0)
PLEV_RANGE = (10_000.0, 100_000.0)
PLEV_JITTER_STD = 2_500.0
TEMP_NOISE_STD = 1.5
OMEGA_NOISE_STD = 0.02


@dataclass(frozen=True)
class Profile:
    """Single benchmark profile."""

    omega: np.ndarray
    temp: np.ndarray
    plev: np.ndarray
    ps: float


def build_profiles(
    n_samples: int = N_SAMPLES,
    n_levels: int = N_LEVELS,
    seed: int = SEED,
) -> list[Profile]:
    """Build random vertical profiles within the README example ranges."""
    rng = np.random.default_rng(seed)
    profiles: list[Profile] = []
    plev_base = np.linspace(PLEV_RANGE[1], PLEV_RANGE[0], n_levels)

    for _ in range(n_samples):
        plev = np.clip(
            plev_base + rng.normal(0.0, PLEV_JITTER_STD, size=n_levels),
            *PLEV_RANGE,
        )
        plev = np.ascontiguousarray(np.sort(plev)[::-1])

        temp_surface = rng.uniform(255.0, TEMP_RANGE[1])
        temp_top = rng.uniform(TEMP_RANGE[0], temp_surface - 5.0)
        temp = np.linspace(temp_surface, temp_top, n_levels)
        temp = np.clip(
            temp + rng.normal(0.0, TEMP_NOISE_STD, size=n_levels),
            *TEMP_RANGE,
        )
        temp = np.ascontiguousarray(np.sort(temp)[::-1])

        omega_bottom = rng.uniform(OMEGA_RANGE[0], -0.05)
        omega_top = rng.uniform(-0.05, OMEGA_RANGE[1])
        omega = np.ascontiguousarray(
            np.clip(
                np.linspace(omega_bottom, omega_top, n_levels)
                + rng.normal(0.0, OMEGA_NOISE_STD, size=n_levels),
                *OMEGA_RANGE,
            )
        )
        ps = min(PLEV_RANGE[1], plev[0] + rng.uniform(0.0, 5_000.0))

        profiles.append(
            Profile(
                omega=omega,
                temp=temp,
                plev=plev,
                ps=ps,
            )
        )

    return profiles


def benchmark_python(profiles: list[Profile], rounds: int = N_ROUNDS) -> float:
    """Return the mean per-call runtime for the pure NumPy implementation."""
    durations = []
    for _ in range(rounds):
        start = time.perf_counter()
        for profile in profiles:
            scaling(profile.omega, profile.temp, profile.plev, profile.ps)
        durations.append(time.perf_counter() - start)
    return statistics_mean(durations) / len(profiles)


def benchmark_numba(
    profiles: list[Profile],
    rounds: int = N_ROUNDS,
) -> tuple[float, float]:
    """Return first-call and warm per-call runtimes for the Numba path."""
    first_profile = profiles[0]

    start = time.perf_counter()
    scaling_nb(
        first_profile.omega,
        first_profile.temp,
        first_profile.plev,
        first_profile.ps,
    )
    first_call = time.perf_counter() - start

    durations = []
    for _ in range(rounds):
        start = time.perf_counter()
        for profile in profiles:
            scaling_nb(profile.omega, profile.temp, profile.plev, profile.ps)
        durations.append(time.perf_counter() - start)
    warm_mean = statistics_mean(durations) / len(profiles)

    return first_call, warm_mean


def verify_outputs(profiles: list[Profile]) -> tuple[float, float]:
    """Return median and max absolute difference across sampled profiles."""
    abs_diffs = []

    for profile in profiles:
        python_value = scaling(profile.omega, profile.temp, profile.plev, profile.ps)
        numba_value = scaling_nb(profile.omega, profile.temp, profile.plev, profile.ps)

        abs_diffs.append(abs(python_value - numba_value))

    return float(np.median(abs_diffs)), max(abs_diffs)


def statistics_mean(values: list[float]) -> float:
    """Small local mean helper to keep dependencies minimal."""
    return sum(values) / len(values)


def format_duration(seconds: float) -> str:
    """Format a duration into a compact human-readable string."""
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.2f} us"
    return f"{seconds * 1e9:.2f} ns"


def format_ratio(numerator: float, denominator: float) -> str:
    """Format a simple speedup ratio."""
    if denominator == 0.0:
        return "inf"
    return f"{numerator / denominator:.2f}x"


def build_readme_block(
    python_time: float,
    numba_first_call: float,
    numba_warm: float,
    median_abs_diff: float,
    max_abs_diff: float,
) -> str:
    """Build the generated README block."""
    speedup = format_ratio(python_time, numba_warm)

    return "\n".join(
        [
            README_START,
            "_Automatically generated from seeded random profiles._",
            "",
            f"Benchmarked with `{N_SAMPLES}` random profiles, `{N_LEVELS}` pressure "
            f"levels per profile, `{N_ROUNDS}` timed rounds, and RNG seed `{SEED}`.",
            "",
            "| Implementation | Mean per call |",
            "|---|---:|",
            f"| `scaling` | {format_duration(python_time)} |",
            f"| `scaling_nb` first call | {format_duration(numba_first_call)} |",
            f"| `scaling_nb` warm | {format_duration(numba_warm)} |",
            "",
            f"Warm-call speedup: `{speedup}`",
            "",
            (
                "Agreement over sampled profiles: "
                f"median abs diff `{median_abs_diff:.3e}`, "
                f"max abs diff `{max_abs_diff:.3e}`"
            ),
            README_END,
        ]
    )


def update_readme(block: str, readme_path: Path = README_PATH) -> None:
    """Replace the generated benchmark section in README.md."""
    content = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(README_START)}.*?{re.escape(README_END)}",
        re.DOTALL,
    )

    if not pattern.search(content):
        raise RuntimeError("Benchmark markers were not found in README.md")

    updated = pattern.sub(block, content, count=1)
    trailing_newline = "\n" if not updated.endswith("\n") else ""
    readme_path.write_text(updated + trailing_newline, encoding="utf-8")


def main() -> int:
    """Run the benchmark update."""
    profiles = build_profiles()
    numba_first_call, numba_warm = benchmark_numba(profiles)
    python_time = benchmark_python(profiles)
    median_abs_diff, max_abs_diff = verify_outputs(profiles)

    block = build_readme_block(
        python_time=python_time,
        numba_first_call=numba_first_call,
        numba_warm=numba_warm,
        median_abs_diff=median_abs_diff,
        max_abs_diff=max_abs_diff,
    )
    update_readme(block)

    print("Updated README benchmark section.")
    print(f"Pure Python mean: {format_duration(python_time)}")
    print(f"Numba first call: {format_duration(numba_first_call)}")
    print(f"Numba warm mean: {format_duration(numba_warm)}")
    print(f"Warm speedup: {format_ratio(python_time, numba_warm)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
