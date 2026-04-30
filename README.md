# precip extremes scaling

Python implementation of Paul O'Gorman's [MATLAB code](http://www.mit.edu/~pog/src/precip_extremes_scaling.m), following equation 2 in [O'Gorman and Schneider, PNAS, 106, 14773-14777, 2009](http://www.pnas.org/content/106/35/14773.abstract).

The package takes:

- a vertical profile of vertical velocity
- a vertical profile of temperature
- a vertical profile of pressure levels
- a surface pressure

and returns precipitation in `kg/m^2/s`.

The highest pressure must be the first element of the input arrays.

## Installation

```bash
uv sync
```

For development dependencies:

```bash
uv sync --group dev
```

## Example

```python
import numpy as np
from precip_extremes_scaling import scaling

omega = np.linspace(-0.3, 0.2, 10)
temp = np.linspace(270, 220, 10)
plev = np.linspace(100000, 10000, 10)
ps = 100000

precip = scaling(omega, temp, plev, ps)  # ~6.1e-05 kg/m^2/s
```

## Xarray example

If your data are already in `xarray`, you can apply the compiled kernel across profile columns with `xr.apply_ufunc`:

```python
import numpy as np
import xarray as xr
import precip_extremes_scaling

# `ds` is an xarray Dataset with:
# - `w`: vertical velocity with a `level` dimension
# - `t`: temperature with a `level` dimension
# - `level`: pressure levels in hPa, ordered from highest pressure to lowest
ds = xr.open_dataset("profiles.nc")

scaling = xr.apply_ufunc(
    precip_extremes_scaling.scaling_nb,
    ds.w,
    ds.t,
    ds.level.values[:] * 100,
    ds.ps,
    input_core_dims=[["level"], ["level"], ["level"], []],
    output_core_dims=[[]],
    output_dtypes=[np.float64],
    vectorize=True,
    dask="parallelized",
)
```

For best performance with Dask:

- Use chunked inputs across horizontal or time dimensions, but keep the full `level` axis in each chunk because it is the core dimension passed to `scaling_nb`.
- Cache or persist reused inputs before calling `apply_ufunc`; as usual, chunked and cached inputs will be much quicker when using Dask.
- Expect the first call to `scaling_nb` to be much slower because Numba has to compile the kernel once. Reusing the same worker processes lets later calls run much faster.
- Keep inputs in a consistent floating-point dtype, ideally `float64`, to avoid extra casting work inside large Dask graphs.

## Benchmarks

<!-- benchmarks:start -->
_Automatically generated from seeded random profiles._

Benchmarked with `1000` random profiles, `10` pressure levels per profile, `5` timed rounds, and RNG seed `0`.

| Implementation | Mean per call |
|---|---:|
| `scaling` | 318.30 us |
| `scaling_nb` first call | 6.45 s |
| `scaling_nb` warm | 2.36 us |

Warm-call speedup: `134.85x`

Agreement over sampled profiles: median abs diff `1.944e-06`, max abs diff `4.288e-04`
<!-- benchmarks:end -->

## Development

Install git hooks:

```bash
uv run pre-commit install
```

Run checks locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/update_benchmarks.py
```

## Release process

This repo is configured for semantic releases from merges into `main`.

- Use Conventional Commits in PRs, such as `feat:`, `fix:`, or `docs:`.
- When a PR is merged into `main`, GitHub Actions runs semantic release.
- Semantic release updates the version, creates a GitHub release, and maintains `CHANGELOG.md`.
