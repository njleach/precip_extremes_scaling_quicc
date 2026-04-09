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
```

## Release process

This repo is configured for semantic releases from merges into `main`.

- Use Conventional Commits in PRs, such as `feat:`, `fix:`, or `docs:`.
- When a PR is merged into `main`, GitHub Actions runs semantic release.
- Semantic release updates the version, creates a GitHub release, and maintains `CHANGELOG.md`.
