import numpy as np
import pytest

from precip_extremes_scaling import scaling


def test_scaling_matches_readme_example() -> None:
    omega = np.linspace(-0.3, 0.2, 10)
    temp = np.linspace(270, 220, 10)
    plev = np.linspace(100000, 10000, 10)
    ps = 100000

    precip = scaling(omega, temp, plev, ps)

    assert precip == pytest.approx(6.1e-05, rel=1e-2)
