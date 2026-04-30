import numpy as np


def msl_to_ps(msl, t, z):
    """
    Convert mean sea level pressure (msl) to surface pressure (ps) at a
    given temperature (t) and geopotential height (z).

    Assumes a standard atmosphere and uses the barometric formula to calculate the
    pressure at height z based on the mean sea level pressure, temperature,
    and geopotential height.

    Parameters:
    msl (float): Mean sea level pressure in Pascals (Pa).
    t (float): Temperature in Kelvin (K).
    z (float): Geopotential height in m2/s2.
    """
    # Constants
    R = 287.05  # Specific gas constant for dry air (J/(kg*K))

    # Calculate the pressure at height z using the barometric formula
    ps = msl * np.exp(-z / (R * t))

    return ps
