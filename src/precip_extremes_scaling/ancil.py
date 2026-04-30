from precip_extremes_scaling import constants


def msl_to_ps(msl, t, z, lapse_rate=-0.0065):
    """
    Convert mean sea level pressure (msl) to pressure at height z assuming a
    uniform (constant) lapse rate.

    The barometric formula is applied for a linear temperature profile
    T(z) = T0 + L * z, where L is lapse_rate (K/m) and T0 is the provided
    temperature t (assumed at z = 0, i.e. mean sea level). Hydrostatic balance
    and the ideal gas law are assumed (constants taken from
    precip_extremes_scaling.constants).

    Parameters:
    msl (float): Mean sea level pressure in Pascals (Pa).
    t (float): Reference temperature T0 in Kelvin (K) at height z = 0.
    z (float): Height above mean sea level in meters^2 per second^2 (m^2/s^2).
    lapse_rate (float): Constant lapse rate L in K/m (negative if temperature
                        decreases with height, e.g. -0.0065 K/m).

    Returns:
    ps (float): Pressure at height z in Pascals (Pa).

    Notes:
    - The lapse rate is assumed uniform between z = 0 and the target height z.
    - Ensure T0 + L * z > 0 to avoid non-physical results.
    """
    # Calculate the pressure at height z using the barometric formula

    ps = msl * (1 + lapse_rate * z / (t * constants.GRAVITY - lapse_rate * z)) ** (
        -constants.GRAVITY / (lapse_rate * constants.GAS_CONSTANT_DRY_AIR)
    )

    return ps
