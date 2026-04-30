"""Core precip extreme scaling calculations."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from precip_extremes_scaling import constants


def saturation_thermodynamics(
    temp: np.ndarray,
    plev: np.ndarray,
    calc_type: str = "simple",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | float]:
    """Return saturation vapor pressure, humidity, mixing ratio, and latent heat."""
    rd = constants.GAS_CONSTANT_DRY_AIR
    rv = constants.GAS_CONSTANT_WATER_VAPOR
    gc_ratio = rd / rv

    if calc_type == "era":
        es0 = constants.ECMWF_SATURATION_VAPOR_PRESSURE_REFERENCE
        t0 = constants.TRIPLE_POINT_TEMPERATURE
        ti = t0 - constants.ICE_TRANSITION_OFFSET

        esl = es0 * np.exp(
            constants.LIQUID_WATER_TETENS_A3
            * (temp - t0)
            / (temp - constants.LIQUID_WATER_TETENS_A4)
        )
        esi = es0 * np.exp(
            constants.ICE_TETENS_A3 * (temp - t0) / (temp - constants.ICE_TETENS_A4)
        )

        ls = constants.LATENT_HEAT_SUBLIMATION * np.ones(temp.size)

        lv0 = constants.LATENT_HEAT_VAPORIZATION_TRIPLE_POINT
        cpl = constants.SPECIFIC_HEAT_LIQUID_WATER
        cpv = constants.SPECIFIC_HEAT_WATER_VAPOR
        lv = lv0 - (cpl - cpv) * (temp - t0)

        iice = temp <= ti
        iliquid = temp >= t0
        imixed = (temp > ti) * (temp < t0)

        es = np.ones(temp.size) * np.nan
        latent_heat = np.ones(temp.size) * np.nan

        if any(iice):
            es[iice] = esi[iice]
            latent_heat[iice] = ls[iice]

        if any(iliquid):
            es[iliquid] = esl[iliquid]
            latent_heat[iliquid] = lv[iliquid]

        if any(imixed):
            a = ((temp[imixed] - ti) / (t0 - ti)) ** 2
            es[imixed] = (1 - a) * esi[imixed] + a * esl[imixed]
            latent_heat[imixed] = (1 - a) * ls[imixed] + a * lv[imixed]
    else:
        t0 = constants.TRIPLE_POINT_TEMPERATURE
        es0 = constants.SIMPLE_SATURATION_VAPOR_PRESSURE_REFERENCE
        latent_heat = constants.LATENT_HEAT_VAPORIZATION
        es = es0 * np.exp(latent_heat / rv * (1.0 / t0 - 1.0 / temp))

    rs = gc_ratio * es / (plev - es)
    qs = rs / (1 + rs)

    return es, qs, rs, latent_heat


def sat_deriv(
    plev: np.ndarray,
    temp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate derivatives of the saturation specific humidity."""
    dp = constants.PRESSURE_FINITE_DIFFERENCE_STEP
    dt = constants.TEMPERATURE_FINITE_DIFFERENCE_STEP

    es_p_plus, qs_p_plus, _, _ = saturation_thermodynamics(temp, plev + dp, "era")
    es_p_minus, qs_p_minus, _, _ = saturation_thermodynamics(temp, plev - dp, "era")
    es_t_plus, qs_t_plus, _, _ = saturation_thermodynamics(temp + dt, plev, "era")
    es_t_minus, qs_t_minus, _, _ = saturation_thermodynamics(temp - dt, plev, "era")

    dqsat_dp = (qs_p_plus - qs_p_minus) / (2.0 * dp)
    dqsat_dt = (qs_t_plus - qs_t_minus) / (2.0 * dt)
    dln_esat_dt = (np.log(es_t_plus) - np.log(es_t_minus)) / 2 / dt

    return dqsat_dp, dqsat_dt, dln_esat_dt


def moist_adiabatic_lapse_rate(
    temp: np.ndarray,
    plev: np.ndarray,
    calc_type: str,
) -> np.ndarray:
    """Return the saturated moist-adiabatic lapse rate in K/m."""
    g = constants.GRAVITY
    cpd = constants.SPECIFIC_HEAT_DRY_AIR
    cpv = constants.SPECIFIC_HEAT_WATER_VAPOR
    rd = constants.GAS_CONSTANT_DRY_AIR
    rv = constants.GAS_CONSTANT_WATER_VAPOR
    gc_ratio = rd / rv

    _, _, rs, latent_heat = saturation_thermodynamics(temp, plev, calc_type)

    if calc_type == "simple":
        lapse_rate = (
            g
            / cpd
            * (1 + latent_heat * rs / rd / temp)
            / (1 + latent_heat**2 * rs / (cpd * rv * temp**2))
        )
    else:
        lapse_rate = (
            g
            / cpd
            * (1 + rs)
            / (1 + cpv / cpd * rs)
            * (1 + latent_heat * rs / rd / temp)
            / (
                1
                + latent_heat**2
                * rs
                * (1 + rs / gc_ratio)
                / (rv * temp**2 * (cpd + rs * cpv))
            )
        )

    return lapse_rate


def integrate(f: np.ndarray, x: np.ndarray) -> float:
    """Approximate a one-dimensional integral."""
    dx1 = np.gradient(x)
    dx1[0] = 0.5 * dx1[0]
    dx1[-1] = 0.5 * dx1[-1]
    return np.sum(f * dx1)


def scaling(
    omega: np.ndarray,
    temp: np.ndarray,
    plev: np.ndarray,
    ps: float,
) -> float:
    """Estimate precipitation in kg/m^2/s from vertical profiles."""
    if plev[0] < plev[1]:
        raise ValueError("unexpected ordering of pressure levels")

    crit_lapse_rate = 0.002
    plev_mask = 0.05e5

    dqsat_dp, dqsat_dt, _ = sat_deriv(plev, temp)
    _, qsat, _, _ = saturation_thermodynamics(temp, plev, "era")
    lapse_rate = moist_adiabatic_lapse_rate(temp, plev, "era")

    temp_virtual = temp * (
        1.0
        + qsat
        * (constants.GAS_CONSTANT_WATER_VAPOR / constants.GAS_CONSTANT_DRY_AIR - 1.0)
    )
    rho = plev / constants.GAS_CONSTANT_DRY_AIR / temp_virtual
    dt_dp = lapse_rate / constants.GRAVITY / rho

    dqsat_dp_total = dqsat_dp + dqsat_dt * dt_dp

    dtemp_dp_env = np.gradient(temp, plev)
    lapse_rate_env = dtemp_dp_env * rho * constants.GRAVITY

    itrop = np.where(lapse_rate_env > crit_lapse_rate)[0]
    if itrop.size != 0 and np.max(itrop) + 1 < len(plev):
        dqsat_dp_total[np.max(itrop) + 1 :] = 0

    dqsat_dp_total[plev < plev_mask] = 0

    dqsat_dp_total_omega = dqsat_dp_total * omega
    dqsat_dp_total_omega[np.isnan(dqsat_dp_total_omega)] = 0

    kbot = plev > ps
    if any(kbot):
        dqsat_dp_total_omega[kbot] = 0

    return -integrate(-dqsat_dp_total_omega, plev) / constants.GRAVITY


@njit()
def _saturation_thermodynamics_point_nb(
    temp: float,
    plev: float,
) -> tuple[float, float, float, float]:
    """Return saturation thermodynamics for a single level."""
    rd = constants.GAS_CONSTANT_DRY_AIR
    rv = constants.GAS_CONSTANT_WATER_VAPOR
    gc_ratio = rd / rv

    es0 = constants.ECMWF_SATURATION_VAPOR_PRESSURE_REFERENCE
    t0 = constants.TRIPLE_POINT_TEMPERATURE
    ti = t0 - constants.ICE_TRANSITION_OFFSET

    esl = es0 * math.exp(
        constants.LIQUID_WATER_TETENS_A3
        * (temp - t0)
        / (temp - constants.LIQUID_WATER_TETENS_A4)
    )
    esi = es0 * math.exp(
        constants.ICE_TETENS_A3 * (temp - t0) / (temp - constants.ICE_TETENS_A4)
    )

    ls = constants.LATENT_HEAT_SUBLIMATION

    lv0 = constants.LATENT_HEAT_VAPORIZATION_TRIPLE_POINT
    cpl = constants.SPECIFIC_HEAT_LIQUID_WATER
    cpv = constants.SPECIFIC_HEAT_WATER_VAPOR
    lv = lv0 - (cpl - cpv) * (temp - t0)

    if temp <= ti:
        es = esi
        latent_heat = ls
    elif temp >= t0:
        es = esl
        latent_heat = lv
    else:
        a = ((temp - ti) / (t0 - ti)) ** 2
        es = (1.0 - a) * esi + a * esl
        latent_heat = (1.0 - a) * ls + a * lv

    rs = gc_ratio * es / (plev - es)
    qs = rs / (1.0 + rs)

    return es, qs, rs, latent_heat


@njit()
def saturation_thermodynamics_nb(
    temp: np.ndarray,
    plev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | float]:
    """Return saturation vapor pressure, humidity, mixing ratio, and latent heat."""
    nlev = temp.size
    es = np.empty(nlev, dtype=temp.dtype)
    qs = np.empty(nlev, dtype=temp.dtype)
    rs = np.empty(nlev, dtype=temp.dtype)
    latent_heat = np.empty(nlev, dtype=temp.dtype)

    for i in range(nlev):
        es_i, qs_i, rs_i, latent_heat_i = _saturation_thermodynamics_point_nb(
            temp[i], plev[i]
        )
        es[i] = es_i
        qs[i] = qs_i
        rs[i] = rs_i
        latent_heat[i] = latent_heat_i

    return es, qs, rs, latent_heat


@njit()
def sat_deriv_nb(
    plev: np.ndarray,
    temp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate derivatives of the saturation specific humidity."""
    dp = constants.PRESSURE_FINITE_DIFFERENCE_STEP
    dt = constants.TEMPERATURE_FINITE_DIFFERENCE_STEP
    inv_2dp = 1.0 / (2.0 * dp)
    inv_2dt = 1.0 / (2.0 * dt)

    nlev = temp.size
    dqsat_dp = np.empty(nlev, dtype=temp.dtype)
    dqsat_dt = np.empty(nlev, dtype=temp.dtype)
    dln_esat_dt = np.empty(nlev, dtype=temp.dtype)

    for i in range(nlev):
        es_p_plus, qs_p_plus, _, _ = _saturation_thermodynamics_point_nb(
            temp[i], plev[i] + dp
        )
        es_p_minus, qs_p_minus, _, _ = _saturation_thermodynamics_point_nb(
            temp[i], plev[i] - dp
        )
        es_t_plus, qs_t_plus, _, _ = _saturation_thermodynamics_point_nb(
            temp[i] + dt, plev[i]
        )
        es_t_minus, qs_t_minus, _, _ = _saturation_thermodynamics_point_nb(
            temp[i] - dt, plev[i]
        )

        dqsat_dp[i] = (qs_p_plus - qs_p_minus) * inv_2dp
        dqsat_dt[i] = (qs_t_plus - qs_t_minus) * inv_2dt
        dln_esat_dt[i] = (math.log(es_t_plus) - math.log(es_t_minus)) * inv_2dt

    return dqsat_dp, dqsat_dt, dln_esat_dt


@njit()
def moist_adiabatic_lapse_rate_nb(
    temp: np.ndarray,
    plev: np.ndarray,
) -> np.ndarray:
    """Return the saturated moist-adiabatic lapse rate in K/m."""
    g = constants.GRAVITY
    cpd = constants.SPECIFIC_HEAT_DRY_AIR
    cpv = constants.SPECIFIC_HEAT_WATER_VAPOR
    rd = constants.GAS_CONSTANT_DRY_AIR
    rv = constants.GAS_CONSTANT_WATER_VAPOR
    gc_ratio = rd / rv

    nlev = temp.size
    lapse_rate = np.empty(nlev, dtype=temp.dtype)

    for i in range(nlev):
        _, _, rs, latent_heat = _saturation_thermodynamics_point_nb(temp[i], plev[i])
        lapse_rate[i] = (
            g
            / cpd
            * (1.0 + rs)
            / (1.0 + cpv / cpd * rs)
            * (1.0 + latent_heat * rs / rd / temp[i])
            / (
                1.0
                + latent_heat**2
                * rs
                * (1.0 + rs / gc_ratio)
                / (rv * temp[i] ** 2 * (cpd + rs * cpv))
            )
        )

    return lapse_rate


@njit()
def gradient_fast(f, x):
    """Fast implementation of np.gradient for 1D arrays.

    Uses central differences for interior points and linear extrapolation
    at the boundaries. Intended for numba-compiled code; may not handle
    all edge cases.
    """
    out = np.empty_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / (x[2:] - x[:-2])
    out[0] = (f[1] - f[0]) / (x[1] - x[0])
    out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    return out


@njit()
def scaling_nb(
    omega: np.ndarray,
    temp: np.ndarray,
    plev: np.ndarray,
    ps: float,
) -> float:
    """Estimate precipitation in kg/m^2/s from vertical profiles.

    Assumes pressure levels are consistent across variables."""
    if plev[0] < plev[1]:
        plev_idx = np.argsort(plev)[::-1]  # sort in descending order
        plev = plev[plev_idx]
        temp = temp[plev_idx]
        omega = omega[plev_idx]

    crit_lapse_rate = 0.002
    plev_mask = 0.05e5

    dqsat_dp, dqsat_dt, _ = sat_deriv_nb(plev, temp)
    g = constants.GRAVITY
    cpd = constants.SPECIFIC_HEAT_DRY_AIR
    cpv = constants.SPECIFIC_HEAT_WATER_VAPOR
    rd = constants.GAS_CONSTANT_DRY_AIR
    rv = constants.GAS_CONSTANT_WATER_VAPOR
    gc_ratio = rd / rv
    rv_over_rd_minus_one = rv / rd - 1.0

    nlev = temp.size
    rho = np.empty(nlev, dtype=temp.dtype)
    dqsat_dp_total = np.empty(nlev, dtype=temp.dtype)

    for i in range(nlev):
        _, qsat, rs, latent_heat = _saturation_thermodynamics_point_nb(temp[i], plev[i])
        temp_virtual = temp[i] * (1.0 + qsat * rv_over_rd_minus_one)
        rho_i = plev[i] / rd / temp_virtual
        rho[i] = rho_i

        lapse_rate_i = (
            g
            / cpd
            * (1.0 + rs)
            / (1.0 + cpv / cpd * rs)
            * (1.0 + latent_heat * rs / rd / temp[i])
            / (
                1.0
                + latent_heat**2
                * rs
                * (1.0 + rs / gc_ratio)
                / (rv * temp[i] ** 2 * (cpd + rs * cpv))
            )
        )
        dqsat_dp_total[i] = dqsat_dp[i] + dqsat_dt[i] * (lapse_rate_i / g / rho_i)

    last_trop_idx = -1
    for i in range(nlev):
        if i == 0:
            dtemp_dp_env = (temp[1] - temp[0]) / (plev[1] - plev[0])
        elif i == nlev - 1:
            dtemp_dp_env = (temp[-1] - temp[-2]) / (plev[-1] - plev[-2])
        else:
            dtemp_dp_env = (temp[i + 1] - temp[i - 1]) / (plev[i + 1] - plev[i - 1])

        if dtemp_dp_env * rho[i] * g > crit_lapse_rate:
            last_trop_idx = i

    if last_trop_idx != -1 and last_trop_idx + 1 < nlev:
        for i in range(last_trop_idx + 1, nlev):
            dqsat_dp_total[i] = 0.0

    dqsat_dp_total_omega = np.empty(nlev, dtype=temp.dtype)
    for i in range(nlev):
        value = dqsat_dp_total[i]
        if plev[i] < plev_mask or plev[i] > ps or omega[i] >= 0.0 or np.isnan(value):
            dqsat_dp_total_omega[i] = 0.0
        else:
            dqsat_dp_total_omega[i] = value * omega[i]

    return -np.trapezoid(-dqsat_dp_total_omega, plev) / g
