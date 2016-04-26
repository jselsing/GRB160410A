#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Importing manupulation packages

import numpy as np
import glob

# Plotting
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')
from util import *
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.special import wofz
import lmfit

# Constants
c = 2.99792458e5 # km/s
m_e = 9.10938291e-28 # g
hbar = 1.054571726e-27 # erg * s
alpha = 1 / 137.035999139 # dimensionless
K = np.pi * alpha * hbar / m_e # cm^2 / s
sigfwhm = 2*np.sqrt(2*np.log(2)) # dimensionless


def voigt(x, sigma=1, gamma=0):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    z = (x + 1j*gamma)/ (sigma*np.sqrt(2.0))
    return wofz(z).real / (sigma*np.sqrt(2*np.pi))


def voigt_tau(x, N, osc, l0, sigma, gamma):
    """ Returns the optical depth (Voigt profile) for a transition.

    Given an transition with rest wavelength wa0, osc strength,
    natural linewidth gam; b parameter (doppler and turbulent); and
    log10 (column density), returns the optical depth in velocity
    space. v is an array of velocity values in km/s. The absorption
    line must be centred at v=0.

    Parameters
    ----------
    x : np.array
      array of velocities centered at the profile in km/s
    N : float
      Column density in cm^-2
    osc : float
      Oscillator strength of transition (dimensionless).
    l0 : float
      Rest wavelength of transition in Angstroms.
    sigma : float
      Sigma parameter for the profile (km/s).
    gamma : float
      Gamma parameter for the transition (km/s).

    Returns
    -------
    tau : The optical depth

    """
    l0 = l0 * 1e-13 # Convert Å to km
    gamma = gamma * l0 / (2 * np.pi) # Convert decay rate to velocity width km/s
    return K * osc * l0 * voigt(x, sigma, gamma) * N


def voigt_abs(pars, t):
    """
    f_obs = f_source * e ^-tau
    """
    N, sigma, z, osc, l0, gamma, resolution_fwhm = pars
    # Move wavelength array to rest
    t = t / (1 + z)
    # Center wavlength array at line
    t_rest = t - l0
    # Convert to km/s
    t = c * t_rest / t
    # Get optical depth
    tau = voigt_tau(t, N, osc, l0, sigma, gamma)
    # Absorb flux
    flux = np.exp(-tau)
    # Get transformation from km/s to pixels for convolution
    mask = (t > - 2000) & (t < 2000)
    transform = np.median(np.diff(t)[mask[:-1]])
    # Convolve with linespread-function
    if resolution_fwhm > 0.:
        flux = convolve(flux, Gaussian1DKernel(resolution_fwhm/(sigfwhm*transform)))
    return flux


def dla_abs(x, a, k, N, sigma, z):
    func = (a*x**k)*voigt_abs(x, N, sigma, z, 0.416, 1216, 6.265e8, 0)
    return func
########


def power_law(pars, t):
    a, k = pars
    return a * t ** k


def main():
    data = np.genfromtxt("data/GRB160410A_bin5.dat")
    wl, flux, error = data[:, 0], data[:, 1], data[:, 2]
    mask = ~(np.isnan(flux) | np.isinf(flux) | np.isnan(error) | np.isinf(error) | np.isnan(1/error**2) | np.isinf(1/error**2))
    wl, flux, error = wl[mask], flux[mask], error[mask]
    pl.errorbar(wl, flux, yerr=error, fmt=".k", capsize=0, elinewidth=0.5, ms=3)
    pl.plot(wl, flux, lw = 1, linestyle="steps-mid", alpha=0.7)


    data = np.genfromtxt("data/GRB160410A_bin1.dat")
    wl, flux, error = data[:, 0], data[:, 1], data[:, 2]
    mask = ~(np.isnan(flux) | np.isinf(flux) | np.isnan(error) | np.isinf(error) | np.isnan(1/error**2) | np.isinf(1/error**2))
    wl, flux, error = wl[mask], flux[mask], error[mask]
    p = lmfit.Parameters()
    p.add_many(('amp_pow',        1.8e-8, True),
               ('slope_pow',       -2.38, True),
               ('N',                6e20, True),
               ('sigma',             200, True),
               ('z',               1.715, True),
               ('f',               0.416, False),
               ('lambda_zero', 1215.6701, False),
               ('gamma',         6.265e8, False),
               ('resolution_fwhm',     0, False))

    def residual(pars):
        v = p.valuesdict()
        resid = power_law([v["amp_pow"], v["slope_pow"]], wl) * voigt_abs([v["N"], v["sigma"], v["z"], v["f"], v["lambda_zero"], v["gamma"], v["resolution_fwhm"]], wl) - flux
        return resid**2 / error**2
    mi = lmfit.minimize(residual, p, method='Nelder')
    lmfit.printfuncs.report_fit(mi.params)
    pl.plot(wl, residual(mi.params) + flux)

    # from scipy import optimize
    # popt, pcov = optimize.curve_fit(dla_abs, wl, flux, sigma=error, p0 = p0, maxfev = 5000)
    # print(popt)
    # pl.plot(wl, dla_abs(wl, *p0), lw = 3, linestyle="steps-mid", alpha=0.7)
    # pl.plot(wl, dla_abs(wl, *popt), lw = 3, linestyle="steps-mid", alpha=0.7)
    pl.xlim(3200, 3400)
    pl.ylim(-1e-18, 1e-16)
    pl.show()


if __name__ == '__main__':
    main()