#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Importing manupulation packages

import numpy as np
import glob

# Plotting
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')
cmap = seaborn.color_palette("muted")
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
    return K * osc * l0 * voigt(x, sigma, gamma) * 10**(N)


def voigt_abs(pars, t):
    """
    f_obs = f_source * e ^-tau
    """
    N, sigma, z, osc, l0, gamma, resolution_fwhm = pars
    # Move wavelength array to rest
    t = t.copy() / (1 + z)
    # Center wavlength array at line
    t_rest = t - l0
    # Convert to km/s
    t = c * t_rest / t
    # Get optical depth
    tau = voigt_tau(t, N, osc, l0, sigma, gamma)
    # Absorb flux
    flux = np.exp(-tau)
    # Correct for instrumental resolution
    if resolution_fwhm > 0.:
        # Get transformation from km/s to pixels for convolution
        mask = (t > - 5000) & (t < 5000)
        transform = np.median(np.diff(t)[mask[:-1]])
        if np.isnan(transform):
            transform = 1e3
        # print(transform)
        # Convolve with linespread-function
        flux = convolve(flux, Gaussian1DKernel(stddev=resolution_fwhm/(sigfwhm*transform), mode="oversample"))
    return flux


def power_law(pars, t):
    a, k = pars
    return 10**(a) * t ** k


def extinction_curve(lambda_in):
    lambda_temp = lambda_in.copy() / 10000
    # print(lambda_temp)
    k = np.zeros_like(lambda_in)
    k[np.where(lambda_temp < 0.6)] = -5.726 + 4.004/lambda_temp - 0.525/lambda_temp**2 + 0.029/lambda_temp**3  + 2.505
    k[np.where(0.6 >= lambda_temp)] = -2.672 - 0.010/lambda_temp + 1.532/(lambda_temp**2) - 0.412/(lambda_temp**3) + 2.505
    # if min(lambda_temp) < 0.15 or max(lambda_temp) > 2.2:
    #     print('Extinction curve only valid between 0.15 and 2.2 micron')
    return k


def extinction_absorption(pars, t):
    z, ebv = pars
    t = t / (1 + z)
    return 10**(0.4 * extinction_curve(t)*ebv)


def residual(pars, t, data=None, error=None):
    """
    Objective function which calculates the residuals. Using for minimizing.
    """

    # Unpack parameter values
    if type(pars) is list or isinstance(pars, np.ndarray):
        amp_pow, slope_pow = pars[0], pars[1]
        N, sigma, f, l0, gamma, res_fwhm  = pars[2], pars[3], pars[6], pars[7], pars[8], pars[9]
        z = pars[4]
        ebv = pars[5]
    else:
        v = pars.valuesdict()
        amp_pow, slope_pow = v["amp_pow"], v["slope_pow"]
        N, sigma, f, l0, gamma, res_fwhm = v["N"], v["sigma"], v["f"], v["lambda_zero"], v["gamma"], v["resolution_fwhm"]
        z = v["z"]
        ebv = v["ebv"]
    # Construct model components
    voigt_component = voigt_abs([N, sigma, z, f, l0, gamma, res_fwhm], t)
    power_law_component = power_law([amp_pow, slope_pow], t)
    extinction_component = extinction_absorption([z, ebv], t)

    # Make model
    model = power_law_component * voigt_component * extinction_component

    if data is None:
        return model
    if error is None:
        return (model - data)
    return (model - data)/error


def main():
    data = np.genfromtxt("data/stitched_spectrum_bin10.dat")
    wl, flux, error = data[:, 0], data[:, 1], data[:, 2]
    mask = ~(np.isnan(flux) | np.isinf(flux) | np.isnan(error) | np.isinf(error) | np.isnan(1/error**2) | np.isinf(1/error**2))
    error[error < 1e-25] = 1e-17
    wl, flux, error = wl[mask], flux[mask], error[mask]
    pl.errorbar(wl, flux, yerr=error, fmt=".k", capsize=0, elinewidth=0.5, ms=3, zorder=1)
    pl.plot(wl, flux, lw = 1, linestyle="steps-mid", alpha=0.7)
    # pl.ylim((-1e-18, 1e-16))
    # pl.show()
    # exit()

    data = np.genfromtxt("data/stitched_spectrum_bin10.dat")
    wl, flux, error = data[:, 0], data[:, 1], data[:, 2]
    mask = ~(np.isnan(flux) | np.isinf(flux) | np.isnan(error) | np.isinf(error) | np.isnan(1/error**2) | np.isinf(1/error**2))
    error[error < 1e-25] = 1e-17
    wl, flux, error = wl[mask], flux[mask], error[mask]
    p = lmfit.Parameters()
    #           (Name,  Value,  Vary,   Min,  Max,  Expr)
    p.add_many(('amp_pow',            -8, True, -np.inf, 0),
               ('slope_pow',       -2.38, True, -3, -2),
               ('N',                  20, True, 18, 22),
               ('sigma',             200, True, 0),
               ('z',               1.716, True, 1.70, 1.73),
               ('ebv',               0.0, True, 0),
               ('f',               0.416, False),
               ('lambda_zero', 1215.6701, False),
               ('gamma',         6.265e8, False),
               ('resolution_fwhm',     30, False))

    mi = lmfit.minimize(residual, p, method='Nelder', args=(wl, flux, error))
    print(lmfit.report_fit(mi.params))
    # pl.plot(wl, residual(mi.params.valuesdict().values(), wl), lw = 3, color=cmap[2], zorder=2)
    # pl.ylim((-1e-18, 1e-16))
    # pl.show()
    # exit()

    def lnprob(pars):
        """
        This is the log-likelihood probability for the sampling.
        """
        model = residual(pars, wl)
        return -0.5 * np.sum(((model - flux) / error)**2 + np.log(2 * np.pi * error**2))

    mini = lmfit.Minimizer(lnprob, mi.params)

    nwalkers = 100
    v = mi.params.valuesdict()
    MLE_vals = np.array([v["amp_pow"], v["slope_pow"], v["N"], v["sigma"], v["z"], v["ebv"]])
    # pos = [MLE_vals + 1e-2*MLE_vals*np.random.randn(len(MLE_vals)) for i in range(nwalkers)]
    res = mini.emcee(nwalkers=nwalkers, burn=100, steps=1000, thin=1, params=mi.params, seed=12345)


    low, mid, high, names = [], [], [], []
    for kk in res.params.valuesdict().keys():
        names.append(str(kk))
        if str(kk) in ["f", "lambda_zero", "gamma", "resolution_fwhm"]:
            low.append(res.params.valuesdict()[str(kk)])
            mid.append(res.params.valuesdict()[str(kk)])
            high.append(res.params.valuesdict()[str(kk)])
        else:
            low.append(np.percentile(res.flatchain[str(kk)], [15.9])[0])
            mid.append(np.percentile(res.flatchain[str(kk)], [50])[0])
            high.append(np.percentile(res.flatchain[str(kk)], [84.2])[0])
    low, mid, high = np.array(low), np.array(mid), np.array(high)


    pl.plot(wl, residual(res.params.valuesdict().values(), wl), lw = 1, color=cmap[2], zorder=3)
    pl.fill_between(wl, residual(low, wl), residual(high, wl), alpha=0.5, color=cmap[2], zorder=2)
    # pl.plot(wl, residual(high, wl), lw = 3, linestyle="dashed")
    pl.xlim(3200, 6000)
    pl.ylim(-1e-18, 1.5e-16)
    pl.xlabel(r"Wavelength / [$\mathrm{\AA}$]")
    pl.ylabel(r'Flux density [erg s$^{-1}$ cm$^{-1}$ $\AA^{-1}$]')
    pl.savefig("figs/DLA_fit_zoom.pdf")
    pl.xlim(3200, 10000)
    pl.ylim(-1e-18, 1.5e-16)
    pl.savefig("figs/DLA_fit.pdf")
    pl.clf()
    # exit()
    from matplotlib.ticker import MaxNLocator
    fig, axes = pl.subplots(6, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(res.chain[:, :, 0].T, color="k", alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].axhline(mid[0], color="#888888", lw=1)
    # axes[0].set_ylim((mid[0] - 3 * low[0], mid[0] + 3 * high[0]))
    axes[0].set_ylabel("Powerlaw amplitude")

    axes[1].plot(res.chain[:, :, 1].T, color="k", alpha=0.4)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].axhline(mid[1], color="#888888", lw=1)
    # axes[1].set_ylim((mid[1] - 3 * low[1], mid[1] + 3 * high[1]))
    axes[1].set_ylabel("Powerlaw slope")

    axes[2].plot(res.chain[:, :, 2].T, color="k", alpha=0.4)
    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    axes[2].axhline(mid[2], color="#888888", lw=1)
    # axes[2].set_ylim((mid[2] - 3 * low[2], mid[2] + 3 * high[2]))
    axes[2].set_ylabel("N")

    axes[3].plot(res.chain[:, :, 3].T, color="k", alpha=0.4)
    axes[3].yaxis.set_major_locator(MaxNLocator(5))
    axes[3].axhline(mid[3], color="#888888", lw=1)
    # axes[3].set_ylim((mid[3] - 3 * low[3], mid[3] + 3 * high[3]))
    axes[3].set_ylabel("$\sigma$")

    axes[4].plot(res.chain[:, :, 4].T, color="k", alpha=0.4)
    axes[4].yaxis.set_major_locator(MaxNLocator(5))
    axes[4].axhline(mid[4], color="#888888", lw=1)
    # axes[4].set_ylim((mid[4] - 3 * low[4], mid[4] + 3 * high[4]))
    axes[4].set_ylabel("z")

    axes[5].plot(res.chain[:, :, 5].T, color="k", alpha=0.4)
    axes[5].yaxis.set_major_locator(MaxNLocator(5))
    axes[5].axhline(mid[5], color="#888888", lw=1)
    # axe5[4].set_ylim((mid[4] - 3 * low[4], mid[4] + 3 * high[4]))
    axes[5].set_ylabel("ebv")
    axes[5].set_xlabel("step number")

    fig.tight_layout(h_pad=0.0)
    fig.savefig("figs/line-time.pdf")

    dt = [("names", np.str, 16), ("value", np.float64), ("+error", np.float64), ("-error", np.float64)]
    data = np.array(zip(np.array(names), mid, abs(mid - low), abs(high-mid)), dtype=dt)
    np.savetxt("data/dla_fitresults.dat", data, header="names value +error -error", fmt = ['%16s', '%1.5e', '%1.5e', '%1.5e'])


    import corner
    corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
    pl.savefig("figs/Cornerplot.pdf", clobber=True)





if __name__ == '__main__':
    main()