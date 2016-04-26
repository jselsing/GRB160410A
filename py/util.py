#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Importing manupulation packages

import numpy as np
import glob

__all__ = ["weighted_avg", "bin_spectrum", "gaussian", "voigt"]


def gaussian(x, amp, cen, sigma):
    # Simple Gaussian
    return amp * np.exp(-(x - cen)**2 / sigma**2)


def voigt(x, amp=1, cen=0, sigma=1, gamma=0):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    from scipy.special import wofz
    z = (x-cen + 1j*gamma)/ (sigma*np.sqrt(2.0))
    return amp * wofz(z).real / (sigma*np.sqrt(2*np.pi))


def weighted_avg(flux, error, axis=2):

    """Calculate the weighted average with errors
    ----------
    flux : masked array-like
        Values to take average of
    error : masked array-like
        Errors associated with values, assumed to be standard deviations.
    mask : masked array-like
        Errors associated with values, assumed to be standard deviations.
    axis : int, default 0
        axis argument passed to numpy.ma.average

    Returns
    -------
    average, error : tuple

    Notes
    -----
    Functionality similar to np.ma.average, only also returns the associated error
    """

    # Normalize to avoid numerical issues in flux-calibrated data
    norm = abs(np.ma.mean(flux))
    flux_func = flux.copy() / norm
    error_func = error.copy() / norm

    weight = 1.0 / (error_func ** 2.0)

    average, sow = np.ma.average(flux_func, weights = weight, axis = axis, returned = True)
    variance = 1.0 / sow

    return (average * norm, np.sqrt(variance)*norm)


def bin_spectrum(wl, flux, error, binh):

    """Bin low S/N 1D data from xshooter
    ----------
    flux : np.array containing 2D-image flux
        Flux in input image
    error : np.array containing 2D-image error
        Error in input image
    binh : int
        binning along x-axis

    Returns
    -------
    binned fits image
    """

    print("Binning image by a factor: "+str(binh))
    if binh == 1:
        return wl, flux, error

    # Outsize
    size = flux.shape[0]
    outsize = size/binh

    # Containers
    wl_out = np.ma.zeros((outsize))
    res = np.ma.zeros((outsize))
    reserr = np.ma.zeros((outsize))

    for ii in np.arange(0, size - binh, binh):
        # Find psotions in new array
        h_slice = slice(ii, ii + binh)
        h_index = (ii + binh)/binh - 1
        # print(h_index)
        # Construct weighted average and weighted std along binning axis
        res[h_index], reserr[h_index] = weighted_avg(flux[ii:ii + binh], error[ii:ii + binh], axis=0)
        wl_out[h_index] = np.median(wl[ii:ii + binh], axis=0)

    return wl_out, res, reserr