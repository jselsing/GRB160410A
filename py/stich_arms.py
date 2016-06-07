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

def stitch_spectra(list_of_paths):
    concatenated_wl, concatenated_spec, concatenated_specerr, concatenated_bpmap, concatenated_normerr  = [], [], [], [], []
    norm_start, norm_end, arm = [], [], []
    for ii in list_of_paths:
        dataframe = np.genfromtxt(ii)
        wl = dataframe[:, 0]
        if "UVB" in ii:
            mask = (wl >= 3050) & (wl <= 5600)
            arm.append("UVB")
        elif "VIS" in ii:
            mask = (wl >= 5600) & (wl <= 10100)
            arm.append("VIS")
        elif "NIR" in ii:
            mask = (wl >= 10100) & (wl <= 24000)
            arm.append("NIR")
        wl = wl[mask]

        flux = dataframe[:, 1][mask] #/ trans
        fluxerror = dataframe[:, 2][mask] #/ trans
        bpmap = np.zeros_like(flux)

        concatenated_wl.append(wl)
        concatenated_spec.append(flux)
        concatenated_specerr.append(fluxerror)
        concatenated_bpmap.append(bpmap)

        from numpy.polynomial import chebyshev
        mask = np.isnan(flux)
        edge_len = 3000
        if "NIR" in ii:
            edge_len *= 2.0/5.0

        chebfit = chebyshev.chebfit(wl[~mask][:int(edge_len)], flux[~mask][:int(edge_len)], deg = 1, w=1/fluxerror[~mask][:int(edge_len)])
        chebfitval = chebyshev.chebval(wl[:int(edge_len)], chebfit)
        norm_start.append(chebfitval[0])


        chebfit = chebyshev.chebfit(wl[~mask][int(-edge_len):], flux[~mask][int(-edge_len):], deg = 1, w=1/fluxerror[~mask][int(-edge_len):])
        chebfitval = chebyshev.chebval(wl[int(-edge_len):], chebfit)
        norm_end.append(chebfitval[-1])
    for ii, kk in enumerate(arm):
        if kk == "VIS":
            start = norm_start[ii]
            end = norm_end[ii]

    for ii, kk in enumerate(arm):
        if kk == "UVB":
            concatenated_spec[ii] *= start/norm_end[ii]
        if kk == "NIR":
            concatenated_spec[ii] *= end/norm_start[ii]

    data_array = np.array([np.hstack(np.array(concatenated_wl)), np.hstack(np.array(concatenated_spec)), np.hstack(np.array(concatenated_specerr)), np.hstack(np.array(concatenated_bpmap))])
    data_array = data_array[:,np.argsort(data_array[0, :])]

    return data_array


def main():
    arms = glob.glob("data/*optext.dat")
    data_array = stitch_spectra(arms)
    data_array[2, :][data_array[3, :].astype("bool")] *= 10
    hbin = 10
    bin_wl, bin_flux, bin_error = bin_spectrum(data_array[0, :], data_array[1, :], data_array[2, :], hbin)
    mask = bin_wl > 3200
    bin_wl, bin_flux, bin_error = bin_wl[mask], bin_flux[mask], bin_error[mask]
    dt = [("wl", np.float64), ("flux", np.float64), ("error", np.float64) ]
    data = np.array(zip(bin_wl, bin_flux, bin_error), dtype=dt)
    np.savetxt("data/GRB160410A_bin"+str(hbin)+".dat", data, header="wavelength flux error", fmt = ['%1.5e', '%1.5e', '%1.5e'] )

if __name__ == '__main__':
    main()