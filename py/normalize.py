
"""
Collect spectra after normalization
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')
import glob

def normalize_spectra():
    import xsh_norm.interactive
    datafile = np.genfromtxt("data/NIROB1skysuboptext.dat")

    answer = 'y'
    while answer == 'y':
        # dat = np.load(nn)
        wl_arr = datafile[:, 1]
        try:
            flux_arr = datafile[:, 2]*datafile[:, 8]
            error_arr = datafile[:, 3]*datafile[:, 8]
        except:
            flux_arr = datafile[:, 2]
            error_arr = datafile[:, 3]
        bp_arr = datafile[:, 4]

        normalise = xsh_norm.interactive.xsh_norm(wl_arr, flux_arr, error_arr, bp_arr, wl_arr, flux_arr, error_arr, "normalized")
        # change = raw_input('Change default maskparameters(y/n)? ')
        # if change == 'y':

        normalise.leg_order = 3 #float(raw_input('Filtering Chebyshev Order (default = 3) = '))
        normalise.endpoint_order = 2#float(raw_input('Order of polynomial used for placing endpoints (default = 3) = '))
        # normalise.exclude_width = 1#float(raw_input('Exclude_width (default = 5) = '))
        # normalise.sigma_mask = 20#float(raw_input('Sigma_mask (default = 5) = '))
        # normalise.lover_mask = #float(raw_input('Lover bound mask (default = -1e-17) = '))
        # normalise.tolerance = #float(raw_input('Filtering tolerance (default = 0.25) = '))
        # normalise.leg_order = #float(raw_input('Filtering Chebyshev Order (default = 3) = '))
        normalise.spacing = 400#float(raw_input('Spacing in pixels between spline points (default = 150) = '))
        # normalise.division = #float(raw_input('Maximum number of allowed points (default = 300) = '))
        # normalise.endpoint_order = #float(raw_input('Order of polynomial used for placing endpoints (default = 3) = '))
        normalise.endpoint = "t"#str(raw_input('Insert endpoint before interpolation(y/n)? '))
        normalise.run()
        # pl.title(names[ii])
        pl.xlabel(r'Observed Wavelength  [$\mathrm{\AA}$]')
        pl.ylabel(r'FLux [erg/cm$^2$/s/$\mathrm{\AA}$]')
        # print(np.median(flux_arr[~np.isnan(flux_arr)]))
        # sn = np.median(flux_arr[~np.isnan(flux_arr)]) / np.median(error_arr[~np.isnan(error_arr)]) 
        pl.ylim(-1e-16, 4e-15)
        pl.show()
        answer = "n"
        # answer = raw_input('Re-run normalisation(y/n)? ')
#            normalise.clear()


if __name__ == '__main__':

    if True:
        normalize_spectra()
