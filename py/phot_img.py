#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from astropy.io import fits
import glob
import aplpy

def main():
    fig = aplpy.FITSFigure('data/GTC/GRB160410A_GTC_med_r.fits')
    fig.show_grayscale()
    fig.show_colorscale()
    fig.add_colorbar()
    fig.colorbar.show()
    fig.colorbar.set_location('right')
    fig.colorbar.set_font(size='medium', weight='medium', \
                      stretch='normal', family='sans-serif', \
                      style='normal', variant='normal')
    fig.colorbar.set_axis_label_text('Flux (Jy/beam)')
    fig.colorbar.set_axis_label_font(size=12, weight='bold')
    fig.set_theme('publication')
    fig.save('figs/phot_img.pdf')


if __name__ == '__main__':
    main()
