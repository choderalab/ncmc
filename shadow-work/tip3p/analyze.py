#!/bin/env python

from progressbar import ProgressBar
import numpy as np
import sys
from simtk import unit, openmm
from simtk.openmm import app
import netCDF4 as netcdf

from simulate import nworksteps, timestep

def analyze_ncfile(filename):
    import os
    name = os.path.splitext(filename)[0]

    # Read work
    ncfile = netcdf.Dataset(filename, 'r')
    work = ncfile.variables['work'][:,:]
    [nwork,nworkvals] = work.shape

    if nwork == 0:
        print('No work trajectories written yet.')
        sys.exit(0)

    # Plot results
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    filename = name + '.pdf'
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(filename) as pdf:
        fig = plt.figure(figsize=(12,10))
        t = np.arange(nworkvals) * nworksteps * timestep / unit.picoseconds

        plt.subplot(3,1,1)
        plt.plot(t, work.T, '-')
        plt.hold(True)
        plt.plot(t, work.mean(0), '-', linewidth=3)
        #plt.xlabel('trajectory length (ps)')
        plt.ylabel('work (kT)')

        plt.subplot(3,1,2)
        plt.plot(t, work.std(0), '-')
        plt.xlabel('trajectory length (ps)')
        plt.ylabel('work stddev (kT)')

        plt.subplot(3,1,3)
        Paccept = np.minimum(np.ones(work.shape), np.exp(-work)).mean(0)
        dPaccept = np.exp(-work).std(0) / np.sqrt(nwork)
        plt.gca().fill_between(t, Paccept-2*dPaccept, Paccept+2*dPaccept, facecolor='grey', interpolate=True, alpha=0.5)
        plt.axis([0, t.max(), 0, 1])
        plt.plot(t, Paccept, '-')
        plt.xlabel('trajectory length (ps)')
        plt.ylabel('acceptance probability')

        pdf.savefig()
        plt.close()

if __name__ == '__main__':
    import glob
    filenames = glob.glob('*.nc')
    for filename in filenames:
        print(filename)
        try:
            analyze_ncfile(filename)
        except Exception as e:
            print(e)
