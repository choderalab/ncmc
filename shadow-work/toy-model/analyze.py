#!/usr/bin/env python

################################################################################
# IMPORTS
################################################################################

import numpy as np
import netCDF4 as netcdf
import itertools

################################################################################
# MAIN
################################################################################

def analyze():
    # Open NetCDF file
    filename = 'toy-work.nc'
    ncfile = netcdf.Dataset(filename, 'r')
    # Load in all parameter combinations
    data = dict()
    timesteps_to_try = set()
    nsteps_to_try = set()
    equilibration_integrators_to_try = ['VVVR', 'GHMC']
    switching_integrators_to_try = ['VV', 'VVVR', 'GHMC']
    for ncgrp in ncfile.groups.values():
        timestep = getattr(ncgrp, 'timestep')
        nsteps = getattr(ncgrp, 'nsteps')

        timesteps_to_try.add(timestep)
        nsteps_to_try.add(nsteps)

        equilibration_integrator = getattr(ncgrp, 'equilibration_integrator')
        switching_integrator = getattr(ncgrp, 'switching_integrator')
        print(equilibration_integrator,switching_integrator,nsteps,timestep)
        for direction in ['forward', 'backward']:
            for workname in ['protocol_work', 'shadow_work', 'total_work']:
                data[(equilibration_integrator,switching_integrator,nsteps,timestep,direction,workname)] = ncgrp.groups[direction].variables[workname][:]

    timesteps_to_try = np.sort(list(timesteps_to_try))
    nsteps_to_try = np.sort(list(nsteps_to_try))

    # Generate plots
    # Plot results
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    # Compare protocol and shadow work to see if they are correlated.
    print("Comparing protocol and shadow work...")
    filename = 'protocol-vs-shadow-work.pdf'
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(filename) as pdf:
        for equilibration_integrator in equilibration_integrators_to_try:
            for switching_integrator in switching_integrators_to_try:
                print((equilibration_integrator,switching_integrator))
                fig = plt.figure(figsize=(14,12))

                fig.text(0.5, 0.95, 'equilibration with %s, switching with %s' % (equilibration_integrator,switching_integrator), ha='center')
                fig.text(0.5, 0.02, 'number of switching steps', ha='center')
                fig.text(0.02, 0.5, 'timestep $\Delta t$', va='center', rotation='vertical')

                fig.text(0.13, 0.06, 'protocol work')
                fig.text(0.10, 0.18, 'shadow work', rotation='vertical')

                ny = len(timesteps_to_try)
                nx = len(nsteps_to_try)
                for (y, timestep) in enumerate(timesteps_to_try):
                    for (x, nsteps) in enumerate(nsteps_to_try):
                        plt.subplot(ny, nx, y*nx + x + 1)

                        protocol_work = data[(equilibration_integrator,switching_integrator,nsteps,timestep,'forward','protocol_work')]
                        shadow_work = data[(equilibration_integrator,switching_integrator,nsteps,timestep,'forward','shadow_work')]
                        total_work = data[(equilibration_integrator,switching_integrator,nsteps,timestep,'forward','total_work')]
                        # Workaround for bug in simulation script storage
                        if switching_integrator == 'GHMC':
                            shadow_work *= 0
                        else:
                            shadow_work = total_work - protocol_work
                        # End workaround
                        plt.plot(protocol_work[:,-1], shadow_work[:,-1], 'b.')

                        plt.hold(True)

                        protocol_work = data[(equilibration_integrator,switching_integrator,nsteps,timestep,'backward','protocol_work')]
                        shadow_work = data[(equilibration_integrator,switching_integrator,nsteps,timestep,'backward','shadow_work')]
                        total_work = data[(equilibration_integrator,switching_integrator,nsteps,timestep,'backward','total_work')]
                        # Workaround for bug in simulation script storage
                        if switching_integrator == 'GHMC':
                            shadow_work *= 0
                        else:
                            shadow_work = total_work - protocol_work
                        # End workaround
                        plt.plot(protocol_work[:,-1], shadow_work[:,-1], 'r.')

                        #plt.xlabel('trajectory length (ps)')
                        if y == len(timesteps_to_try)-1 : plt.xlabel('$2^{%d}$' % np.log2(nsteps))
                        if x == 0 : plt.ylabel('$2^{%d}$' % np.log2(timestep))
                        oldaxis = np.array(plt.axis())
                        limit = max(abs(oldaxis))
                        limit = 10 # DEBUG
                        plt.axis([-limit, +limit, -limit, +limit])
                        if y == len(timesteps_to_try)-1 and x == 0:
                            plt.xticks([-limit, 0, limit])
                            plt.yticks([-limit, 0, limit])
                        else:
                            plt.gca().tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')                        
                        plt.axis('equal')

                #fig.tight_layout()
                fig.savefig('protocol-vs-shadow-work-%s-%s.png' % (equilibration_integrator,switching_integrator))
                #pdf.savefig()
                plt.close()

if __name__ == '__main__':
    analyze()
