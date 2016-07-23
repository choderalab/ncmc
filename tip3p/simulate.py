#!/bin/env python

from progressbar import ProgressBar
import numpy as np
from simtk import unit, openmm
from simtk.openmm import app
import netCDF4 as netcdf

# PARAMETERS
temperature = 300.0 * unit.kelvin
kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
pressure = 1.0 * unit.atmospheres
frequency = 50 # barostat update frequency
timestep = 2.0 * unit.femtoseconds # timestep
nequil = 100 # number of equilibration iterations
nequilsteps = 500 # number of steps per equilibration iteration

# Simulate the system to collect work values
nwork = 20 # number of work values to collect
tmax = 40 * unit.picoseconds # maximum switching time
nworksteps = 50 # number of steps per work recoridng
nsteps = int(np.round(tmax / timestep)) # total number of steps to integrator for
nworkvals = int(nsteps / nworksteps)

def run():
    # Create a TIP3P water box
    from openmmtools import testsystems
    waterbox = testsystems.WaterBox(box_edge=25.0*unit.angstroms, cutoff=9*unit.angstroms, model='tip3p', switch_width=1.5*unit.angstroms, constrained=True, dispersion_correction=True, nonbondedMethod=app.CutoffPeriodic, ewaldErrorTolerance=1.0e-6)
    print('System contains %d molecules' % (waterbox.system.getNumParticles() / 3))

    # Add barostat
    barostat = openmm.MonteCarloBarostat(pressure, temperature, frequency)

    # Create OpenMM context
    from openmmtools import integrators
    integrator = integrators.VelocityVerletIntegrator(timestep)
    #platform = openmm.Platform.getPlatformByName('OpenCL')
    #platform.setPropertyDefaultValue('OpenCLPrecision', 'double')
    context = openmm.Context(waterbox.system, integrator)
    context.setPositions(waterbox.positions)

    # Equilibrate with barostat
    print('equilibrating...')
    barostat.setFrequency(frequency)
    from progressbar import Percentage, Bar, ETA, RotatingMarker
    widgets = ['equilibration: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]
    progress = ProgressBar(widgets=widgets)
    for iteration in progress(range(nequil)):
        context.setVelocitiesToTemperature(temperature)
        integrator.step(nequilsteps)
    # Disable barostat
    barostat.setFrequency(0)

    # Open NetCDF file for writing.
    ncfile = netcdf.Dataset('work.nc', 'w')
    ncfile.createDimension('nwork', 0) # extensible dimension
    ncfile.createDimension('nworkvals', nworkvals+1)
    ncfile.createVariable('work', np.float64, ('nwork', 'nworkvals'))
    work = np.zeros([nwork, nworkvals+1], np.float64)
    for i in range(nwork):
        context.setVelocitiesToTemperature(temperature)
        integrator.step(nequilsteps) # equilibrate
        state = context.getState(getEnergy=True)
        initial_energy = state.getPotentialEnergy() + state.getKineticEnergy()
        widgets = ['Work %5d / %5d: ' % (i, nwork), Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]
        progress = ProgressBar(widgets=widgets)
        for workval in progress(range(nworkvals)):
            integrator.step(nworksteps)
            state = context.getState(getEnergy=True)
            current_energy = state.getPotentialEnergy() + state.getKineticEnergy()
            work[i,workval+1] = (current_energy - initial_energy) / kT
            ncfile.variables['work'][i,workval+1] = work[i,workval+1]
        print(work[i,:])
        ncfile.sync()

if __name__ == '__main__':
    run()
