#!/bin/env python

from progressbar import ProgressBar
import numpy as np
from simtk import unit, openmm
from simtk.openmm import app
import netCDF4 as netcdf
import copy
import sys

# PARAMETERS
temperature = 300.0 * unit.kelvin
kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
pressure = 1.0 * unit.atmospheres
frequency = 50 # barostat update frequency
timestep = 2.0 * unit.femtoseconds # timestep
nequil = 20 # number of equilibration iterations
nequilsteps = 500 # number of steps per equilibration iteration

# Simulate the system to collect work values
nwork = 1000 # number of work values to collect
tmax = 40 * unit.picoseconds # maximum switching time
nworksteps = 50 # number of steps per work recoridng
nsteps = int(np.round(tmax / timestep)) # total number of steps to integrator for
nworkvals = int(nsteps / nworksteps)

def fix_system(system):
    # Use switching function
    for force in system.getForces():
        if force.__class__.__name__ == 'NonbondedForce':
           force.setUseSwitchingFunction(True)
           force.setSwitchingDistance(6.0 * unit.angstroms)
           force.setReactionFieldDielectric(1e10)
           parameters = force.getPMEParameters()
           print('PME Parameters:')
           print(parameters)
           print('box_size:')
           print(system.getDefaultPeriodicBoxVectors())


def dhfr():
    from simtk.openmm import app
    ff = app.ForceField('amber99sb.xml', 'tip3p.xml')
    pdb = app.PDBFile('5dfr_solv-cube_equil.pdb')
    method = app.PME
    cutoff = 8 * unit.angstroms
    dt = 0.002*unit.picoseconds
    constraints = app.HBonds
    hydrogenMass = None
    system = ff.createSystem(pdb.topology, nonbondedMethod=method, nonbondedCutoff=cutoff, constraints=constraints, hydrogenMass=hydrogenMass, ewaldErrorTolerance=1.0e-6, useDispersionCorrection=True)
    fix_system(system)
    testsystem_name = 'DHFR'

    return [system, pdb.positions, testsystem_name]

def tip3p():
    # Create a TIP3P water box
    from openmmtools import testsystems
    testsystem = testsystems.WaterBox(box_edge=25.0*unit.angstroms, cutoff=9*unit.angstroms, model='tip3p', switch_width=1.5*unit.angstroms, constrained=True, dispersion_correction=True, nonbondedMethod=app.PME, ewaldErrorTolerance=1.0e-6)
    testsystem_name = testsystem.__class__.__name__
    fix_system(testsystem.system)
    return [testsystem.system, testsystem.positions, testsystem_name]

def run():
    if sys.argv[1] == 'dhfr':
        [system, positions, testsystem_name] = dhfr()
    elif sys.argv[1] == 'tip3p':
        [system, positions, testsystem_name] = tip3p()

    precision = sys.argv[2]
    platform_name = sys.argv[3]

    print('%s %s : contains %d particles' % (testsystem_name, precision, system.getNumParticles()))

    # Remove CMMotionRemover and barostat
    indices_to_remove = list()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force_name = force.__class__.__name__
        print(force_name)
        if force_name in ['MonteCarloBarostat', 'CMMotionRemover']:
            print('Removing %s (force index %d)' % (force_name, index))
            indices_to_remove.append(index)
    indices_to_remove.reverse()
    for index in indices_to_remove:
        system.removeForce(index)

    # Add barostat
    barostat = openmm.MonteCarloBarostat(pressure, temperature, frequency)

    # Create OpenMM context
    from openmmtools import integrators
    integrator = integrators.VelocityVerletIntegrator(timestep)
    integrator.setConstraintTolerance(1.0e-8)
    platform = openmm.Platform.getPlatformByName(platform_name)
    if platform_name == 'CUDA':
        platform.setPropertyDefaultValue('CudaPrecision', precision)
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)

    # Equilibrate with barostat
    print('equilibrating...')
    barostat.setFrequency(frequency)
    from progressbar import Percentage, Bar, ETA, RotatingMarker
    widgets = ['equilibration: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]
    progress = ProgressBar(widgets=widgets)
    for iteration in progress(range(nequil)):
        context.setVelocitiesToTemperature(temperature)
        integrator.step(nequilsteps)

    # Get positions, velocities, and box vectors
    state = context.getState(getPositions=True, getVelocities=True)
    box_vectors = state.getPeriodicBoxVectors()
    positions = state.getPositions(asNumpy=True)
    velocities = state.getVelocities(asNumpy=True)
    del context, integrator

    # Remove CMMotionRemover and barostat
    indices_to_remove = list()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force_name = force.__class__.__name__
        print(force_name)
        if force_name in ['MonteCarloBarostat', 'CMMotionRemover']:
            print('Removing %s (force index %d)' % (force_name, index))
            indices_to_remove.append(index)
    indices_to_remove.reverse()
    for index in indices_to_remove:
        system.removeForce(index)

    #
    integrator = integrators.VelocityVerletIntegrator(timestep)
    integrator.setConstraintTolerance(1.0e-8)
    context = openmm.Context(system, integrator, platform)
    context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    context.setVelocities(velocities)

    # Open NetCDF file for writing.
    ncfile = netcdf.Dataset('work-%s-%s-%s.nc' % (testsystem_name, precision, platform_name), 'w')
    ncfile.createDimension('nwork', 0) # extensible dimension
    ncfile.createDimension('nworkvals', nworkvals+1)
    ncfile.createVariable('work', np.float32, ('nwork', 'nworkvals'))
    work = np.zeros([nwork, nworkvals+1], np.float32)
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
    print('start')
    run()
