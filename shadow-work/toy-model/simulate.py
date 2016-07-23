"""
Generate NCMC simulation data for one-dimensional toy model system.

The reduced potential has the form
```
u(x;\lambda) = (1/2) (x - \lambda)^2 + \lambda
```
with the parameter `\lambda \in [0,1]` sampling one of two stable states at 0 or 1.

"""

################################################################################
# IMPORTS
################################################################################

import numpy as np
import netCDF4 as netcdf
import itertools

################################################################################
# PARAMETERS
################################################################################

timesteps_to_try = [ 2.0**n for n in range(-4, -2) ]
nsteps_to_try = [ 2**n for n in range(0, 2)]
equilibration_integrators_to_try = ['VVVR', 'GHMC'] # integrator to use for equilibration
switching_integrators_to_try = ['VV', 'VVVR', 'GHMC'] # integrator to use for NCMC switches
ntrials = 10 # number of work trials

################################################################################
# METHODS
################################################################################

def potential(x, current_lambda):
    return 0.5 * (x - current_lambda)**2 + current_lambda

def kinetic(v):
    return 0.5 * v**2

def force(x, current_lambda):
    return (x - current_lambda)

def initial_sample(current_lambda=0.0):
    """
    Generate an initial sample from the true equilibrium at the specified lambda.

    """
    x = np.random.randn() + current_lambda
    v = np.random.randn()
    return (x,v)

def equilibrate(sample, timestep, integrator, current_lambda=0.0):
    """
    Equilibrate using the specified integrator.

    Parameters
    ----------
    sample : (x,v)
        (x,v) pair of initial sample
    timestep : float
        Equilibration timestep
    integrator : str
        Name of integrator
    current_lambda : float, optional, default = 0.0
        Current lambda value

    Returns
    -------
    sample : (x,v)
        (x,v) pair of final sample

    """
    (x,v) = sample

    nequilsteps = 500 # number of equilibration steps
    gamma = 1.0 / (20 * timestep)
    b = np.exp(-gamma * timestep)
    sigma = 1.0

    if integrator in 'VVVR':
        for step in range(nequilsteps):
            # velocity perturbation
            v = np.sqrt(b) * v + np.sqrt(1-b) * sigma * np.random.randn()
            # symplectic step
            v = v + 0.5 * timestep * force(x, current_lambda)
            x = x + v * timestep
            v = v + 0.5 * timestep * force(x, current_lambda)
            # velocity perturbation
            v = np.sqrt(b) * v + np.sqrt(1-b) * sigma * np.random.randn()
    elif integrator == 'GHMC':
        nreject = 0
        for step in range(nequilsteps):
            # velocity perturbation
            v = np.sqrt(b) * v + np.sqrt(1-b) * sigma * np.random.randn()
            # Metropolized symplectic step
            (x_old, v_old) = (x,v)
            initial_energy = kinetic(v) + potential(x, current_lambda)
            v = v + 0.5 * timestep * force(x, current_lambda)
            x = x + v * timestep
            v = v + 0.5 * timestep * force(x, current_lambda)
            final_energy = kinetic(v) + potential(x, current_lambda)
            delta_energy = final_energy - initial_energy
            if not (np.random.rand() < np.exp(-delta_energy)):
                (x,v) = (x_old, -v_old)
                nreject += 1
            # velocity perturbation
            v = np.sqrt(b) * v + np.sqrt(1-b) * sigma * np.random.randn()
        print('accepted %8d / %8d (%.1f%%) GHMC steps' % ((nequilsteps-nreject), nequilsteps, float(nequilsteps-nreject)/float(nequilsteps)*100))
    else:
        raise Exception("Integrator must be one of 'VVVR' or 'GHMC'; specified ''%s'" % integrator)

    return (x,v)

def integrate(sample, timestep, integrator, current_lambda):
    (x,v) = sample

    gamma = 1.0 / (20 * timestep)
    b = np.exp(-gamma * timestep)
    sigma = 1.0

    step_initial_energy = kinetic(v) + potential(x, current_lambda)

    if integrator in 'VV':
        # symplectic step
        v = v + 0.5 * timestep * force(x, current_lambda)
        x = x + v * timestep
        v = v + 0.5 * timestep * force(x, current_lambda)
    elif integrator in 'VVVR':
        # velocity perturbation
        v = np.sqrt(b) * v + np.sqrt(1-b) * sigma * np.random.randn()
        # symplectic step
        v = v + 0.5 * timestep * force(x, current_lambda)
        x = x + v * timestep
        v = v + 0.5 * timestep * force(x, current_lambda)
        # velocity perturbation
        v = np.sqrt(b) * v + np.sqrt(1-b) * sigma * np.random.randn()
    elif integrator == 'GHMC':
        # velocity perturbation
        v = np.sqrt(b) * v + np.sqrt(1-b) * sigma * np.random.randn()
        # Metropolized symplectic step
        (x_old, v_old) = (x,v)
        initial_energy = kinetic(v) + potential(x, current_lambda)
        v = v + 0.5 * timestep * force(x, current_lambda)
        x = x + v * timestep
        v = v + 0.5 * timestep * force(x, current_lambda)
        final_energy = kinetic(v) + potential(x, current_lambda)
        delta_energy = final_energy - initial_energy
        if not (np.random.rand() < np.exp(-delta_energy)):
            (x,v) = (x_old, -v_old)
        # velocity perturbation
        v = np.sqrt(b) * v + np.sqrt(1-b) * sigma * np.random.randn()
    else:
        raise Exception("Integrator must be one of 'VVVR' or 'GHMC'; specified ''%s'" % integrator)

    step_final_energy = kinetic(v) + potential(x, current_lambda)
    step_shadow_work = step_final_energy - step_initial_energy

    return ((x,v), step_shadow_work)

def switch(sample, integrator, timestep, nsteps, initial_lambda=0, final_lambda=1):
    protocol_work = np.zeros([nsteps+1], np.float32)
    shadow_work = np.zeros([nsteps+1], np.float32)

    cumulative_shadow_work = 0.0
    cumulative_protocol_work = 0.0

    current_lambda = initial_lambda

    # Initial sample step to make switching symmetric
    [sample, step_shadow_work] = integrate(sample, integrator, timestep, current_lambda=current_lambda)
    cumulative_shadow_work += step_shadow_work

    for step in range(nsteps):
        # Protocol work
        (x,v) = sample
        initial_energy = kinetic(v) + potential(x, current_lambda)
        current_lambda = (final_lambda - initial_lambda) * float(step+1)/float(nsteps) + initial_lambda
        final_energy = kinetic(v) + potential(x, current_lambda)
        cumulative_protocol_work += (final_energy - initial_energy)

        # Shadow work
        [sample, step_shadow_work] = integrate(sample, integrator, timestep, current_lambda=current_lambda)
        cumulative_shadow_work += step_shadow_work

        protocol_work[step+1] = cumulative_protocol_work
        shadow_work[step+1] = cumulative_shadow_work

    total_work = protocol_work + shadow_work

    return [protocol_work, shadow_work, total_work]

################################################################################
# MAIN
################################################################################

def main():
    """
    Generate simulation data.
    """
    filename = 'toy-work.nc'
    ncfile = netcdf.Dataset(filename, 'w')

    # Try all combinations of parameters
    index = 0
    for (timestep, nsteps, equilibration_integrator, switching_integrator) in itertools.product(timesteps_to_try, nsteps_to_try, equilibration_integrators_to_try, switching_integrators_to_try):
        print (timestep, nsteps, equilibration_integrator, switching_integrator)

        # Create storage.
        ncgrp = ncfile.createGroup('combination%d' % index)
        setattr(ncgrp, 'timestep', timestep)
        setattr(ncgrp, 'nsteps', nsteps)
        setattr(ncgrp, 'equilibration_integrator', equilibration_integrator)
        setattr(ncgrp, 'switching_integrator', switching_integrator)
        dimnames = ('ntrials%d' % index, 'nsteps%d' % index)
        ncfile.createDimension(dimnames[0], 0)
        ncfile.createDimension(dimnames[1], nsteps+1)

        for direction in ['forward', 'backward']:
            subgrp = ncgrp.createGroup(direction)
            subgrp.createVariable('protocol_work', np.float32, dimnames)
            subgrp.createVariable('shadow_work', np.float32, dimnames)
            subgrp.createVariable('total_work', np.float32, dimnames)

            if direction == 'forward':
                initial_lambda = 0.0
                final_lambda = 1.0
            else:
                initial_lambda = 1.0
                final_lambda = 0.0

            sample = initial_sample(current_lambda=initial_lambda)
            for trial in range(ntrials):
                sample = equilibrate(sample, timestep, equilibration_integrator, current_lambda=initial_lambda)
                [protocol_work, shadow_work, total_work] = switch(sample, timestep, switching_integrator, nsteps=nsteps, initial_lambda=0, final_lambda=1)
                subgrp.variables['protocol_work'][trial,:] = protocol_work
                subgrp.variables['shadow_work'][trial,:] = total_work
                subgrp.variables['total_work'][trial,:] = total_work

        # Increment counter
        ncfile.sync()
        index += 1


if __name__ == '__main__':
    main()
