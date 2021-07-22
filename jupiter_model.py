import numpy as np
from krono import const, gravity, models
from krono.eos import mh13_scvh, aneos_pure
import observables

## Select planet
obs = observables.Jupiter()

## Create eos instances
#  Pass 'ice' for water ice, or pass 'serpentine' to get density typical of
#  silicates. Instead of aneos_pure you can also use aneos_mix which mixes ice
#  and rock on the fly, introducing a large number of eos calls (slowest part
#  of the code). Choosing pure ice or rock is much faster.
hhe_eos = mh13_scvh.eos()
z_eos = aneos_pure.eos('ice')

debug = False
user_params = {
    'z1':0.5*0.015, # outer envelope metallicity (solar is ~0.015)
    'rio':2e-1, # Inner jump's outer (normalized) radius
    'roo':8e-1, # Outer jump's outer (normalized) radius
    'y2_xy':4e-1, # Deep helium mass fraction y/(x+y)=y/(1-z)
    'drho_a':-0.1e-1, # Amplitude (drho/rho) of rho_hhe perturbation.
    'drho_c':10. # Centroid logp of rho_hhe perturbation.
}

## The driver
#  This is a wrapper that calls the gravity and eos routines iteratively to,
#  hopefully, end up with a self-consistent model.
def run_one(par):

    params = {} # will be passed to gravity and model instances

    params['small'] = obs.m # dimensionless
    params['mtot'] = obs.M*1000
    params['req'] = obs.a0*100
    params['nz'] = 4096
    params['verbosity'] = 1 if debug else 0
    params['ymean_xy'] = 0.275
    params['t1'] = obs.T0 # K
    params['drho_a'] = par['drho_a']
    params['drho_w'] = 1.
    params['drho_c'] = par['drho_c']

    # Create dual-cavity model mimicking a three-layer model
    params['z1'] = par['z1']
    params['z2'] = 0.5 # initial guess for inner envelope metallicity
    params['roo'] = par['roo']
    params['roi'] = par['roo'] - 1e-2 # effectively a jump
    params['rio'] = par['rio']
    params['rii'] = par['rio'] - 1e-2 # effectively a jump
    params['y2_xy'] = par['y2_xy']
    params['gradient_shape'] = 'sigmoid'
    model = models.dualCavityModel(hhe_eos, z_eos, params)

    # Convergence tolerances (relative)
    params['j2n_rtol'] = 1e-4
    params['ymean_rtol'] = 1e-4
    params['mtot_rtol'] = 1e-4

    #params['use_gauss_lobatto'] = False # special mesh for Janosz's oscillation code

    # Finally, make a tof4 instance and relax the model
    t = gravity.tof4(model, params)
    t.relax()
    return t

# Run
t = run_one(user_params)
print(f'J2 relative error = {(t.j2 - obs.J2)/obs.J2}')
