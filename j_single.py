import numpy as np
# import const
# from pathlib import Path

from krono import const, gravity, models
from krono.eos import mh13_scvh, aneos_pure

# create eos instances
hhe_eos = mh13_scvh.eos()
z_eos = aneos_pure.eos('ice') # water ice; can instead pass 'serpentine' to get density typical of silicates
# instead of aneos_pure you can also use aneos_mix which mixes ice and rock on the fly, introducing a large
# number of eos calls (slowest part of the code). choosing pure ice or rock is much faster.

def logerr(e, uid, theta):
    errtype = str(type(e)).split('.')[-1].split("\'")[0]
    if '<' in errtype:
        raise ValueError(errtype)
    with open(f'got_{errtype}.dat', 'a') as fw:
        fw.write(f'{uid} ')
        [fw.write(f'{qty} ') for qty in theta]
        fw.write('\n')
        print(f'{uid}', end=' ')
        [print(f'{qty}', end=' ') for qty in theta]
        print(f'-> got_{errtype}.dat\n')
    if debug:
        raise e

def run_one(par):

    ''' driver to a run a single ToF model for user-specified parameter dictionary `par`'''

    params = {} # will be passed to gravity.tof4 and models.{model type} instances

    omega0 = np.sqrt(const.cgrav * const.mjup/ const.rjup ** 3)
    omega_jup = np.pi * 2 / (9.933 * 3600) # 9h 56m = 9.933h
    small = (omega_jup / omega0) ** 2 # sometimes denoted m; m_rot in the paper
    params['small'] = small

    params['mtot'] = const.mjup
    params['req'] = 71492e5
    params['nz'] = 4096
    params['verbosity'] = 1 if debug else 0
    params['ymean_xy'] = 0.275
    params['t1'] = 165.
    params['drho_a'] = par['drho_a']
    params['drho_w'] = 1.
    params['drho_c'] = par['drho_c']
    params['max_iters_outer'] = 999

    # create a dual-cavity model, seems to perform well even when mimicking a three-layer model as set up here
    params['z1'] = 3*0.015 # ~3x solar outer envelope metallicity
    params['z2'] = 0.5 # z in inner envelope layer; will be varied to satisfy total mass for this model type
    params['rii'] = par['rio'] - 1e-2 # treat inner composition gradient as very thin; effectively a jump
    params['rio'] = par['rio']
    params['roi'] = par['roo'] - 1e-2 # treat outer composition gradient as very thin; effectively a jump
    params['roo'] = par['roo']
    params['y2_xy'] = par['y2_xy']
    params['gradient_shape'] = 'sigmoid' # otherwise 'linear' for piecewise-linear z and y/(x+y) # ignored if 'model_type' is 'threeLayerModel'
    model = models.dualCavityModel(hhe_eos, z_eos, params)

    # relative tolerances
    params['j2n_rtol'] = 1e-5
    params['ymean_rtol'] = 1e-4
    params['mtot_rtol'] = 1e-5

    params['use_gauss_lobatto'] = False # special mesh for Janosz's oscillation code

    # finally make a tof4 instance and relax the model
    t = gravity.tof4(model, params)
    try:
        t.relax()
    except Exception as err:
        t.error = err

    if hasattr(t, 'error'):
        logerr(t.error, t.uid, np.array([par[qty] for qty in par]))
        # if debug: raise t.error

    return t

user_params = {
    'rio':2e-1, # inner stable region's outer radius, normalized to total radius
    'roo':6e-1, # outer stable region's outer radius, normalized to total radius
    'y2_xy':4e-1, # deep helium mass fraction y/(x+y)=y/(1-z)
    'drho_a':-0.03, # amplitude (drho/rho) of sigmoid-shaped rho_hhe perturbation.
    'drho_c':10. # centroid logp of sigmoid-shaped rho_hhe perturbation. if close to surface then will affect density everywhere in the planet; if very deep then will have close to no effect.
}

debug = True
t = run_one(user_params)
print(f'got j2 = {t.j2}')
print(f'got j4 = {t.j4}')
print(f'got j6 = {t.j6}')