#------------------------------------------------------------------------------
# Driver for simple 3-layer model. Run
#   python driver_3l_model.py --help
# for list of required and optional parameters.
#------------------------------------------------------------------------------
import sys, os
import numpy as np
import argparse
import observables
from timeit import default_timer as timer
from krono import gravity, const, models
from krono.eos import mh13_scvh, aneos_pure

def _main(args):

    mass = const.mjup
    gm = mass * const.cgrav

    r_vol = const.rjup
    r_pol = const.rjup_pol
    r_eq = const.rjup_eq

    rotation_period = 9.9259 * 60 * 60
    omega_rot = 2. * np.pi / rotation_period
    omega_dyn = np.sqrt(const.cgrav * mass / r_vol ** 3)
    small = (omega_rot / omega_dyn) ** 2

    print(small)

    # z1, rstab, y2_xy, f_ice = 0.05927145799416273,    0.7068565323794515,    0.855706683903526,    0.04392185816793537    # from best model in chain n4

    params = {}
    params['small'] = small
    params['adjust_small'] = False # if True, adjust the nondimensional spin parameter to preserve *dimensional* spin frequency as the model's mean radius changes during iterations
    params['mtot'] = const.mjup
    params['req'] = r_eq
    params['nz'] = 4096
    params['verbosity'] = 1
    params['t1'] = 166. # sushil says that anywhere from 162 to 173-174 is compatible with Voyager occultation data near equator; galileo falls roughly in the middle of this
    params['f_ice'] = 0.5

    # composition choices
    params['ymean_xy'] = 0.275 # M_He / (M_H + M_He); outer iterations will adjust y1_xy to satisfy this
    # one of y1_xy or y2_xy will be adjusted during iterations to approach target ymean_xy, depending on value of argument y_adjust_qty passed to models.{twoLayerModel,etc} below
    params['y1_xy'] = 0.238 # y/(x+y) outside outer cavity
    params['y2_xy'] = 0.35 # y/(x+y) in inner envelope
    params['z1'] = 0.015 # heavy element mass fraction outside inner cavity
    params['z2'] = 0.045 # heavy element mass fraction within inner cavity
    params['ri'] = 0.1 # core boundary fractional radius
    params['ro'] = 0.85 # inner/outer envelope boundary fractional radius

    # relative tolerances
    params['j2n_rtol']   = 1e-4
    params['ymean_rtol'] = 1e-4
    params['mtot_rtol']  = 1e-4
    params['max_iters_outer'] = 199

    params['use_gauss_lobatto'] = False

    # initialize eos objects once, can pass to many tof4 objects
    try:
        hhe_eos = mh13_scvh.eos()
        z_eos = aneos_pure.eos('ice')
    except OSError:
        raise Exception('failed to initialize eos; did you unpack eos_data.tar.gz?')

    model = models.threeLayerModel(hhe_eos, z_eos, params, y_adjust_qty='y2_xy') # y2_xy will be varied during iterations to match desired y1_xy specified above

    # finally make a tof4 instance and relax the model
    t = gravity.tof4(model, params)
    t.relax()

def _PCL():
    # Return struct with command line arguments as fields.

    parser = argparse.ArgumentParser(
        description="Run krono's pseudo 3-layer model with fixed z1 and y1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('planet', choices=['jupiter','saturn'],
        help="Target planet.")

    parser.add_argument('y1', type=float,
        help="y/(x+y) in outer envelope.")

    parser.add_argument('z1', type=float,
        help="Outer envelope metalicity.")

    parser.add_argument('z2', type=float,
        help="Inner envelope metalicity.")

    parser.add_argument('-v', '--verbosity', type=int, default=1,
        help="Control runtime message verbosity.")

    parser.add_argument('--prefix', default='',
        help="Base name for output directory.")

    mdlgroup = parser.add_argument_group('Additional model options')

    mdlgroup.add_argument('--rc', type=float, default=0.1,
        help="Initial guess of core radius (this is the model's " + 
             "adjustable parameter).")

    mdlgroup.add_argument('--rt', type=float, default=0.8,
        help="Envelope inner/outer transition.")

    mdlgroup.add_argument('--adjust-mrot', action='store_true',
        help="Don't sample rotation parameter (use obs.m instead).")

    mdlgroup.add_argument('--nzones', type=int, default=496,
        help="Number of zones (model resolution).")

    mdlgroup.add_argument('--f-ice', type=float, default=0.5,
        help="Not really sure what this does, maybe eos mixing?")

    mdlgroup.add_argument('--y-mean', type=float, default=0.275,
        help="Target M_He/(M_H + M_He) fraction (solar is 0.275).")

    mdlgroup.add_argument('--J-tol', type=float, default=1e-4,
        help="J value convergence tolerance.")

    mdlgroup.add_argument('--y-tol', type=float, default=1e-4,
        help="Mean y value convergence tolerance.")

    mdlgroup.add_argument('--M-tol', type=float, default=1e-4,
        help="Mass convergence tolerance.")

    mdlgroup.add_argument('--max-iters', type=int, default=199,
        help="Stop if fail to converge after this many iterations.")

    mdlgroup.add_argument('--use_gauss_lobatto', action='store_true',
        help="I don't know what this does.")

    eosgroup = parser.add_argument_group('EOS options')

    eosgroup.add_argument('--drho-a', type=float, default=0.0,
        help="Force ad-hoc relative density change (usually negative).")

    eosgroup.add_argument('--drho-c', type=float, default=10.,
        help="Start ad-hoc density change at this log P (cgs).")

    eosgroup.add_argument('--drho-w', type=float, default=1.0,
        help="Smooth ad-hoc density change over log P range (cgs).")

    tofgroup = parser.add_argument_group('TOF options',
        'Options controlling ToF gravity calculation')

    tofgroup.add_argument('--toforder', type=int, default=4, choices=[4],
        help="Theory of figures expansion order.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # Parse command line arguments
    clargs = _PCL()
    print(clargs)