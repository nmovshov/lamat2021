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

    # Determine planet and load its observables
    if args.planet.lower() == 'saturn':
        obs = observables.Saturn_winds()
    elif args.planet.lower() == 'jupiter':
        obs = observables.Jupiter_tof4()
    else:
        raise ValueError(
                f"Unsupported target planet {args.planet}.")

    # Make a directory to store output (currently hard coded in gravity)
    # outdir = '{}_{}_output'.format(obs.pname,args.prefix)
    # if not os.path.isdir(outdir):
    #     os.mkdir(outdir)
    # else:
    #     print("\nWARNING: directory {} already exists, ".format(outdir))
    #     print("files may be overwritten.\n")

    # Build the model parameters dict
    params = {}
    params['small'] = obs.m
    params['adjust_small'] = args.adjust_mrot
    params['mtot'] = obs.M*1000
    params['req'] = obs.a0*100
    params['nz'] = args.nzones
    params['verbosity'] = args.verbosity
    params['t1'] = obs.T0
    params['f_ice'] = args.f_ice

    params['ymean_xy'] = args.y_mean
    params['y1_xy'] = args.y1
    params['y2_xy'] = 0.35 # initial guess, adjusted by model
    params['z1'] = args.z1
    params['z2'] = args.z2
    params['ri'] = args.rc
    params['ro'] = args.rt

    params['j2n_rtol']   = args.J_tol
    params['ymean_rtol'] = args.y_tol
    params['mtot_rtol']  = args.M_tol
    params['max_iters_outer'] = args.max_iters

    params['drho_a'] = args.drho_a
    params['drho_w'] = args.drho_w
    params['drho_c'] = args.drho_c

    params['use_gauss_lobatto'] = args.use_gauss_lobatto

    # Initialize eos objects
    try:
        hhe_eos = mh13_scvh.eos()
        z_eos = aneos_pure.eos('ice')
    except OSError:
        raise Exception('Failed to initialize eos; did you unpack eos_data.tar.gz?')

    # Initialize the model
    model = models.threeLayerModel(hhe_eos,z_eos,params,y_adjust_qty='y2_xy')

    # Finally, make a tof4 instance and relax the model
    t = gravity.tof4(model, params)
    t.relax()
    return t, obs

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

    mdlgroup.add_argument('--nzones', type=int, default=4096,
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
    mdl, obs = _main(clargs)
    print(f"Model J2 = {mdl.j2}")
    print(f"Model J4 = {mdl.j4}")
    print(f"Model J6 = {mdl.j6}")
    J_err = np.sqrt(((mdl.j2 - obs.J2)/obs.dJ2)**2 +
                    ((mdl.j4 - obs.J4)/obs.dJ4)**2 +
                    ((mdl.j6 - obs.J6)/obs.dJ6)**2)
    print(f"Model-to-observation Mahalanobis distance = {J_err}")
