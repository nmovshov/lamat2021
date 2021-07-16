'''
provides a class for simple one-stop-shopping eos calls.
not called internally anywhere in krono.
'''

class eos:
    def __init__(self):
        from . import mh13_scvh
        from . import aneos_mix
        self.hhe_eos = mh13_scvh.eos()
        self.z_eos = aneos_mix.eos()

    def get(self, logp, logt, y_xy, z, f_ice):
        '''

        inputs

        logp  : (length nz) log10 pressure in dyne cm^-2
        logt  : (length nz) log10 temperature in K
        y_xy  : (length nz) Y/(X+Y) = Y/(1-Z), the He mass fraction relative to H+He
        z     : (length nz) Z, the heavy element mass fraction, itself a sum of ice and rock
        f_ice : (scalar)    X_ice, mass fraction of H2O ice relative to ice+serpentine (X_ice = 1 - X_serpentine)

        outputs

        res : a dictionary of eos results for the H/He/H2O/serpentine blend.

        densities are blended in the additive volume approximation. grada, gamma1, etc. come straight
        from the H/He eos and ignore modifications from Z which should be modest.

        '''

        assert logt.shape == logp.shape, 'all eos inputs but f_ice must have same shape'
        assert y_xy.shape == logt.shape, 'all eos inputs but f_ice must have same shape'
        assert z.shape == logt.shape, 'all eos inputs but f_ice must have same shape'

        hhe_res = self.hhe_eos.get(logp, logt, y_xy)
        z_res = self.z_eos.get(logp, logt, f_ice)

        res = {}

        rhoinv = z / 10 ** z_res['logrho'] + (1. - z) / 10 ** hhe_res['logrho']
        res['rho'] = rhoinv ** -1

        res['grada'] = hhe_res['grada']
        res['gamma1'] = hhe_res['gamma1']

        # and other things?

        return res
