import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from . import aneos_rhot

class eos:
    ''' does pure ice or pure rock from aneos ice or serpentine tables '''
    def __init__(self, material='ice', path_to_data=None, extended=False):
        if not path_to_data:
            import os
            path_to_data = os.environ['KRONO_DIR'] + '/eos/data'
        if extended:
            path = f'{path_to_data}/aneos_{material}_pt_hi-p.dat'
        else:
            path = f'{path_to_data}/aneos_{material}_pt.dat'
        self.names = 'logrho', 'logt', 'logp', 'logu', 'logs' # , 'chit', 'chirho', 'gamma1'
        self.data = np.genfromtxt(path, names=self.names, usecols=(0, 1, 2, 3, 4)) # will fail if haven't saved version of aneos_*_pt.dat with eight columns

        # this version of aneos.py loads tables already regularized to rectangular in P, T.
        # thus use PT as a basis so we can use RegularGridInterpolator (fast.)
        self.logpvals = np.unique(self.data['logp'])
        self.logtvals = np.unique(self.data['logt'])

        assert len(self.logpvals) == len(self.logtvals), 'aneos was implemented assuming square grid in p-t'
        self.npts = len(self.logpvals)
        self.logrho_on_nodes = np.zeros((self.npts, self.npts))
        self.logu_on_nodes = np.zeros((self.npts, self.npts))
        self.logs_on_nodes = np.zeros((self.npts, self.npts))

        for i, logpval in enumerate(self.logpvals):
            data_this_logp = self.data[self.data['logp'] == logpval]
            for j, logtval in enumerate(self.logtvals):
                data_this_logp_logt = data_this_logp[data_this_logp['logt'] == logtval]
                self.logrho_on_nodes[i, j] = data_this_logp_logt['logrho']
                self.logu_on_nodes[i, j] = data_this_logp_logt['logu']
                self.logs_on_nodes[i, j] = data_this_logp_logt['logs']

        pt_basis = (self.logpvals, self.logtvals)
        self._get_logrho = RegularGridInterpolator(pt_basis, self.logrho_on_nodes)
        self._get_logu = RegularGridInterpolator(pt_basis, self.logu_on_nodes)
        self._get_logs = RegularGridInterpolator(pt_basis, self.logs_on_nodes)

        self.rhot_eos = aneos_rhot.eos(material, path_to_data)

    def get_logrho(self, logp, logt):
        return self._get_logrho((logp, logt))

    def get(self, logp, logt, f_ice):
        '''f_ice is ignored'''
        res = {}
        logrho = self._get_logrho((logp, logt))
        logu = self._get_logu((logp, logt))
        logs = self._get_logs((logp, logt))

        res['logrho'] = logrho
        res['logu'] = logu
        res['logs'] = logs

        # some derivs are easier to evaluate from the original rho, t basis
        rhot_res = self.rhot_eos.get(logrho, logt)
        res['rhot'] = rhot_res['rhot'] # -delta
        res['rhop'] = rhot_res['rhop']

        res['chirho'] = res['rhop'] ** -1
        res['chit'] = -res['rhot'] / res['rhop']
        res['grada'] = np.nan * np.ones_like(logrho)
        res['gamma1'] = np.nan * np.ones_like(logrho)

        return res

