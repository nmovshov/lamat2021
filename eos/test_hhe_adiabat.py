import numpy as np
# import scvh
from eos import scvh
import time

class hhe_adiabat:
    
    def __init__(self, nz, y, t1, integration_style='brute_force', interp='linear'):
        
        self.nz = nz
        self.p = np.logspace(6, 14., self.nz)[::-1]
        self.t = np.zeros_like(self.p)
        self.rho = np.zeros_like(self.p)
        self.t[-1] = t1
        
        self.hhe_eos = scvh.eos()
        
        t0 = time.time()
                
        if integration_style == 'brute_force':
            for k in np.arange(self.nz)[::-1]:
                dlnp = np.log(self.p[k-1]) - np.log(self.p[k])
                eos_res = self.hhe_eos.get(np.log10(self.p[k]), np.log10(self.t[k]), y)
                grada = eos_res['grada']
                self.rho[k] = 10 ** eos_res['logrho']
                assert not np.isnan(grada), 'NaN in grada'
                if k == 0: break
                dlnt = grada * dlnp
                self.t[k-1] = self.t[k] * (1. + dlnt)
            print(f'brute_force: et={time.time()-t0:.3f} s; tcenter={self.t[0]:.6f} K')
        elif integration_style == 'solve_ivp':
            from scipy.interpolate import interp1d
            from scipy.integrate import solve_ivp
            self.t[:-1] = 1e4 # take T=1e4 K for initial eos call; leave surface temperature alone
            for i in np.arange(11):
                self.grada = self.hhe_eos.get(np.log10(self.p), np.log10(self.t), y)['grada']
                interp_grada = interp1d(self.p, self.grada, fill_value='extrapolate', kind=interp)
                def dtdp(p, t):
                    return t / p * interp_grada(p)
                p_eval = self.p[::-1] # integrate from surface to center
                sol = solve_ivp(dtdp, (p_eval[0], p_eval[-1]), np.array([self.t[-1]]), t_eval=p_eval)
                assert sol.success, 'failed in integrate_temperature'
                self.t[:] = sol.y[0][::-1]
                print(f'solve_ivp: kind={interp:8}; iter={i:5}; et={time.time()-t0:.3f} s; tcenter={self.t[0]:.6f} K')
        else:
            print(f'integration_style {integration_style} not recognized; doing nothing.')
            
            
    def write_table(self, label):
        with open('%s.adiabat' % label, 'w') as f:
            f.write('%20s %20s %20s\n' % ('rho', 'p', 't'))
            for trio in zip(self.rho, self.p, self.t):
                f.write('%20e %20e %20e\n' % trio)
            print('wrote %s.adiabat' % label)

ad = hhe_adiabat(1024, 0.238, 140., 'brute_force')
ad = hhe_adiabat(1024, 0.238, 140., 'solve_ivp')
ad = hhe_adiabat(1024, 0.238, 140., 'solve_ivp', 'cubic')
    