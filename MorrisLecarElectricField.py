import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import EventTrace as S

class MorrisLecarElectricField:
    def __init__(self, synapse_g, external_field=lambda t: 0, soma_current=lambda t: 0, dend_current=lambda t: 0, p=0.5, dt=0.1):
        """
        Inputs: each functions
            external_field: potential difference of the two compartments due to an external electric field
            soma_current: current injection at the soma
            dend_current: current injection at the dendrite
        p: geometric factor
        dt: delta t
        """
        self.external_field = external_field    # mV
        self.soma_current = soma_current        # uamp * cm ** -2
        self.dend_current = dend_current
        self.synapse_g = synapse_g
        self.p = p
        self.dt = dt
        self.time_elapsed = 0
        self.E_params = {
            'E_na': 50,
            'E_k': -100,
            'E_sl': -70,
            'E_dl': -70,   # mV
            'g_na': 20,
            'g_k': 20,
            'g_sl': 2,
            'g_dl': 2,     # msiemens * cm ** -2
            'C_m': 2.0,      # ufarad * cm ** -2
            'beta_m': -1.2,
            'beta_w': 0.0,
            'gamma_m': 18,
            'gamma_w': 10,   # mV
            'varphi_w': 0.15, # msecond ** -1
            'g_c': 1.1,        # msiemens * cm ** -2
        }
        self.ext_params = {
            'V_ds': self.external_field,     # mV
            'I_s': self.soma_current,
            #'I_d': 0e-08,        # uamp * cm ** -2
            'I_d': self.dend_current
        }
        self.params = {
            'E_params'  : self.E_params,
            'ext_params': self.ext_params
        }
        self.state = [-60, -60, 0]
        self.e_par = self.E_params

    def m_inf(self, v):
        return 0.5 * (1.0 + np.tanh((v - self.e_par['beta_m']) / self.e_par['gamma_m']))

    def w_inf(self, v):
        return 0.5 * (1.0 + np.tanh((v - self.e_par['beta_w']) / self.e_par['gamma_w']))

    def tau_w(self, v):
        return 1.0 / np.cosh((v - self.e_par['beta_w']) / self.e_par['gamma_w'])

    def I_na(self, v):
        return self.e_par['g_na'] * self.m_inf(v) * (v - self.e_par['E_na'])

    def I_k(self, v, w):
        return self.e_par['g_k'] * w * (v - self.e_par['E_k'])

    def I_sl(self, v):
        return self.e_par['g_sl'] * (v - self.e_par['E_sl'])

    def I_dl(self, v):
        return self.e_par['g_dl'] * (v - self.e_par['E_dl'])
    
    def set_soma_ext(self, I_ext):
        self.params['ext_params']['I_s'] = I_ext

    def set_dend_ext(self, I_ext):
        self.params['ext_params']['I_d'] = I_ext

    def set_field_ext(self, V_ext):
        self.params['ext_params']['V_ds'] = V_ext
        
    def d_dt(self, state, t):
        v_s, v_d, w = state
        e_par = self.params['E_params']
        ext_par = self.params['ext_params']
        g_na, g_k, g_sl, g_dl = e_par['g_na'], e_par['g_k'], e_par['g_dl'], e_par['g_dl']
        E_na, E_k, E_sl, E_dl = e_par['E_na'], e_par['E_k'], e_par['E_sl'], e_par['E_dl']
        beta_m, beta_w, gamma_m, gamma_w = e_par['beta_m'], e_par['beta_w'], e_par['gamma_m'], e_par['gamma_w']
        C_m, g_c = e_par['C_m'], e_par['g_c']
        varphi_w = e_par['varphi_w']
        v_ds, I_s, I_d = ext_par['V_ds'](t), ext_par['I_s'](t), ext_par['I_d'](t)

        I_ds = g_c * (v_d + v_ds - v_s)

        dV_ddt = (I_d / (1 - self.p) - I_ds / (1 - self.p) - self.I_dl(v_d)) / C_m
        dV_sdt = (I_s / self.p + I_ds / self.p - self.I_na(v_s) - self.I_k(v_s, w) - self.I_sl(v_s)) / C_m
        dwdt = varphi_w * (self.w_inf(v_s) - w) / self.tau_w(v_s)

        statep = [dV_sdt, dV_ddt, dwdt]
        return statep
        
    def step(self, time_run):
        self.state = odeint(self.d_dt, self.state, np.arange(0, time_run, self.dt))
        return self.state

    def run(self, run_time):
        self.datas = self.step(run_time)
        self.synapse = S.EventTrace(self.datas[:,0], peak=self.synapse_g)
        
def run_neurons(neurons, run_time, pltflag=0):
    for ind, neuron in enumerate(neurons):
        neuron.run(run_time)
        if pltflag:
            plt.plot(neuron.datas[:,0])
    if pltflag:
        plt.show()
    return neurons
