import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import deque
import EventTrace as S

class HodgkinHuxley_passive:
    def __init__(self, radius=20e-04, I=lambda t: 0, dt=0.1):
        self.dt = dt
        self.I_all = []
        E_params = {
            'E_leak' : -70,                     # mV
            'G_leak' : 0.003,                   # uS
            #'C_m'    : 0.03,                    # nF
            'C_m'    : 100,                       # uF * cm ** -2
            'I_ext'  : I,                       # nA
            'G_pas'  : 1e-9,                   # umho * cm ** -2
            'radius' : radius                   # cm
        }
        E_params['area'] = 4 * np.pi * E_params['radius'] ** 2
        E_params['rm'] = 1.0 / (E_params['G_pas'] * E_params['area'])
        E_params['cm'] = E_params['C_m'] * E_params['area']
        Na_params = {
            'Na_E'          : 50,               # mV
            'Na_G'          : 1.0,              # uS
            'k_Na_act'      : 3.0e+0,
            'A_alpha_m_act' : 0.2,
            'B_alpha_m_act' : -40,
            'C_alpha_m_act' : 1.0,
            'A_beta_m_act'  : 0.06,
            'B_beta_m_act'  : -49.0,
            'C_beta_m_act'  : 20.0,
            'l_Na_inact'    : 1.0,
            'A_alpha_m_inact' : 0.08,
            'B_alpha_m_inact' : -40.0,
            'C_alpha_m_inact' : 1.0,
            'A_beta_m_inact'  : 0.4,
            'B_beta_m_inact'  : -36.0,
            'C_beta_m_inact'  : 2.0
        }
        K_params = {
            'k_E'           : -90,              # mV
            'k_G'           : 0.2,              # uS
            'k_K'           : 4.0,
            'A_alpha_m_act' : 0.02,
            'B_alpha_m_act' : -31.0,
            'C_alpha_m_act' : 0.8,
            'A_beta_m_act'  : 0.005,
            'B_beta_m_act'  : -28.0,
            'C_beta_m_act'  : 0.4
        }
        self.params = {
            'E_params'  : E_params,
            'Na_params' : Na_params,
            'K_params'  : K_params
        }
        self.state = [-70, 0, 1, 0, 0]

    def set_I_ext(self, I_ext):
        self.params['E_params']['I_ext'] = I_ext

    def d_dt(self, state, t):
        E, m, h, n, p = state
        Epar = self.params['E_params']
        Na   = self.params['Na_params']
        K    = self.params['K_params']

        # external current (from "voltage clamp", other compartments, other neurons, etc)
        I_ext = Epar['I_ext']

        # calculate Na rate functions and I_Na
        alpha_act = Na['A_alpha_m_act'] * (E-Na['B_alpha_m_act']) / (1.0 - np.exp((Na['B_alpha_m_act']-E) / Na['C_alpha_m_act']))
        beta_act = Na['A_beta_m_act'] * (Na['B_beta_m_act']-E) / (1.0 - np.exp((E-Na['B_beta_m_act']) / Na['C_beta_m_act']) )
        dmdt = ( alpha_act * (1.0 - m) ) - ( beta_act * m )

        alpha_inact = Na['A_alpha_m_inact'] * (Na['B_alpha_m_inact']-E) / (1.0 - np.exp((E-Na['B_alpha_m_inact']) / Na['C_alpha_m_inact']))
        beta_inact  = Na['A_beta_m_inact'] / (1.0 + (np.exp((Na['B_beta_m_inact']-E) / Na['C_beta_m_inact'])))
        dhdt = ( alpha_inact*(1.0 - h) ) - ( beta_inact*h )

        # Na-current:
        I_Na =(Na['Na_E']-E) * Na['Na_G'] * (m**Na['k_Na_act']) * h

        # calculate K rate functions and I_K
        alpha_kal = K['A_alpha_m_act'] * (E-K['B_alpha_m_act']) / (1.0 - np.exp((K['B_alpha_m_act']-E) / K['C_alpha_m_act']))
        beta_kal = K['A_beta_m_act'] * (K['B_beta_m_act']-E) / (1.0 - np.exp((E-K['B_beta_m_act']) / K['C_beta_m_act']))
        dndt = ( alpha_kal*(1.0 - n) ) - ( beta_kal*n )
        I_K = (K['k_E']-E) * K['k_G'] * n**K['k_K']

        # leak current
        I_leak = (Epar['E_leak']-E) * Epar['G_leak']

        # calculate derivative of E
        #dEdt = (I_leak + I_K + I_Na + I_ext(t)) / Epar['C_m']
        dEdt = (I_leak + I_K + I_Na + I_ext(t) - E / Epar['rm']) / Epar['cm']
        statep = [dEdt, dmdt, dhdt, dndt, dEdt * (I_K + I_Na + I_leak + I_ext(t))]

        return statep

    def step(self, time_run):
        self.state = odeint(self.d_dt, self.state, np.arange(0, time_run, self.dt))
        return self.state
    
def run_neurons(neurons, run_time, pltflag=0):
    for ind, neuron in enumerate(neurons):
        neuron.datas = neuron.step(run_time)
        if pltflag:
            plt.plot(neuron.datas[:,0])
    if pltflag:
        plt.show()
    return neurons


class HodgkinHuxley_active:
    def __init__(self, I, dt):
        self.dt = dt
        E_params = {
            'E_leak' : -70,                     # mV
            'G_leak' : 0.003,                   # uS
            'C_m'    : 0.03,                    # nF
            'I_ext'  : I,                       # nA
        }
        Na_params = {
            'Na_E'          : 50,               # mV
            'Na_G'          : 1.0,              # uS
            'k_Na_act'      : 3.0e+0,
            'A_alpha_m_act' : 0.2,
            'B_alpha_m_act' : -40,
            'C_alpha_m_act' : 1.0,
            'A_beta_m_act'  : 0.06,
            'B_beta_m_act'  : -49.0,
            'C_beta_m_act'  : 20.0,
            'l_Na_inact'    : 1.0,
            'A_alpha_m_inact' : 0.08,
            'B_alpha_m_inact' : -40.0,
            'C_alpha_m_inact' : 1.0,
            'A_beta_m_inact'  : 0.4,
            'B_beta_m_inact'  : -36.0,
            'C_beta_m_inact'  : 2.0
        }
        K_params = {
            'k_E'           : -90,              # mV
            'k_G'           : 0.2,              # uS
            'k_K'           : 4.0,
            'A_alpha_m_act' : 0.02,
            'B_alpha_m_act' : -31.0,
            'C_alpha_m_act' : 0.8,
            'A_beta_m_act'  : 0.005,
            'B_beta_m_act'  : -28.0,
            'C_beta_m_act'  : 0.4
        }
        Ca_params = {
            'ca_E'          : 150,              # mV
            'ca_G'          : 0.0,              # uS
            'ca_K'          : 5.0,
            'A_alpha_m' : 0.08,
            'B_alpha_m' : -10.0,
            'C_alpha_m' : 11.0,
            'A_beta_m'  : 0.001,
            'B_beta_m'  : -10.0,
            'C_beta_m'  : 0.5,
            'ca_G_k'        : 0.01,             # uS
            'rho_AP'        : 4e-03,                # ms ** -1 * mV ** -1
            'delta_AP'      : 30e-03,               # ms ** -1
        }
        Nmda_params = {
            'nmda_E'        : 0.0,
            'A_alpha_m_act' : 0.7,
            'C_alpha_m_act' : 17.0,
            'A_beta_m_act'  : 0.1,
            'C_beta_m_act'  : 17.0,
            'rho_nmda'      : 0.5e-03,              # ms ** -1 * mV ** -1
            'delta_nmda'    : 3e-03,                # ms ** -1
            'nmda_G'        : 0.2 
        }
        self.params = {
            'E_params'  : E_params,
            'Na_params' : Na_params,
            'K_params'  : K_params,
            'Ca_params' : Ca_params,
            'Nmda_params': Nmda_params
        }
        self.state = [-70, 0, 1, 0, 0, 0]
 
    def d_dt(self, state, t):
        E, m, h, n, q, Ca_AP = state
        Epar = self.params['E_params']
        Na   = self.params['Na_params']
        K    = self.params['K_params']
        Ca   = self.params['Ca_params']
        # external current (from "voltage clamp", other compartments, other neurons, etc)
        I_ext = Epar['I_ext']

        # calculate Na rate functions and I_Na
        alpha_act = Na['A_alpha_m_act'] * (E-Na['B_alpha_m_act']) / (1.0 - np.exp((Na['B_alpha_m_act']-E) / Na['C_alpha_m_act']))
        beta_act = Na['A_beta_m_act'] * (Na['B_beta_m_act']-E) / (1.0 - np.exp((E-Na['B_beta_m_act']) / Na['C_beta_m_act']) )
        dmdt = ( alpha_act * (1.0 - m) ) - ( beta_act * m )

        alpha_inact = Na['A_alpha_m_inact'] * (Na['B_alpha_m_inact']-E) / (1.0 - np.exp((E-Na['B_alpha_m_inact']) / Na['C_alpha_m_inact']))
        beta_inact  = Na['A_beta_m_inact'] / (1.0 + (np.exp((Na['B_beta_m_inact']-E) / Na['C_beta_m_inact'])))
        dhdt = ( alpha_inact*(1.0 - h) ) - ( beta_inact*h )

        # Na-current:
        I_Na =(Na['Na_E']-E) * Na['Na_G'] * (m**Na['k_Na_act']) * h

        # calculate K rate functions and I_K
        alpha_kal = K['A_alpha_m_act'] * (E-K['B_alpha_m_act']) / (1.0 - np.exp((K['B_alpha_m_act']-E) / K['C_alpha_m_act']))
        beta_kal = K['A_beta_m_act'] * (K['B_beta_m_act']-E) / (1.0 - np.exp((E-K['B_beta_m_act']) / K['C_beta_m_act']))
        dndt = ( alpha_kal*(1.0 - n) ) - ( beta_kal*n )
        I_K = (K['k_E']-E) * K['k_G'] * n**K['k_K']
        
        # calculate Ca rate functions and I_Ca
        alpha_ca = Ca['A_alpha_m'] * (E - Ca['B_alpha_m']) / (1.0 - np.exp((Ca['B_alpha_m'] - E) / Ca['C_alpha_m']))
        beta_ca = Ca['A_beta_m'] * (Ca['B_beta_m'] - E) / (1.0 - np.exp((E - Ca['B_beta_m']) / Ca['C_beta_m']))
        dqdt = (alpha_ca * (1.0 - q) ) - (beta_ca * q)
        I_Ca = (Ca['ca_E'] - E) * Ca['ca_G'] * q ** Ca['ca_K']

        dCa_apdt = ((Ca['ca_E'] - E) * Ca['rho_AP'] * q ** Ca['ca_K']) - (Ca['delta_AP'] * Ca_AP)
        I_KCa = (K['k_E'] - E) * Ca['ca_G_k'] * Ca_AP

        # leak current
        I_leak = (Epar['E_leak']-E) * Epar['G_leak']

        # calculate derivative of E
        dEdt = (I_leak + I_K + I_Na + I_Ca + I_KCa + I_ext) / Epar['C_m']
        statep = [dEdt, dmdt, dhdt, dndt, dqdt, dCa_apdt]

        return statep

    def step(self, time_run):
        self.state = odeint(self.d_dt, self.state, np.arange(0, time_run, self.dt))
        return self.state

if __name__ == '__main__':
    neuron = HodgkinHuxley_passive(radius=20e-04, I=lambda t: 3)
    run_neurons([neuron], 100, 1)
    R = (1.0 / (1e-9 * 4 * np.pi * 20e-04 ** 2))
    print np.trapz(neuron.datas[:,0]**2 / R) / 100 * 10e3 * 8e8
    plt.show()
