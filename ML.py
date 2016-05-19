import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
###########################################################################
## Problem 4: Morris-Lecar model with expanded channel gating dynamics   ##
###########################################################################
# Constants
C_m  = 1.0      # membrane capacitance, in uF/cm^2
g_Ca = 1.1      # maximum conducances, in mS/cm^2
g_K  = 2.0 
g_L  = 0.5 
E_Ca = 100      # Nernst reversal potentials, in mV
E_K  = -70 
E_L  = -50 

# Channel gating kinetics
# Functions of membrane voltage
def m_infty(V): return (1 + sp.tanh((V + 1) / 15)) / 2
def w_infty(V): return (1 + sp.tanh(V / 30)) / 2
def tau_w(V): return 5 / sp.cosh(V / 60)              # in ms

# Membrane currents (in uA/cm^2)
def I_Ca(V): return g_Ca * m_infty(V) * (V - E_Ca)
def I_K(V, w): return g_K * w * (V - E_K)
def I_L(V): return g_L * (V - E_L)
def I_ext(t): return 40+sp.floor(t/20)

t_start = 0
t_stop = 200
t_step = 0.1
t = sp.arange(t_start, t_stop, t_step)

#####################################
# 4.1 Numerical solution (ODE solver)
#####################################
# Differential equation for X(1)=Vm and X(2)=w
def dXdt(X,t):
    V, w = X
    dVdt = (I_ext(t) - I_Ca(V) - I_K(V, w) - I_L(V)) / C_m
    dwdt = (w_infty(V) - w) / tau_w(V)
    return dVdt, dwdt

X = odeint(dXdt, [-70, 0], t)

Vm_numerical = X[:,0] # the first column is the V values
w_numerical  = X[:,1] # the second column is the w values

plt.plot(t, Vm_numerical)
plt.plot(t, w_numerical)
plt.plot(t, I_ext(t))
plt.show()
