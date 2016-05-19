import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from time import time

class LeakyIntegrateFireNeuron:
    def __init__(self, Rm, tau_m, tau_ref, tau_g, Vth, V_spike, I):
        self.Rm = Rm
        self.tau_m, self.tau_ref, self.tau_g = tau_m, tau_ref, tau_g
        self.Vth, self.V_spike = Vth, V_spike
        self.I = I
        self.time_elapsed = 0
        self.dt = 0.1
        self.last_spike = -99
        self.t_ref = 0.3
        self.E_l = -70
        self.V_reset = -75
        self.Vm = self.E_l
        self.spiking = 0
        self.connections = []
        self.g = []

    def add_connection(self, neuron):
        self.connections.append(neuron)
        self.g.append(0)
    
    def check_connection(self):
        for ind, neuron in enumerate(self.connections):
            if neuron.spiking:
                self.g[ind] += self.g_c

    def d_dt(self, state, t):
        vm = state
        #vm = state[0]
        #g = state[1:]
        d_dt = np.zeros_like(state)
        d_dt[0] = (self.E_l - vm + (self.I * self.Rm)) / self.tau_m
        """
        for i in range(1, len(g)):
            d_dt[i] = -g[i] / self.tau_g
        """
        return d_dt

    def step(self, dt):
        self.time_elapsed += dt
        print self.Vm
        if self.time_elapsed - self.last_spike >= self.t_ref:
            self.Vm = integrate.odeint(self.d_dt, self.Vm, [self.time_elapsed, self.time_elapsed+dt])[1]
            #self.Vm = state[0]
            #self.g = state[1:]
        else:
            self.Vm = self.V_reset
        if self.Vm >= self.Vth:
            self.Vm += self.V_spike
            self.spiking = 1
            self.last_spike = self.time_elapsed
            print 'spiked'
        else:
            self.spiking = 0

N = 0
main_neuron = LeakyIntegrateFireNeuron(10, 10, 10, 1, -55, 20, 10)
neurons = [main_neuron]
"""
for n in range(N):
    neuron = LeakyIntegrateFireNeuron(10, 1, 10, 10, 1, -55, 20, 40)
    main_neuron.add_connection(neuron)
    neurons.append(neuron)
"""
dt = 1./10
s, v = [], []
for i in range(100):
    main_neuron.step(dt)
    v.append(main_neuron.Vm)
    s.append(main_neuron.spiking * 1)
plt.plot(v, '+')
#plt.plot(s, '+')
plt.show()
"""
fig = plt.figure()
ax = fig.add_subplot(111)
lines = []
for ind, neuron in enumerate(neurons):
    line, = ax.plot([], [], 'b', lw=10)
    lines.append(line)

def plot_neurons():
    for ind, neuron in enumerate(neurons):
        line, = ax.plot([2*ind, 2*ind], [2*ind, 2*ind], 'b', lw=10)
        lines.append(line)
    return ax, fig, lines

def animate(i):
    global neurons, dt, lines
    for ind, neuron in enumerate(neurons):
        neuron.step(dt)
        if neuron.spiking:
            lines[ind].set_color('r')
        else:
            lines[ind].set_color('b')
    return lines

t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=100, interval=interval, blit=False, init_func=plot_neurons)

plt.show()
"""
