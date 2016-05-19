import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import itertools
from collections import deque, defaultdict
from functools import partial
from time import time
import matplotlib.animation as animation



def run(muscle, run_time, dt):
    muscle_fibers = muscle.fibers
    neurons = [fiber.neuron for fiber in muscle_fibers]
    datas = defaultdict(dict)
    f_datas = defaultdict(dict)
    m_datas = defaultdict(list)
    for t in np.arange(0, run_time, dt):
        for ind, neuron in enumerate(neurons):
            for k, v in neuron.state.items():
                datas[ind].setdefault(k, np.array([]))
                datas[ind][k] = np.append(datas[ind][k], v)
        for ind, fiber in enumerate(muscle_fibers):
            f_datas[ind].setdefault('activated', np.array([]))
            f_datas[ind]['activated'] = np.append(f_datas[ind]['activated'], fiber.activated)
        m_datas['force'].append(muscle.force)
        muscle.step()
    for ind, neuron in enumerate(neurons):
        neuron.datas = datas[ind]
    for ind, fiber in enumerate(muscle_fibers):
        fiber.datas = f_datas[ind]
    muscle.datas = m_datas
    return muscle

def plot_neuron_traces(neurons, params=['Vm', 'I_ext']):
    plt.figure()
    for sub in range(len(params)):
        plt.subplot(len(params), 1, sub + 1)
        for neuron in neurons:
            plt.plot(neuron.datas[params[sub]])
    plt.show()

def plot_traces(muscle):
    plt.figure()
    plt.subplot(4,1,1)
    for neuron in muscle.neurons:
        plt.plot(neuron.datas['I_ext'])
    plt.subplot(4,1,2)
    for neuron in muscle.neurons:
        plt.plot(neuron.datas['Vm'])
    plt.subplot(4,1,3)
    for fiber in muscle.fibers:
        plt.plot(fiber.datas['activated'], '+')
    plt.subplot(4,1,4)
    plt.plot(muscle.datas['force'])
    plt.show()
 
    
def spike_rate(neurons, dt, win_len):
    rates = []
    win_len = int(win_len / dt)
    for neuron in neurons:
        if win_len == len(neuron.datas['spiking']):
            rate = sum(neuron.datas['spiking']) / win_len
        else:
            rate = [sum(e) / (win_len * dt) for e in rolling_window(neuron.datas['spiking'], win_len)]
        rates.append(rate)
    return rates

def rolling_window(list, win_len):
    for i in range(len(list)-win_len+1):
        yield [list[i+o] for o in range(win_len)]

if __name__ == '__main__':
# set initial states and time vector
    dt = 0.1
    time_run = 2000   # mseconds
    current_trace = 1e-10*np.abs(np.sin(np.linspace(0, dt, time_run / dt) * 100))
    #current_trace = 0.5e-10*np.ones(time_run / dt)
    state0 = [-70e-03, current_trace]
    current_trace = 1e-10*np.abs(np.sin(np.linspace(0, dt, time_run / dt) * 150))
    state1 = [-70e-03, current_trace]
    t = np.arange(0, time_run, dt)

# run simulation
    neuron00 = EfferentNeuron(*state0)
    neuron01 = EfferentNeuron(*state1)
    muscle_fiber0 = MuscleFiber(neuron00, 5, 100, dt)
    muscle_fiber1 = MuscleFiber(neuron01, 5, 100, dt)
    muscle = Muscle([muscle_fiber0, muscle_fiber1], dt)
    #neurons = [neuron1, neuron2]
    neurons = run(muscle, time_run, dt)
    plot_neuron_traces(muscle.neurons)
    rates = spike_rate(muscle.neurons, dt, 2000)
    plot_traces(muscle)
    plt.show()
