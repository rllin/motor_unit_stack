"""This is a basic example of one motor unit with one entity
per layer.  The layers include:
    1) a Morris Lecar 2 compartmental cortical neuron that 
       can be influenced by an external electric field
    2) a passive Hodgkin Huxley single compartmental neuron 
       representing a alpha motor neuron that can receive 
       synaptic current input
    3) a muscle unit that twitches for every spike event of the
       alpha moter neuron and can perform twitch summation
"""
import numpy as np
import matplotlib.pyplot as plt
import MorrisLecarElectricField as MLEF
import HodgkinHuxley as HH
import EventTrace as S
import Muscles as M


run_time = 200
input_current = np.abs(30 + 50*np.sin(100 * np.linspace(0, 0.1, run_time / 0.1)))
input_current = 40 * np.ones(run_time / 0.1)
cort_neuron = MLEF.MorrisLecarElectricField(soma_current=lambda t: input_current[t])
MLEF.run_neurons([cort_neuron], run_time)
synapse = S.EventTrace(cort_neuron.datas[:,0], peak=0.2)
effe_neuron = HH.HodgkinHuxley_passive(I=lambda t: synapse.result_trace[t])
HH.run_neurons([effe_neuron], run_time)
muscle_fiber0 = M.MuscleFiber(effe_neuron, 1, 5.0, 7.0)
muscle_fiber1 = M.MuscleFiber(effe_neuron, 5, 40, 80.0)
muscle = M.MotorUnit([muscle_fiber0, muscle_fiber1], 2, )

plt.figure()
plt.subplot(511)
plt.plot(input_current)
plt.subplot(512)
plt.plot(cort_neuron.datas[:,0])
plt.subplot(513)
plt.plot(synapse.result_trace)
plt.subplot(514)
plt.plot(effe_neuron.datas[:,0])
plt.subplot(515)
plt.plot(muscle.total_force)
plt.show()

