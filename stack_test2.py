import numpy as np
import matplotlib.pyplot as plt
import MorrisLecarElectricField as MLEF
import HodgkinHuxley as HH
import EventTrace as S
import Muscles as M
import Stack as ST


run_time = 200
input_current = np.abs(30 + 50*np.sin(100 * np.linspace(0, 0.1, run_time / 0.1)))
input_current = 50 * np.ones(run_time / 0.1)
cort_neuron0 = MLEF.MorrisLecarElectricField(0.2)
cort_neuron1 = MLEF.MorrisLecarElectricField(0.2)

motor_units = ST.generate_linear_spectrum_motor_units(2, 20e-04, 100e-04, 10, 10, 5.0, 10.0, 7.0, 14.0)
#motor_units = ST.generate_linear_spectrum_motor_units(2, 20e-04, 200e-04, 10, 100, 5.0, 50.0, 7.0, 70.0)
stack = ST.Stack(run_time, [cort_neuron0, cort_neuron1], motor_units, cortical_soma_input=lambda t: input_current[t])
stack.run()


plt.figure()
plt.subplot(611)
plt.plot(input_current)
plt.subplot(612)
plt.plot(cort_neuron0.datas[:,0])
plt.plot(cort_neuron1.datas[:,0])
plt.subplot(613)
plt.plot(cort_neuron0.synapse.result_trace)
plt.plot(cort_neuron1.synapse.result_trace)
plt.subplot(614)
plt.plot(stack.efferent_neurons[0].datas[:,0])
plt.plot(stack.efferent_neurons[1].datas[:,0])
plt.subplot(615)
plt.plot(stack.motor_units[0].total_force)
print stack.motor_units[0].total_force
print stack.motor_units[1].total_force
plt.plot(stack.motor_units[1].total_force)
plt.subplot(616)
plt.plot(stack.muscle.total_force)
plt.savefig('example_stack.pdf')
