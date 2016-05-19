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

class Stack:
    def __init__(self,
            run_time,
            cortical_neurons,
            motor_units,
            cortical_soma_input=lambda t: 0,
            cortical_dend_input=lambda t: 0,
            cortical_ext_input=lambda t: 0):
        
        self.run_time = run_time
        self.motor_units = motor_units
        self.cortical_neurons, self.efferent_neurons = cortical_neurons, [e.neuron for e in self.motor_units]
        self.cortical_soma_input, self.cortical_dend_input, self.cortical_ext_input = cortical_soma_input, cortical_dend_input, cortical_ext_input
        
    def connect_external(self):
        for neuron in self.cortical_neurons:
            neuron.set_soma_ext(self.cortical_soma_input)
            neuron.set_dend_ext(self.cortical_dend_input)
            neuron.set_field_ext(self.cortical_ext_input)

    def run_external(self):
        MLEF.run_neurons(self.cortical_neurons, self.run_time)

    def connect_internal(self):
        for cort_neuron in self.cortical_neurons:
            for motor_unit in self.motor_units:
                motor_unit.neuron.set_I_ext(lambda t: cort_neuron.synapse.result_trace[t])


    def run_internal(self): 
        HH.run_neurons(self.efferent_neurons, self.run_time)

    def define_muscle(self):
        self.muscle = M.Muscle(self.motor_units)
        return self.muscle
        
    def run(self):
        self.connect_external()
        self.run_external()
        self.connect_internal()
        self.run_internal()
        self.define_muscle()
        self.muscle.get_total_force()
        

def generate_linear_spectrum_motor_units(
        num,
        low_radius, high_radius,
        low_force, high_force,
        low_rise, high_rise,
        low_decay, high_decay):
    motor_units = []
    radii = np.linspace(low_radius, high_radius, num)
    peak = np.linspace(low_force, high_force, num)
    rise = np.linspace(low_rise, high_rise, num)
    decay = np.linspace(low_decay, high_decay, num)
     
    for i in range(num):
        neuron = HH.HodgkinHuxley_passive(radii[i])
        motor_units.append(M.MotorUnit(neuron, i+1, peak[i], rise[i], decay[i]))
    return motor_units
