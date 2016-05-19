"""Total force from electric field."""
import sys
import pickle
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../../')
from multiprocessing import Pool, cpu_count
import MorrisLecarElectricField as MLEF
import HodgkinHuxley as HH
import EventTrace as S
import Muscles as M
import Stack as ST
import util

run_time = 500
dt = 0.1
num_cort_neurons = 5
soma_current = 0.3*np.array([100,]*5000)

def run_loop(params):
    external = params
    print 'external: ', external
    cort_neurons = [MLEF.MorrisLecarElectricField(0.2, p=0.2) for e in range(num_cort_neurons)]
    #motor_units = ST.generate_linear_spectrum_motor_units(10, 20e-04, 10e-03, 1.0, 11.0, 5.0, 10.0, 7.0, 14.0)
    motor_units = ST.generate_linear_spectrum_motor_units(100, 10e-04, 200e-03, 0.0, 100.0, 5.0, 250.0, 7.0, 490.0)
    stack = ST.Stack(run_time, cort_neurons, motor_units, cortical_soma_input=lambda t: soma_current[t], cortical_ext_input=lambda t: external)
    stack.run()
    stack.muscle.get_total_force()
    return [e.total_force for e in stack.motor_units]

pool = Pool(cpu_count() - 1)
external_fields = np.arange(0, 60, 1)
data = pool.map(run_loop, external_fields)
pickle.dump(data, open('mvc_grow.p', 'wb'))
