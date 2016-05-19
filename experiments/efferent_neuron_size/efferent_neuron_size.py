"""Experiment to show that electric field can link to activating higher sized neurons"""
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

run_time = 1000
dt = 0.1
excitatory_synapses_soma = S.SynapsePool(100, 10.0, run_time, peak=0.10, excitatory_flag=1, dt=0.1)
soma_current = excitatory_synapses_soma.event_trace()

def run_loop(params):
    external_field, radii = params
    print 'radii: ', radii
    cort_neuron = MLEF.MorrisLecarElectricField(0.2, soma_current=lambda t: soma_current[t], external_field=lambda t: external_field, p=0.2)
    MLEF.run_neurons([cort_neuron], run_time)
    synapse = S.EventTrace(cort_neuron.datas[:,0], peak=0.2)
    effe_neuron = HH.HodgkinHuxley_passive(radii, I=lambda t: synapse.result_trace[t])
    HH.run_neurons([effe_neuron], run_time)    
    return effe_neuron.datas[:,0]

pool = Pool(cpu_count())
radii = np.arange(20e-04, 200e-04, 60e-04)
external_fields = np.arange(0, 200, 1)
datas = {}
for r in radii:
    data = pool.map(run_loop, zip(external_fields, r * np.ones(len(external_fields))))
    datas[str(r)] = data
pickle.dump(datas, open('radius.p', 'wb'))
