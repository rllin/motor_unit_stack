import sys
sys.path.insert(0, '../../')
import MorrisLecarElectricField as MLEF
import EventTrace as S
import util

import pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

run_time = 2000
dt = 0.1
#excitatory_synapses_soma = S.SynapsePool(100, 10.0, run_time, peak=0.10, excitatory_flag=1, dt=0.1)
excitatory_synapses_soma = S.SynapsePool(100, 10.0, run_time, peak=0.183, excitatory_flag=1, dt=0.1)
soma_current = excitatory_synapses_soma.event_trace()

def run_loop(params):
    external, p = params
    print 'params: ', params
    neuron = MLEF.MorrisLecarElectricField(0.2, soma_current=lambda t: soma_current[t], external_field=lambda t: external, p=p)
    neuron.run(run_time)
    return neuron.datas

"""
data = run_loop((0, 0.2))
plt.plot(data[:,0])
plt.show()
"""
pool = Pool(cpu_count() - 1)
external_fields = np.arange(0, 200, 1)
datas = {}
for p in np.arange(0.1, 1.0, 0.1):
    data = pool.map(run_loop, zip(external_fields, p * np.ones(len(external_fields))))
    datas[str(p)] = data
pickle.dump(datas, open('external_field_p_tuned.p', 'wb'))
