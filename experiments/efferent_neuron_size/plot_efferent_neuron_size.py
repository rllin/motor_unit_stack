import sys
sys.path.insert(0, '../../')
import util
import pickle
import matplotlib.pyplot as plt
import numpy as np

data = pickle.load(open('radius.p'))

keys = sorted(data.keys())

print keys
for key in keys:
    plt.plot([util.firing_rate(e, 0, 0.1) for e in data[key]])
plt.savefig('efferent_neuron_size.pdf')
