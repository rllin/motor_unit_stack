import pickle
import sys
sys.path.insert(0, '../../')
import util
import matplotlib.pyplot as plt

datas = pickle.load(open('external_field_p_tuned.p'))

keys = sorted(datas.keys())

for p in keys:
    freqs = [util.firing_rate(e, 0, 0.1) for e in datas[str(p)]]
    plt.plot(freqs)
plt.savefig('ef_ML_tuned.pdf')

