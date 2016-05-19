import sys
sys.path.insert(0, '../../../../Trials/emg/util/')
sys.path.insert(0, '../../')
import util
import pickle
import matplotlib.pyplot as plt
import numpy as np
import analytics as an
import processing as pr

#datas = pickle.load(open('force_widest.p'))
datas = pickle.load(open('mvc_grow.p'))

max_rfds = []
for tr in np.sum(datas, axis=1):
    rgfd = an.rgfd(tr, 10000, 50)
    max_rfds.append(np.max(rgfd))
    plt.subplot(311)
    plt.plot(tr)
    plt.subplot(312)
    plt.plot(rgfd)
#plt.savefig('electric_field_rfd_wider.pdf')
plt.subplot(313)
plt.plot(max_rfds)
#plt.show()
plt.savefig('mvc_grow.pdf')

