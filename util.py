from itertools import izip
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def find_spikes(trace, threshold, plotflag=0):
    """ finds bursts and determines firing rate, flip the
    plot flag in order to check the threshold chosen is reasonable
    trace: signal, high pass to prevent fake peaks
    threshold: above which peak will be counted
    plotflag: plots peaks

    returns: number of spikes, assumes caller will handle time aspect
    and also a mapping of each event to up and down locations
    """
    result = {}
    spike_times = []
    pos = np.array(trace) > threshold
    npos = ~pos
    down_coords = (pos[:-1] & ~pos[1:]).nonzero()[0]
    up_coords = (npos[:-1] & pos[1:]).nonzero()[0]
    coords_pair = izip(*(up_coords, down_coords))
    for coord in coords_pair:
        result[coord[0]] = (coord[0], coord[1])
        spike_times.append(coord[0] + np.argmax(trace[slice(*coord)]))
    if plotflag == 1:
        plt.subplot(211)
        plt.plot(trace)
        plt.plot(up_coords, [threshold,]*len(up_coords), 'g+')
        plt.plot(down_coords, [threshold,]*len(down_coords), 'r+')
        plt.subplot(212)
        for t, tr in result.items():
            plt.plot(trace[slice(*tr)])
        plt.show()
    return len(spike_times), spike_times

def firing_rate(trace, threshold, dt):
    """ finds spikes and returns firing rate in Hz """
    num, locs = find_spikes(trace, threshold, 0)
    return num / (len(trace) * dt / 1000.0)

def rfd(trace, interval):
    """ Calculates the rate of grip force development:
    trace: 1 dimensional data
    interval: milliseconds """
    results = []
    trace = np.array(trace)
    n_points = interval
    print 'n_points: ', n_points
    for i in range(len(trace) / n_points - 1):
        results.append((trace[(i + 1) * n_points : (i + 2) * n_points] - trace[i * n_points : (i + 1) * n_points]) / interval)
    return results


def bandpass(trace, fnyq, low, high):
    b, a = signal.butter(2, [low / fnyq, high / fnyq], btype='band')
    return signal.filtfilt(b, a, trace)


