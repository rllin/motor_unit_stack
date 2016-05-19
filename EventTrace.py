import numpy as np
import util
import bisect

class EventTrace:
    """ This generates a trace composed of differences of two exponentials 
    per spike event from the input trace.  Can be used for both synapses and
    twitch summations."""
    def __init__(self, trace, peak, tau_rise=5.0, tau_decay=7.0, dt=0.1):
        """
        peak: max value reached per isolated event
        tau_rise: time to peak
        tau_decay: time to decay
        trace: time series
        dt: delta t
        """
        self.trace = trace
        self.dt = dt
        self.result_trace = np.zeros_like(self.trace)
        self.time_trace = np.arange(0, len(trace) * self.dt, self.dt)
        self.params = {
            'peak': peak,     # uS
            'tau_rise': tau_rise,    # ms
            'tau_decay': tau_decay
        }
        self.threshold = 0
        num, spike_times = util.find_spikes(self.trace, self.threshold)
        for spike in spike_times:
            self.result_trace += self.event(spike)

    def event(self, t0):
        trace = np.zeros_like(self.time_trace)
        tau_rise = self.params['tau_rise']
        tau_decay = self.params['tau_decay']
        peak = self.params['peak']
        decay = np.exp(-(self.time_trace) / tau_decay)
        rise = np.exp(-(self.time_trace) / tau_rise)
        t_peak = (tau_decay * tau_rise) / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        norm = -np.exp(-(t_peak) / tau_rise) + np.exp(-(t_peak) / tau_decay)
        blip = peak * (decay - rise) / norm
        trace[t0:] = blip.clip(min=0)[:len(blip) - t0]
        return trace

class Synapse:
    """Uses EventTrace event trains"""

    def __init__(self, expected_interval, run_time, poisson_flag=1, dt=0.1):
        self.run_time = run_time
        self.expected_interval = expected_interval
        self.poisson_flag = poisson_flag
        self.dt = dt
        if self.poisson_flag:
            self.events = np.int64(np.cumsum(np.random.poisson(self.expected_interval, self.run_time / self.dt / self.expected_interval)) / self.dt)
        else:
            self.events = np.int64(np.cumsum(np.hstack((self.dt, self.expected_interval* np.ones(self.run_time / self.dt)))) / self.dt)
        self.events = self.events[:bisect.bisect_left(self.events, self.run_time / self.dt)]
        self.trace = np.zeros(self.run_time / self.dt)
        self.trace[self.events] = 1
    
    def get_event_trace(self, peak, rise, decay, dt):
        self.event_trace = EventTrace(self.trace, peak, rise, decay, dt)
        return self.event_trace


class SynapsePool:
    def __init__(self, num_synapse, expected_interval, run_time,
            peak=0.065, rise=5.0, decay=7.0,
            excitatory_flag=1, dt=0.1):
        self.num_synapse = num_synapse
        self.expected_interval = expected_interval
        self.excitatory_flag = excitatory_flag
        self.dt = dt
        self.synapses = [Synapse(expected_interval, run_time, 1, self.dt) for e in range(self.num_synapse)]
        if self.excitatory_flag:
            self.peak, self.rise, self.decay = peak, rise, decay
        else:
            self.peak, self.rise, self.decay = peak, rise, decay

    def event_trace(self):
        self.event_traces = []
        for s in self.synapses:
            trace = s.get_event_trace(self.peak, self.rise, self.decay, self.dt) 
            self.event_traces.append(trace)
        if self.excitatory_flag:
            return sum([e.result_trace for e in self.event_traces])
        else:
            return -sum([e.result_trace for e in self.event_traces])


