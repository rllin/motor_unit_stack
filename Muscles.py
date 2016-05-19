import EventTrace as S
import numpy as np

class MuscleFiber:
    def __init__(self, neuron, peak_twitch_force, tau_rise, tau_decay):
        self.neuron = neuron
        self.peak_twitch_force = peak_twitch_force
        self.tau_rise, self.tau_decay = tau_rise, tau_decay

    def grab_trace(self):
        self.trace = S.EventTrace(self.neuron.datas[:,0], self.peak_twitch_force, self.tau_rise, self.tau_decay)
        return self.trace

class MotorUnit:
    def __init__(self, neuron, num_fibers, peak_twitch_force, tau_rise, tau_decay):
        self.neuron = neuron
        self.num_fibers = num_fibers
        self.peak_twitch_force = np.random.normal(1, 0.05, self.num_fibers).clip(min=0) * peak_twitch_force
        self.tau_rise = np.random.normal(1, 0.05, self.num_fibers).clip(min=0) * tau_rise
        self.tau_decay = np.random.normal(1, 0.05, self.num_fibers).clip(min=0) * tau_decay
        self.muscle_fibers = [MuscleFiber(self.neuron, self.peak_twitch_force[e], self.tau_rise[e], self.tau_decay[e]) for e in range(self.num_fibers)]

    def get_total_force(self):
        for fiber in self.muscle_fibers:
            fiber.grab_trace()
        self.total_force = sum([e.trace.result_trace for e in self.muscle_fibers])
        return self.total_force

class Muscle:
    def __init__(self, motor_units):
        self.motor_units = motor_units

    def get_total_force(self):
        for motor_unit in self.motor_units:
            motor_unit.get_total_force()
        self.total_force = sum([e.total_force for e in self.motor_units])
        return self.total_force
