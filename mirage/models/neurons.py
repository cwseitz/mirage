import numpy as np
import matplotlib.pyplot as plt

class Compartment():

    def __init__(self, time, cap=281e-12, area=1, v_rest=-70e-3, injected=None):

        #area is in mm^2
        #cap is the total capacitance

        self.dt = time[1]-time[0]
        if injected is not None:
            self.injected = injected
        else:
            self.injected = np.zeros_like(time)

        self.time = time
        self.voltage = v_rest; self.current = 0
        self.voltage_arr = np.zeros_like(time)
        self.current_arr = np.zeros_like(time)

        self.channels = []
        self.cap = cap
        self.area = area

    def add_channel(self, channel):
        self.channels.append(channel)

    def step_voltage(self,injected=0):

        dv = (self.dt/self.cap)*((injected/self.area)-self.current)
        self.voltage += dv

    def step_current(self):

        self.current = 0
        for channel in self.channels:
            channel.step(self.voltage)
            self.current += channel.current

    def simulate(self):

        for i in range(len(self.time)):
            self.step_current()
            self.current_arr[i] = self.current
            self.step_voltage(self.injected[i])
            self.voltage_arr[i] = self.voltage
