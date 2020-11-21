import numpy as np
import matplotlib.pyplot as plt

class DelayedRectifier():

    def __init__(self, conductance, reversal):

        self.conductance = conductance
        self.reversal = reversal
        self.current_arr = []

        self.state = 0 #closed
        self.p_open = 0

    def alpha(self,v):
        return 0.01*(v+55)/(1-np.exp(-0.1*(v+55))) #opening rate

    def beta(self,v):
        return 0.125*np.exp(-0.0125*(v+65)) #closing rate

    def step(self,v):

        """Update the open probability"""

        dp=(self.alpha(v)*(1-self.p_open)-self.beta(v)*self.p_open)*dt
        self.p_open += dp
        self.state = np.random.binomial(1, self.p_open)
        self.current = self.conductance*(v-self.reversal)*self.state
        self.current_arr.append(self.current)

    def simulate(self, v):

        """Run a full simulation for known voltage"""

        self.current = np.zeros(*v.shape)
        for i in range(len(v)-1):
            self.step(v[i])
            self.current[i] = self.state

class Leak():
    def __init__(self, conductance, reversal):

        self.conductance = conductance
        self.reversal = reversal

    def step(self, voltage):
        self.current = self.conductance*(voltage-self.reversal)

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
            self.step_voltage(injected[i])
            self.voltage_arr[i] = self.voltage


T = 0.1 #1s
dt = 1e-3 #1ms
t = np.linspace(0, T, int(T/dt))

injected=np.piecewise(t, [t < 0.025, t >= 0.025, t >= 0.075], [0, 5e-9, 0])
neuron = Compartment(t, injected=injected)

#single 100nS channels
leaks = [Leak(100e-12,-70e-3) for i in range(100)]
neuron.channels += leaks

rectifiers = [DelayedRectifier(100e-12,-70e-3) for i in range(1000)]
neuron.channels += rectifiers
neuron.simulate()

#plot
fig, ax1 = plt.subplots()
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Membrane voltage (mV)', color='red')
ax1.plot(t*1e3, neuron.voltage_arr*1e3, color='red')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Current (nA)', color='blue')
ax2.plot(t*1e3, neuron.current_arr*1e9, color='blue')
ax2.plot(t*1e3, neuron.injected*1e9, color='blue', linestyle='--')

fig.tight_layout()
plt.show()
