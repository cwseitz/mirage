import numpy as np
import matplotlib.pyplot as plt
from mirage.models import *

T = 0.1 #1s
dt = 1e-3 #1ms
t = np.linspace(0, T, int(T/dt))

injected=np.piecewise(t, [t < 0.025, t >= 0.025, t >= 0.075], [0, 5e-9, 0])
neuron = Compartment(t, injected=injected)

#single 100nS channels
leaks = [Leak(100e-12,-70e-3, dt) for i in range(100)]
neuron.channels += leaks

rectifiers = [DelayedRectifier(100e-12,-70e-3, dt) for i in range(1000)]
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
