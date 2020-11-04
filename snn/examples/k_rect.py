from snn.models import *
import matplotlib.pyplot as plt

t = np.linspace(0, 20, 1000) #20ms
dt = 0.05 #1ms
v=np.piecewise(t, [t < 10, t >= 10, t >= 15], [-100, 10, -100])
p,i = delayed_rectifier(v,dt,nchannels=10)

p = np.mean(p,axis=0)
i = np.mean(i,axis=0)

#plt.plot(t, v)
plt.plot(t, p)
#plt.plot(t, i)
plt.show()
