import numpy as np
import matplotlib.pyplot as plt

class Channel():

    def __init__(self, conductance, reversal, dt):

        self.conductance = conductance
        self.reversal = reversal
        self.dt = dt

class NMDA(Channel):

    """The NMDA glutamate receptor (excitatory, ligand-gated)"""

    def __init__(self, conductance, reversal, dt):

        super().__init__(conductance, reversal, dt)

class AMPA(Channel):

    """The AMPA glutamate receptor (excitatory, ligand-gated)"""

    def __init__(self, conductance, reversal, dt):

        super().__init__(conductance, reversal, dt)

class DelayedRectifier(Channel):

    def __init__(self, conductance, reversal, dt):

        super().__init__(conductance, reversal, dt)

        self.current_arr = []
        self.state = 0 #closed
        self.p_open = 0

    def alpha(self,v):
        return 0.01*(v+55)/(1-np.exp(-0.1*(v+55))) #opening rate

    def beta(self,v):
        return 0.125*np.exp(-0.0125*(v+65)) #closing rate

    def step(self,v):

        """Update the open probability"""

        dp=(self.alpha(v)*(1-self.p_open)-self.beta(v)*self.p_open)*self.dt
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

class Leak(Channel):
    def __init__(self, conductance, reversal, dt):

        super().__init__(conductance, reversal, dt)

    def step(self, voltage):
        self.current = self.conductance*(voltage-self.reversal)
