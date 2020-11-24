import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mirage.models import Compartment

#Instantiate the neurons
t = np.linspace(0,1,100)
neuron0 = Compartment(t)
neuron1 = Compartment(t)
neuron2 = Compartment(t)
neuron3 = Compartment(t)


#Build the graph
G = nx.Graph()
G.add_nodes_from([neuron1,neuron2,neuron3])
G.add_edge(neuron0,neuron1)
G.add_edge(neuron1,neuron2)
G.add_edge(neuron2,neuron3)
G.add_edge(neuron1,neuron3)


#Show the graph
pos = nx.spring_layout(G, iterations=200)
nx.draw(G, pos, node_color=range(4), node_size=800, cmap=plt.cm.coolwarm)
plt.show()
