import os
import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from itertools import count


class GNN_Measurement(object) :
    def __init__(self, x, y, t, label = -1, n = None) :
        self.x = x
        self.y = y
        self.t = t
        self.track_label = label
        self.node = n

class HitPairPredictor() :
    def __init__(self, start_x, y0_range, tau0_range) :
        self.start = start_x
        self.min_y0 = -y0_range
        self.max_y0 =  y0_range
        self.min_tau = -tau0_range
        self.max_tau =  tau0_range
    
    def predict(self, m1, m2, start = 0) :
        dx = m2.x-m1.x
        tau0 = (m2.y-m1.y)/dx
        y0 =(m1.y*m2.x-m2.y*m1.x+self.start*(m2.y-m1.y))/dx
        if tau0 > self.max_tau or tau0 < self.min_tau : return 0
        if y0 > self.max_y0 or y0 < self.min_y0 : return 0
        return 1



def plot_network_graph(G, title, output=None, cmap=plt.cm.jet):
    fig, ax = plt.subplots()

    # create colour map based on degree attribute
    groups = set(nx.get_node_attributes(G,'degree').values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = G.nodes()
    colors = [mapping[nodes[n]['degree']] for n in nodes()]

    # drawing nodes and edges separately
    pos=nx.get_node_attributes(G,'coord')
    ec = nx.draw_networkx_edges(G, pos, alpha=0.5)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
                                node_size=100, cmap=cmap, ax=ax)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    major_ticks = np.arange(0, 11, 1)
    ax.set_xticks(major_ticks)
    plt.xlabel("layer in x axis")
    plt.ylabel("y coord")
    plt.title(title)
    plt.colorbar(nc)
    plt.axis('on')
    if output: plt.savefig(output)
    plt.show()



# define variables
sigma0 = 0.1 #r.m.s of track position measurements

Lc = np.array([1,2,3,4,5,6,7,8,9,10]) #detector layer coordinates along x-axis
Nl = len(Lc) #number of layers
start = 0

y0 = np.array([-3,   1, -1,   3, -2,  2.5, -1.5]) #track positions at start
yf = np.array([12, -4,  5, -14, -6, -24,   0]) #final track positions

tau0 = (yf-y0)/(Lc[-1]-start)

Ntr = len(y0) #number of simulated tracks

fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(1, 2, 1)

major_ticks = np.arange(0, 11, 1)
ax1.set_xticks(major_ticks)

G = nx.Graph()
nNodes = 0



mcoll = [] #collections of track position measurements
for l in range(Nl) : mcoll.append([])

for i in range(Ntr) :
    yc = y0[i] + tau0[i]*(Lc[:] - start)
    xc = Lc[:]
    for l in range(Nl) : 
        nu = sigma0*np.random.normal(0.0,1.0) #random error
        gm = GNN_Measurement(xc[l], yc[l] + nu, tau0[i], i, n=nNodes)
        mcoll[l].append(gm)
        # add hits to the Graph network as nodes
        G.add_node(nNodes, GNN_Measurement=gm)
        nNodes += 1
    ax1.scatter(xc, yc)

ax1.set_title('Ground truth')
ax1.grid()

#for l in range(Nl) : print('Layer',l,'with',len(mcoll[l]),'hits')
# create hit pairs in the next two neighbouring layers using pair predictor

# generate and plot pair predictions
hpp = HitPairPredictor(start, 7.0, 3.5) #max y0 and tau value
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xticks(major_ticks)
nPairs = 0

for L1 in range(Nl) :
    for L2 in range(L1+1,Nl) :
        if L2-L1 > 2 : break #use the next 2 layers only
        for m1 in mcoll[L1] :
            for m2 in mcoll[L2] :
                result = hpp.predict(m1, m2)
                if result ==  0 : continue #pair (m1,m2) is rejected
                nPairs += 1
                ax2.plot([m1.x, m2.x],[m1.y,m2.y],alpha = 0.5)
                edge = (m1.node, m2.node)
                G.add_edge(*edge)


# number of edges associated to each node
for i in range(nNodes) :
    degree = len(G[i])
    G.nodes[i]["degree"] = degree
    G.nodes[i]["coord"] = (G.nodes[i]["GNN_Measurement"].x, G.nodes[i]["GNN_Measurement"].y)


print('found',nPairs,'hit pairs')
# print(G.number_of_nodes())
# print(G.number_of_edges())
# print(list(G.nodes(data=True)))

ax2.set_title('Hit pairs')
ax2.grid()
plt.tight_layout()
plt.show()

# draw the graph network
plot_network_graph(G, "Simulated tracks as Graph network \n with degree of nodes plotted in colour", cmap=plt.cm.hot)

# filter graph: remove all vertices with degree above threshold
threshold = 4
filteredNodes = [x for x,y in G.nodes(data=True) if y['degree'] > threshold]
for n in filteredNodes:
    G.remove_node(n)

# plot filtered graph
# plot_network_graph(G, "Filtered nodes degree <= 4")

# extract subgraphs
diGraph = nx.to_directed(G)
subGraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(diGraph)]


fig, ax = plt.subplots()
for s in subGraphs:
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
    pos=nx.get_node_attributes(s,'coord')
    ec = nx.draw_networkx_edges(s, pos, alpha=0.5)
    nc = nx.draw_networkx_nodes(s, pos, node_color=color[0], node_size=75)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
major_ticks = np.arange(0, 11, 1)
ax.set_xticks(major_ticks)
plt.xlabel("layer in x axis")
plt.ylabel("y coord")
plt.title("Nodes with degree <= 4, weakly connected subgraphs")
plt.axis('on')
plt.show()