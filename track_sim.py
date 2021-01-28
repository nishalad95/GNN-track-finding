import os
import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from itertools import count
from scipy.stats import norm
import scipy.stats
from numpy.linalg import inv

class GNN_Measurement(object) :
    def __init__(self, x, y, t, label = -1, n = None) :
        self.x = x
        self.y = y
        self.t = t # track inclination - gradient
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
        tau0 = (m2.y-m1.y)/dx # gradient
        y0 =(m1.y*m2.x-m2.y*m1.x+self.start*(m2.y-m1.y))/dx
        if tau0 > self.max_tau or tau0 < self.min_tau : return 0
        if y0 > self.max_y0 or y0 < self.min_y0 : return 0
        return 1





# define variables
sigma0 = 0.1 #r.m.s of track position measurements
S = np.matrix([[sigma0**2, 0], [0, sigma0**2]])

Lc = np.array([1,2,3,4,5,6,7,8,9,10]) #detector layer coordinates along x-axis
Nl = len(Lc) #number of layers
start = 0

# In total there are 7 tracks, initial and final y track positions given below
y0 = np.array([-3,   1, -1,   3, -2,  2.5, -1.5]) #track positions at start
yf = np.array([12, -4,  5, -14, -6, -24,   0]) #final track positions

# tau - dy/dx - track inclination
# tau0 - gradient calculated from final and initial positions in x and y
tau0 = (yf-y0)/(Lc[-1]-start)
Ntr = len(y0) #number of simulated tracks

# plotting the intial track toy model
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
        G.add_node(nNodes, GNN_Measurement=gm, coord_Measurement=(xc[l], yc[l] + nu))
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


# add measurements to each node and edge
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

                # edge state vector
                state_vector = [m1.y]
                grad = (m2.y - m1.y) / (m2.x - m1.x)
                state_vector.append(grad)

                # covariance
                H = np.matrix([[1, 0], [1/(m1.x - m2.x), 1/(m2.x - m1.x)]])
                cov = H.dot(S).dot(H.T)

                # print(state_vector)
                # print(cov)

                G.add_edge(*edge, state_vector=state_vector, covariance=cov)



# compute mean and variance of edge orientation to each node
for n in range(nNodes):
    dy_dx = []
    node_coords = (G.nodes[n]["GNN_Measurement"].x, G.nodes[n]["GNN_Measurement"].y)
    connected_nodes = G[n]
    for cn in connected_nodes:
        # connected_node_coords = G.nodes[cn]['coord']
        connected_node_coords = (G.nodes[cn]["GNN_Measurement"].x, G.nodes[cn]["GNN_Measurement"].y)
        dy = node_coords[1] - connected_node_coords[1]
        dx = node_coords[0] - connected_node_coords[0]
        grad = dy / dx
        dy_dx.append(grad)

    G.nodes[n]["edge_mean"] = np.mean(dy_dx)
    G.nodes[n]["edge_var"] = np.var(dy_dx)



print('found',nPairs,'hit pairs')
# print(G.edges())
# print(G.number_of_nodes())
# print(G.number_of_edges())
# print(list(G.nodes(data=True))

ax2.set_title('Hit pairs')
ax2.grid()
plt.tight_layout()
plt.show()

# draw the graph network
# plot_network_graph(G, "Simulated tracks as Graph network \n with degree of nodes plotted in colour", cmap=plt.cm.hot)

# filter graph: remove all vertices with edge orientation above threshold
threshold = 1.0
filteredNodes = [x for x,y in G.nodes(data=True) if y['edge_var'] > threshold]
for n in filteredNodes:
    G.remove_node(n)

# plot filtered graph
# plot_network_graph(G, "Filtered Graph Edge orientation var <" + str(threshold) + ", \n weakly connected subgraphs")

# extract subgraphs - weakly connected
diGraph = nx.to_directed(G)
subGraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(diGraph)]


# plot the subgraphs
fig, ax = plt.subplots()
for s in subGraphs:
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
    pos=nx.get_node_attributes(s,'coord_Measurement')
    ec = nx.draw_networkx_edges(s, pos, alpha=0.5)
    nc = nx.draw_networkx_nodes(s, pos, node_color=color[0], node_size=75)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
major_ticks = np.arange(0, 12, 1)
ax.set_xticks(major_ticks)
plt.xlabel("layer in x axis")
plt.ylabel("y coord")
plt.title("Edge orientation var <" + str(threshold) + ", weakly connected subgraphs")
plt.axis('on')
plt.show()


# k-means clustering on edges with KL-distance

def computeKL(cluster_centre, edge):
    p1 = cluster_centre[2]
    p2 = edge[2]
    
    # KL = Trace[(cov1 - cov2)*(inv(cov2) - inv(cov1))] + [(mean1 - mean2).T * (inv(cov1) + inv(cov2)) * (mean1 - mean2)]
    cov1 = p1['covariance']
    cov2 = p2['covariance']
    inv_cov1 = inv(cov1)
    inv_cov2 = inv(cov2)
    mean1 = np.array(p1['state_vector'])
    mean2 = np.array(p2['state_vector'])

    trace = np.trace((cov1 - cov2).dot((inv_cov2 - inv_cov1)))
    matrix = (np.transpose(mean1 - mean2)).dot(inv_cov1 + inv_cov2).dot(mean1 - mean2)
    return trace + matrix



for subGraph in subGraphs:
    for node in subGraph.nodes():
        print("node number: ", node)
        print(subGraph.edges(node, data=True))
        print("cluster centre: ", list(subGraph.edges(node, data=True))[0])

        cluster_centre = list(subGraph.edges(node, data=True))[0]

        for edge in subGraph.edges(node, data=True):
            print("edge", edge)
            kl_distance = computeKL(cluster_centre, edge)
            print("kl distance: ", kl_distance)
        #     print(edge)
        #     print(edge[2])
        #     edge_data = edge[2]
        # print(type(s.edges(n, data=True)))

        # run k-means
        # initialize the center to a random candidate, based in num of clusters
        # compute the KL distance between the centre and every other candidate
        # assign the candidate to the cluster
        # recompute means of clusters
        # repeat this until convergence
