import networkx as nx
import numpy as np

# create nx graph
G = nx.DiGraph()
num_nodes = 10

for i in range(num_nodes):
    
    if i!=7:
        print("adding node num:", i)
        G.add_node(i, 
                    volume_id=i, 
                    in_volume_layer_id=i,
                    vivl_id=(i,i))

        if i < num_nodes-1:
            G.add_edge(i, i+1)
            G.add_edge(i+1, i)


G.remove_node(7)


print("Nodes in G: ")
for n in G.nodes(data=True):
    print(n)
# print("Edges in G: ")
# for e in G.edges(data=True):
#     print(e)

# get all attributes
vivl_id=nx.get_node_attributes(G,'vivl_id')
print("vivl_id attribute: \n", vivl_id)

vivl_id_values = vivl_id.values()
# should output no duplicates:
if len(vivl_id_values) == len(set(vivl_id_values)): print("no duplicates")
else: print("duplicates")


# add 2nd node in same layer
G.add_node(11, volume_id=4, in_volume_layer_id=4, vivl_id=(4,4))
G.add_edge(3,11)
G.add_edge(11,3)

print("Nodes in G: ")
for n in G.nodes(data=True):
    print(n)
# print("Edges in G: ")
# for e in G.edges(data=True):
#     print(e)

# get all attributes
vivl_id=nx.get_node_attributes(G,'vivl_id')
print("vivl_id attribute: \n", vivl_id)

vivl_id_values = vivl_id.values()
print("vivl_id values:\n", vivl_id_values)
print(type(vivl_id_values))

# should output duplicates:
if len(vivl_id_values) == len(set(vivl_id_values)): print("no duplicates")
else: print("duplicates")



# testing the order of output node_idx
print("testing node output order")
G.add_node(7, volume_id=7, in_volume_layer_id=7, vivl_id=(7,7))
G.add_edge(6,7)
G.add_edge(7,6)
G.add_node(-1, volume_id=4, in_volume_layer_id=4, vivl_id=(4,4))
G.add_edge(-1,6)
G.add_edge(6,-1)

print("Nodes in G: ")
for n in G.nodes(data=True):
    print(n)
print("Edges in G: ")
for e in G.edges(data=True):
    print(e)

vivl_id=nx.get_node_attributes(G,'vivl_id')
print("vivl_id attribute: \n", vivl_id)


volume_id=nx.get_node_attributes(G,'volume_id')
print("volume_id attribute: \n", volume_id)


# order the graph by its edge connections