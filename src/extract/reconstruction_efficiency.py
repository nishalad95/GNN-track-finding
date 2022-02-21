import csv
import pandas as pd
import numpy as np
from statistics import mode
import os
import networkx as nx



# UNITS OF MOMEMTUM: Gev/c
truth = pd.read_csv("src/trackml_mod/truth/event000001000-truth.csv", sep=',')

# get truth particles above pT cut
truth = truth.assign(tpT=lambda row: (np.sqrt(row.tpx**2 + row.tpy**2 + row.tpz**2)))
print("original length of truth applying pT cut", len(truth))

pt_cut = 1.  # units of momentum GeV/c
truth = truth.loc[truth.tpT >= pt_cut]
print(truth)
print("pt_cut:", pt_cut, "GeV/c")
print("number of hits associated with pt cut:", len(truth))

unique_particle_ids = truth.particle_id.unique()
print("number of unique particles: ", len(unique_particle_ids))

# check if each particle is associated to >4 hits --> our criterai for a good reference track
n=4                     # minimum number of hits for good track candidate acceptance (>=n)
reference_tracks = []
for p in unique_particle_ids:
    num_hits = len(truth.loc[truth.particle_id == p])
    if num_hits >= 4:
        # reference track found
        reference_tracks.append(p)

print("number of reference tracks:", len(reference_tracks))


# now look at the tracks extracted using the GNN algorithm

# loop through each track candidate
track_candidates = []
i = 0
inputDir = "src/trackml_mod/output/iteration_2/candidates/"
subgraph_path = "_subgraph.gpickle"
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    track_candidates.append(sub)
    i += 1
    path = inputDir + str(i) + subgraph_path

# for every node in track candidate
successful_reconstructed_tracks = 0
for track in track_candidates:
    truth_particle = nx.get_node_attributes(track,'truth_particle').values()
    most_common_truth_particle = mode(truth_particle)
    if most_common_truth_particle in reference_tracks:
        # successfully reconstructed track
        successful_reconstructed_tracks += 1

print("number of successfully reconstructed tracks:", successful_reconstructed_tracks)
print("number of reference tracks:", len(reference_tracks))
track_recon_eff = successful_reconstructed_tracks * 100 / len(reference_tracks)
print("track_recon_eff: ", track_recon_eff, "%")