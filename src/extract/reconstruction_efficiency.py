import csv
import pandas as pd
import numpy as np
from statistics import mode
import os
import networkx as nx
import matplotlib.pyplot as plt


# every hit has an associated particle id: from the truth information file for hits
# full info about particles itself is contained in the particles file

#########################################
# obtain reference tracks from MC truth
#########################################

# identify the ids of particles that have pT > 1GeV, units GeV/c
particles = pd.read_csv("src/trackml_mod/truth/event000001000-particles.csv", sep=',')
particles = particles.assign(pT=lambda row: (np.sqrt(row.px**2 + row.py**2)))
pt_cut = 1.  # units of momentum GeV/c
particles = particles.loc[particles.pT >= pt_cut]
particle_ids = particles.particle_id.to_list()

# go through the truth file for hits & identify hits cooresponding to each particle_id
truth = pd.read_csv("src/trackml_mod/truth/event000001000-truth.csv", sep=',')
hit_ids = truth.loc[truth.particle_id.isin(particle_ids)]
hit_ids = hit_ids.hit_id.to_list()

# extract the subset of hits contained in the volume of interest (i.e. only endcap for now)
hits = pd.read_csv("src/trackml_mod/truth/event000001000-hits.csv", sep=',')
hits['particle_id'] = truth['particle_id']
left_endcap_hits = hits.loc[(hits.hit_id.isin(hit_ids)) & (hits.volume_id == 7)]
# print(left_endcap_hits)
# print("num of left endcap hits > 1 GeV:", len(left_endcap_hits))

# # plot
left_endcap_hits = left_endcap_hits.assign(r=lambda row: (np.sqrt(row.x**2 + row.y**2)))
# x = left_endcap_hits.x.to_list()
# y = left_endcap_hits.y.to_list()
# z = left_endcap_hits.z.to_list()
# r = left_endcap_hits.r.to_list()
# plt.scatter(x, y)
# plt.show()
# plt.scatter(z, r)
# plt.show()


# Compute number of reference tracks: Apply a cut on the num of hits
# require >= 4 hits per track (in the volume of interest) to be a reference track
num_hits = 4
unique_particle_ids = left_endcap_hits.particle_id.unique()
num_reference_tracks = 0
reference_tracks_dict = {}          # key:value particle_id:[hit_ids]
for i, p in enumerate(unique_particle_ids):
    ref_track = left_endcap_hits.loc[left_endcap_hits.particle_id == p]
    if len(ref_track) >= num_hits:
        
        # check the "one-hit-per-module" constraint
        one_hit_per_module = ref_track[ref_track.duplicated(['layer_id','module_id'], keep=False)]
        if len(one_hit_per_module) == 0:
            # good reference track found
            num_reference_tracks += 1
            reference_tracks_dict[p] = ref_track.hit_id.to_list()   # add to dictionary
            # if i < 1:
            # print("ref track:\n", ref_track)
            # print("hit_ids in reference track")
            # print(reference_tracks_dict[p])
            # x = ref_track.x.to_list()
            # y = ref_track.y.to_list()
            # z = ref_track.z.to_list()
            # r = ref_track.r.to_list()
            # plt.scatter(x, y, s=10)
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.show()
            # plt.scatter(z, r, s=10)
            # plt.xlabel("z")
            # plt.ylabel("r")
            # plt.show()
        else:
            # TODO: handle this case
            print(">1 hit per module!! TODO: will need to handle this case!")
            print(one_hit_per_module)

print("num of reference tracks:", num_reference_tracks)
# print("reference tracks:", reference_tracks)


#########################################################
# Look at what tracks have been succesfully reconstructed
#########################################################


# Now consider the tracks extracted using the GNN algorithm
# loop through each track candidate
# TODO
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

# for every node in the track candidate get the hit_ids
for i, track in enumerate(track_candidates):
    hit_ids = nx.get_node_attributes(track, 'truth_hit_id').values()
    hit_ids = np.concatenate(list(hit_ids))
    if i < 2:
        print("hit ids from gnn reconstruction:\n", hit_ids)





# old code
# # loop through each track candidate
# track_candidates = []
# i = 0
# inputDir = "src/trackml_mod/output/iteration_2/candidates/"
# subgraph_path = "_subgraph.gpickle"
# path = inputDir + str(i) + subgraph_path
# while os.path.isfile(path):
#     sub = nx.read_gpickle(path)
#     track_candidates.append(sub)
#     i += 1
#     path = inputDir + str(i) + subgraph_path

# # for every node in track candidate
# successful_reconstructed_tracks = 0
# for track in track_candidates:
#     truth_particle = nx.get_node_attributes(track,'truth_particle').values()
#     most_common_truth_particle = mode(truth_particle)
#     if most_common_truth_particle in reference_tracks:
#         # successfully reconstructed track
#         successful_reconstructed_tracks += 1

# print("number of successfully reconstructed tracks:", successful_reconstructed_tracks)
# print("number of reference tracks:", len(reference_tracks))
# track_recon_eff = successful_reconstructed_tracks * 100 / len(reference_tracks)
# print("track_recon_eff: ", track_recon_eff, "%")