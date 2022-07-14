import csv
import numpy as np
import pandas as pd


event_path = "src/trackml_mod/events/event_1_filtered_graph_"
truth_event_path = "src/trackml_mod/truth/event000001000-"

# truth event
hits_particles = pd.read_csv(truth_event_path + "truth.csv")
particles_nhits = pd.read_csv(truth_event_path + "particles.csv")

# nodes to hits
nodes_hits = pd.read_csv(event_path + "nodes_to_hits.csv")
truth = nodes_hits[['node_idx', 'hit_id']]
hit_ids = truth['hit_id']
particle_ids = []
for i, hid in enumerate(hit_ids):
    pid = hits_particles.loc[hits_particles['hit_id'] == hid]['particle_id'].item()
    
    if i == 0: print(pid, type(pid))
    
    pid = int(pid)

    if i == 0: print(pid, type(pid))
    
    particle_ids.append(pid)

    if i == 0: print(particle_ids, type(particle_ids))

truth['particle_id'] = particle_ids

print(truth)

# number of hits for each truth particle
nhits = np.array([])
particle_ids = truth['particle_id']
for pid in particle_ids:
    try:
        n = particles_nhits.loc[particles_nhits['particle_id'] == pid]['nhits'].item()
    except ValueError:
        n = 0
    nhits = np.append(nhits, n)
truth['nhits'] = nhits
