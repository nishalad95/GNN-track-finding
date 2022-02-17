import csv
import pandas as pd
import numpy as np

# UNITS OF MOMEMTUM: Gev/c
truth = pd.read_csv("src/trackml_mod/truth/event000001000-truth.csv", sep=',')

# get truth particles above pT cut
truth = truth.assign(tpT=lambda row: (np.sqrt(row.tpx**2 + row.tpy**2 + row.tpz**2)))
print("original length of truth", len(truth))

pt_cut = 1.  # units of momentum GeV/c
truth = truth.loc[truth.tpT >= pt_cut]
print(truth)
print("pt_cut:", pt_cut)
print("number of hits associated with pt cut:", len(truth))

unique_particle_ids = truth.particle_id.unique()
print("unique_particle_ids", unique_particle_ids)
print("number of unique particles: ", len(unique_particle_ids))

# check if each particle is associated to >4 hits --> our criterai for a good reference track