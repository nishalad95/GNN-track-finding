import csv
import networkx as nx
import pandas as pd


def get_particle_id(row, df):
    particle_ids = []
    for hit_id in row:
        particle_id = df.loc[df.hit_id == hit_id]['particle_id'].item()
        particle_ids.append(particle_id)
    return particle_ids

def create_hit_dissociation(row):
    hit_dissociation = {"hit_ids" : row['hit_id'], "particle_ids" : row['particle_id']}
    return hit_dissociation

truth_event_path = "src/trackml_mod/truth/event000001000-"
truth_event_file = truth_event_path + "nodes-particles-id.csv"
node_hit_particle_truth = pd.read_csv(truth_event_file)
# print("node_hit_particle_truth\n", node_hit_particle_truth)

group = node_hit_particle_truth.groupby('node_idx')
grouped_hit_id = group.apply(lambda row: row['hit_id'].unique())
grouped_hit_id = pd.DataFrame({'node_idx':grouped_hit_id.index, 'hit_id':grouped_hit_id.values})
# print("grouped_hit_id\n", grouped_hit_id)

grouped_hit_id['particle_id'] = grouped_hit_id['hit_id'].apply(lambda row: get_particle_id(row, node_hit_particle_truth))
print("grouped_hit_id\n", grouped_hit_id)


grouped_hit_id['hit_dissociation'] = grouped_hit_id.apply(lambda row: {"hit_id": row['hit_id'], "particle_id":row['particle_id']}, axis=1)
print("grouped_hit_id\n", grouped_hit_id)