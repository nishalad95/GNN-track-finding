import pandas as pd
import numpy as np

# identify the number of hits contained the left endcap region 7 and add it to the csv file

# identify all the hits in region 7 (left endcap)
df_hits = pd.read_csv('src/trackml_mod/event_truth/event000001000-hits.csv')
volume7_hits = df_hits.loc[df_hits.volume_id == 7]
volume7_hits = volume7_hits.hit_id.to_list()
# print(volume7_hits)

# add the hits which are in volume 7 as a column in the main dataframe used in recon. eff calculations
df_nodes_particles_ids = pd.read_csv('src/trackml_mod/event_truth/event000001000-full-mapping-minCurv-0.3-800.csv')
df_nodes_particles_ids["endcap_volume7_binary"] = 0
for hit in volume7_hits:
    df_nodes_particles_ids.loc[df_nodes_particles_ids.hit_id == hit, 'endcap_volume7_binary'] = 1
# print(df_nodes_particles_ids)

# get the particle_ids that are in volume 7 according to the df above
particles_ids_endcap_volume7 = df_nodes_particles_ids.loc[df_nodes_particles_ids.endcap_volume7_binary == 1].particle_id.to_list()
# print(particles_ids_endcap_volume7)

df_nodes_particles_ids["nhits_endcap_volume7"] = 0
for p in particles_ids_endcap_volume7:
    particle_id_df = df_nodes_particles_ids.loc[df_nodes_particles_ids.particle_id == p]
    total_volume7_hits = particle_id_df['endcap_volume7_binary'].sum()
    # print("total", total_volume7_hits)
    df_nodes_particles_ids.loc[df_nodes_particles_ids.particle_id == p, "nhits_endcap_volume7"] = total_volume7_hits

print(df_nodes_particles_ids)
df_nodes_particles_ids.to_csv('src/trackml_mod/event_truth/event000001000-full-mapping-minCurv-0.3-800.csv', index=False)