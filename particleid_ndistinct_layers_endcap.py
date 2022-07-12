import pandas as pd

# CURRENTLY only works for endcap volume 7!!

# identify the number of distinct layers per particle id, contained within the endcap volume 7 region 
# and add it to the csv file

# load in the file
df = pd.read_csv('src/trackml_mod/event_truth/event000001000-full-mapping-minCurv-0.3-800.csv')

# identify rows with hits in volume 7
volume7_particle_list = df.loc[df.volume_id == 7].particle_id.to_list()
unique_volume7_particle_list = list(set(volume7_particle_list))

# loop through each particle_id and determine number of distinct layers for its hits in the endcap vol7
for i, p in enumerate(unique_volume7_particle_list):
    grouped_by_particle_id = df.loc[df.particle_id == p]
    num_distinct_layers = grouped_by_particle_id.layer_id.nunique()
    df.loc[df.particle_id == p, 'num_distinct_layers_endcap'] = num_distinct_layers
    
print(df)
df.to_csv('src/trackml_mod/event_truth/event000001000-full-mapping-minCurv-0.3-800.csv')