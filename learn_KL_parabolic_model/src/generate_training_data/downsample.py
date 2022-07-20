import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def downsample(df, max_size):
    seed = int(np.random.uniform() * 100)
    if len(df) > max_size:
        frac = max_size / (len(df))
        df = df.sample(frac=frac, replace=True, random_state=seed)
    return df

filename = "output/track_sim/sigma4.0/1000_events_training_data.csv"
df = pd.read_csv(filename)


df = downsample(df, 50000)
correct_pairs = df.loc[df['truth'] == 1]
incorrect_pairs = df.loc[df['truth'] == 0]
incorrect_pairs = downsample(incorrect_pairs, len(correct_pairs))
print("len correct", len(correct_pairs))
print("len incorrect", len(incorrect_pairs))

df.to_csv('output/track_sim/sigma4.0/1000_events_training_data_downsampled.csv', index=False)

# Pairwise KL distance vs Empirical variance of edge orientation
fig = plt.figure(figsize=(10,7))
plt.scatter(incorrect_pairs['emp_var'], incorrect_pairs['kl_dist'], marker='o', label="0")
plt.scatter(correct_pairs['emp_var'], correct_pairs['kl_dist'], marker='x', label="1")
plt.legend(loc="best")
plt.ylabel("Pairwise KL distance")
plt.xlabel("Empirical variance of edge orientation")
plt.title("MC Truth for Pairwise Edge Connections")
# plt.xlim(0,1.6)
plt.ylim(0,200)
plt.savefig("output/training_data_KL_downsampled.png", dpi=300)
plt.show()