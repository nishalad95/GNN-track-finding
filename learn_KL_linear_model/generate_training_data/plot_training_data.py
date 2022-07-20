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

filename = "output/track_sim/sigma0.05/10000_events_training_data.csv"
df = pd.read_csv(filename)

df = downsample(df, 100000)
correct_pairs = df.loc[df['truth'] == 1]
incorrect_pairs = df.loc[df['truth'] == 0]
incorrect_pairs = downsample(incorrect_pairs, len(correct_pairs))

print("len correct", len(correct_pairs))
print("len incorrect", len(incorrect_pairs))


# Pairwise KL distance vs Empirical variance of edge orientation
fig = plt.figure(figsize=(10,7))
plt.scatter(incorrect_pairs['emp_var'], incorrect_pairs['kl_dist'], marker='o', label="0")
plt.scatter(correct_pairs['emp_var'], correct_pairs['kl_dist'], marker='x', label="1")
plt.legend(loc="best")
plt.ylabel("Pairwise KL distance")
plt.xlabel("Empirical variance of edge orientation")
plt.title("MC Truth for Pairwise Edge Connections")
plt.xlim(0,1.6)
plt.ylim(0,300)
plt.show()


# Pairwise KL distance vs Degree of Node
fig = plt.figure(figsize=(10,7))
plt.scatter(incorrect_pairs['degree'], incorrect_pairs['kl_dist'], marker='o', label="0")
plt.scatter(correct_pairs['degree'], correct_pairs['kl_dist'], marker='x', label="1")
plt.legend(loc="best")
plt.ylabel("Pairwise KL distance")
plt.xlabel("Degree of Node")
plt.title("MC Truth for Pairwise Edge Connections")
plt.ylim(0,300)
plt.show()


# subplots of 1D histograms
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12,8))
i = 2
for row in ax:
    for col in row:
        degree = df.loc[df['degree'] == i]
        inc_degree = degree.loc[degree['truth'] == 0]['kl_dist']
        cor_degree = degree.loc[degree['truth'] == 1]['kl_dist']
        col.hist(inc_degree, bins=100, label="0")
        col.hist(cor_degree, bins=100, label="1")
        col.legend(loc="best")
        col.set_xlim(0,100)
        col.set_title("Node degree: " + str(i))
        i += 1

for a in ax.flat:
    a.set(xlabel='Pairwise KL distance', ylabel='Frequency')

fig.subplots_adjust(hspace=.5, wspace=.5)
plt.show()