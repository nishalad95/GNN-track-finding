import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

filename = "output/track_sim/sigma1.0/10000_events_training_data.csv"
df = pd.read_csv(filename)

correct_pairs = df.loc[df['truth'] == 1]
incorrect_pairs = df.loc[df['truth'] == 0]


# Pairwise KL distance vs Empirical variance of edge orientation
fig = plt.figure(figsize=(10,7))
plt.scatter(incorrect_pairs['emp_var'], incorrect_pairs['kl_dist'], marker='o', label="0")
plt.scatter(correct_pairs['emp_var'], correct_pairs['kl_dist'], marker='x', label="1")
plt.legend(loc="best")
plt.ylabel("Pairwise KL distance")
plt.xlabel("Empirical variance of edge orientation")
plt.title("MC Truth for Pairwise Edge Connections")
plt.xlim(0,1.2)
plt.ylim(0,100)
plt.show()


# Pairwise KL distance vs Degree of Node
fig = plt.figure(figsize=(10,7))
plt.scatter(incorrect_pairs['degree'], incorrect_pairs['kl_dist'], marker='o', label="0")
plt.scatter(correct_pairs['degree'], correct_pairs['kl_dist'], marker='x', label="1")
plt.legend(loc="best")
plt.ylabel("Pairwise KL distance")
plt.xlabel("Degree of Node")
plt.title("MC Truth for Pairwise Edge Connections")
plt.ylim(0,100)
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