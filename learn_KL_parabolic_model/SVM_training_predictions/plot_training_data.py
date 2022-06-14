import csv
import pandas as pd
import matplotlib.pyplot as plt

# read in csv data
df = pd.read_csv('output/track_sim/sigma0.1/2_events_training_data.csv')
print(df)  

# plot empirical variance vs pairwise kl distance
correct_pairs = df.loc[df['truth'] == 1]
incorrect_pairs = df.loc[df['truth'] == 0]
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
# plt.xlim(0,1.6)
plt.ylim(0,1e6)
plt.show()