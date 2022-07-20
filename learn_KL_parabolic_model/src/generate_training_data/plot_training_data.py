import csv
import pandas as pd
import matplotlib.pyplot as plt

# read in csv data
df = pd.read_csv('output/track_sim_trackml_parabolic_model/minCurv_0.3_134/event_graph_data/1_events_training_data.csv')
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
plt.xlim(0,2.5)
plt.ylim(0,100)
# plt.savefig("output/training_data_KL.png", dpi=300)
plt.show()


# plot pairwise KL distance distribution as a 1d plot, for variance <= 1.0
df = df.loc[df['emp_var'] <= 1.0]
print(df)
correct_pairs = df.loc[df['truth'] == 1]
incorrect_pairs = df.loc[df['truth'] == 0]
print("len correct", len(correct_pairs))
print("len incorrect", len(incorrect_pairs))

fig = plt.figure(figsize=(10,7))
plt.hist(incorrect_pairs['kl_dist'], label="0", density=False, bins=1000)
plt.hist(correct_pairs['kl_dist'], label="1", density=False, bins=100)
plt.legend(loc="best")
plt.xlabel("Pairwise KL distance")
plt.ylabel("Frequency")
plt.title("MC Truth for Pairwise Edge Connections for node variance of edge orientation <= 1.0")
# plt.xlim(0,2.0)
plt.xlim(0,50)
# plt.savefig("output/training_data_KL.png", dpi=300)
plt.show()
