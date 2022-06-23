import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='extract track candidates')
parser.add_argument('-i', '--input', help='input directory path')
args = parser.parse_args()
inputDir = args.input

path = inputDir + "/extracted_"
track_purities = pd.read_csv(path + 'track_purities.csv', header=None)
particle_purities = pd.read_csv(path + 'particle_purities.csv', header=None)

print("length of track purities: ", len(track_purities))
print("length of particle purities: ", len(particle_purities))

track_purity_mean = track_purities.mean(axis=0)
particle_purity_mean = particle_purities.mean(axis=0)

print("mean of track purities: ", track_purity_mean)
print("mean of particle purities: ", particle_purity_mean)

plt.hist(track_purities, bins=30, density=False, histtype='step', label='track purity', align="left", rwidth = .6)
plt.hist(particle_purities, bins=30, density=False, histtype='step', label='particle purity', align="left", rwidth = .6)
plt.ylabel('Frequency')
plt.xlabel('Purity')
plt.xlim([-0.05, 1.1])
plt.legend(loc='best')
plt.savefig(inputDir + "/purity_distributions.png", dpi=300)
# plt.show()