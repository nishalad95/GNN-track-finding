import matplotlib.pyplot as plt
import pandas as pd

path = "src/output/extracted_"
track_purities = pd.read_csv(path + 'track_purities.csv')
particle_purities = pd.read_csv(path + 'particle_purities.csv')


print("length of track purities: ", len(track_purities))
print("length of particle purities: ", len(particle_purities))

plt.hist(track_purities, bins=30, density=False, histtype='step', label='track purity', align="left", rwidth = .6)
plt.hist(particle_purities, bins=30, density=False, histtype='step', label='particle purity', align="left", rwidth = .6)
plt.ylabel('Frequency')
plt.xlabel('Purity')
plt.xlim([-0.05, 1.1])
plt.legend(loc='best')
plt.savefig("src/output/purity_distributions.png", dpi=300)
# plt.show()