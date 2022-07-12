import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_before = pd.read_csv('parabolic_param_a_before_cca.txt', sep="\n")
data_before.columns = ["a"]
data_before['a'] = data_before.apply(lambda a: np.abs(a))
data_after = pd.read_csv('parabolic_param_a_after_cca.txt', sep="\n")
data_after.columns = ["a"]
data_after['a'] = data_after.apply(lambda a: np.abs(a))

print(data_before)
print(data_after)

plt.figure(figsize=(8,6))
plt.hist(data_before, bins=50, density=False, histtype='step', label='before CCA', align="left", rwidth = .6)
# plt.hist(data_after, bins=50, density=False, histtype='step', label='after CCA', align="left", rwidth = .6, alpha=0.5)
plt.ylabel('Frequency')
plt.xlabel('a')
# plt.legend(loc='best')
plt.show()