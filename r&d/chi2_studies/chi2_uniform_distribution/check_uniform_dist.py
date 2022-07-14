import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

fig, ax = plt.subplots()

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 0.1))

my_file = open("src/output/iteration_1/candidates/pvals.csv", "r")
x = my_file.read()
x = x.split("\n")

x_floats = []
for val in x[:-2]:
    x_floats.append(float(val))


# my_file = open("src/output/iteration_2/candidates/pvals.csv", "r")
# x = my_file.read()
# x = x.split("\n")

# for val in x[:-2]:
#     x_floats.append(float(val))


plt.xticks(np.arange(0.0, 1.1, 0.1))
ax.hist(x_floats, bins=50, density=False)
plt.xlabel("p-value distribution from chi2 fit for extracted candidates")
plt.ylabel("Frequency")
plt.show()