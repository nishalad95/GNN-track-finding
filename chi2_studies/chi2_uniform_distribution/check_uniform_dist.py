import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

fig, ax = plt.subplots()
# ax.xaxis.set_major_formatter(FormatStrFormatter('%10.0f'))

start, end = ax.get_xlim()
# print(start, end)
ax.xaxis.set_ticks(np.arange(start, end, 0.1))

# my_file = open("pvals.txt", "r")
my_file = open("src/trackml_mod/output/iteration_1/candidates/pvals.csv", "r")
x = my_file.read()
x = x.split("\n")

x_floats = []
for val in x[:-2]:
    x_floats.append(float(val))


my_file = open("src/trackml_mod/output/iteration_2/candidates/pvals.csv", "r")
x = my_file.read()
x = x.split("\n")

for val in x[:-2]:
    x_floats.append(float(val))

# print(x_floats)
# print(len(x_floats))
plt.xticks(np.arange(0.0, 1.1, 0.1))
ax.hist(x_floats, bins=50)
plt.xlabel("p-value of chi2 fit")
plt.show()