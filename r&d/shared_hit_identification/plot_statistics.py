import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

data = pd.read_csv('statistics_with_truth_1particle/iteration1.txt', sep=" ", header=None)
# data = pd.read_csv('statistics_with_truth_1particle/iteration1.txt', sep=" ", header=None)
print(data)
print("minimum distance:")
print(data.min())
# fig = plt.subplots()
# data.plot.hist(bins=100, ax=ax)
ax = data.plot(kind='hist', bins=100)
kde = data.plot(kind='kde', ax=ax, secondary_y=True)
ax.set_xlim([-0.1, 2.0])
ax.set_xlabel("Distance")
ax.set_title("Maximum distance determined from agglomerative clustering")
x = kde.get_children()[0]._x
y = kde.get_children()[0]._y

y = list(y)
max_value = max(y)
max_index = y.index(max_value)
distance_threshold = x[max_index]

print("Threshold value:")
print(distance_threshold)

plt.show()
