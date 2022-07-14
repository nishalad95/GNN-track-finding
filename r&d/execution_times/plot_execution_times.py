import pandas as pd
import matplotlib.pyplot as plt

execution_times = pd.read_csv('execution_times.txt', delimiter = "\t")
stages = pd.read_csv('stages.txt', delimiter = "\t")

df = pd.concat([execution_times, stages], axis=1, join="inner")
df.drop(df.tail(1).index,inplace=True)
df = df.rename(columns={"0": "times", "start": "stages"})
# print(df)

ax = df.plot.bar(x='stages', y='times', rot=0, figsize=(8,6))
ax.set_ylabel("execution time (s)")
ax.set_xticklabels(df.stages, rotation=0)
ax.tick_params(axis='x', labelsize=6)
plt.savefig("execution_times.png", dpi=300)
plt.show()