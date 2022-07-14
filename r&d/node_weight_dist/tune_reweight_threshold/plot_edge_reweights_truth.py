import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

df = pd.read_csv('remaining_edge_reweights.csv', sep=',')
print(df)

truth1 = df['reweight'].loc[df['truth'] == 1]
truth0 = df['reweight'].loc[df['truth'] == 0]
print("len(truth1)", len(truth1))
print("len(truth0)", len(truth0))

sns.distplot(truth0, label='0', kde=False)
sns.distplot(truth1, label='1', kde=False)
plt.xlabel('reweight')
plt.ylabel('frequency')
plt.legend()
plt.show()