import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv("chi2_data_sigma_0.1.csv", sep=" ")
df2 = pd.read_csv("chi2_data_sigma_0.2.csv", sep=" ")
df3 = pd.read_csv("chi2_data_sigma_0.3.csv", sep=" ")
df4 = pd.read_csv("chi2_data_sigma_0.4.csv", sep=" ")
df5 = pd.read_csv("chi2_data_sigma_0.5.csv", sep=" ")


dfs = [df1, df2, df3, df4, df5]
sigma = [0.1, 0.2, 0.3, 0.4, 0.5]
iteration_cuts = []

final_data = pd.DataFrame(columns=['truth', 'distance', 'cut', 'sigma'])
for i, df in enumerate(dfs):

    df["sigma"] = sigma[i]
    iteration_cuts = df["cut"].unique()
    iteration_cuts = sorted(iteration_cuts, reverse=True)
    print("Sigma 0.", i, ":")
    print(iteration_cuts)

    # data = df.loc[df["cut"] == iteration_cuts[0]]
    data = df
    final_data = final_data.append(data)


print(final_data)

mc_truth_1 = final_data #.loc[final_data["truth"] == 1]
sns.scatterplot(data=mc_truth_1, x="sigma", y="distance", hue="truth", style="truth")
plt.ylim(0, 30)
plt.show()

for i in range(len(sigma)):
    histo1 = mc_truth_1.loc[mc_truth_1["sigma"] == sigma[i]]
    print(histo1)
    sns.histplot(data=histo1, x="distance", hue="truth")
    plt.xlim([0,10])
    plt.show()


# sns.scatterplot(data=final_data, x="sigma", y="distance", hue="truth", style="truth")
# plt.ylim(0, 15)
# plt.show()

# histo1 = final_data.loc[final_data["cut"] == 10.0]
# print(histo1)
# sns.histplot(data=histo1, x="distance", hue="truth")
# plt.show()



# def original_accepted(distance, cut):
#     if distance < cut: return 1
#     else : return 0

# df['original_accepted'] = df.apply(lambda row : 
#                             original_accepted(row['distance'], row['cut']), 
#                             axis = 1)

# df['new_accepted'] = df.apply(lambda row : 
#                             original_accepted(row['distance'], row['e_measurement']), 
#                             axis = 1)
# print(df)


# df_truth_1 = df.loc[df['truth'] == 1]
# len_truth_1 = len(df_truth_1)

# orig_accep_1 = df_truth_1.loc[df_truth_1['original_accepted'] == 1]
# len_orig_accep_1 = len(orig_accep_1)
# perc_1 = (len_orig_accep_1 * 100) / len_truth_1
# print("original %", perc_1)


# new_accep_1 = df_truth_1.loc[df_truth_1['new_accepted'] == 1]
# len_new_accep_1 = len(new_accep_1)
# perc_2= (len_new_accep_1 * 100) / len_truth_1
# print("new %", perc_2)