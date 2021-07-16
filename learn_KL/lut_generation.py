import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("output/kl_dist_degree/predictions_kl_degree_svm_poly3_c0.1_gamma0.1.csv")
# df = pd.read_csv("output/kl_dist_vs_emp_var/predictions_kl_empvar_svm_poly3_c0.1_gamma0.1.csv")
sns_plot = sns.lmplot('degree', 'kl_dist', data=df, hue='predictions', fit_reg=False, height=8, aspect=1.5)
ax = plt.gca()
plt.show()

# LUT dimension variables: degree of node
m_c = np.ndarray((4,),float)
m_w = 15       # width of LUT
m_h =  100      # height of LUT
m_c[0] = 2   # min of x
m_c[1] = 14   # max of x
m_c[2] = 0   # min of y
m_c[3] = 100   # max of y

# # LUT dimension variables: empirical variance of edge orientation for a given node
# m_c = np.ndarray((4,),float)
# m_w = 28       # width of LUT
# m_h =  100      # height of LUT
# m_c[0] = 0   # min of x
# m_c[1] = 1.4   # max of x
# m_c[2] = 0   # min of y
# m_c[3] = 100   # max of y


bin_inc_x = m_c[1] / m_w
bin_inc_y = m_c[3] / m_h

lut = []
degree_bins = np.arange(0.0, m_c[1], bin_inc_x)
print("degree bins", degree_bins)

kl_dist_bins = np.arange(0.0, m_c[3], bin_inc_y)
print("kl bins", kl_dist_bins)

lut_2 = []
for i in range(len(degree_bins)-1):
    # calculate min and max tau
    df_band = df.loc[(df['degree'] > degree_bins[i]) & (df['degree'] <= degree_bins[i+1])
                            & (df.predictions == 1)]
    
    if len(df_band) != 0:

        band_min = df_band['kl_dist'].min()
        band_max = df_band['kl_dist'].max()

        # convert into bin representation
        bin_min = int(round(band_min / bin_inc_y))
        bin_max = int(round(band_max / bin_inc_y))

        # append to LUT
        lut_2.append([i, bin_min, bin_max])


# degree of node
df_lut = pd.DataFrame(index=np.arange(0,m_h,1).tolist(), columns=np.arange(2,m_w,1).tolist())
# emp var
# df_lut = pd.DataFrame(index=np.arange(0,m_h,1).tolist(), columns=np.arange(0,m_w,1).tolist())

print("DF LUT")
print(df_lut)

# create dataframe 'matrix' of bins equivalent to lut
for i in range(len(lut_2)):
    df_lut[lut_2[i][0]][lut_2[i][1] : lut_2[i][2] + 1] = 1
    
df_lut = df_lut.fillna(0)
lut_2_df = df_lut.iloc[::-1]


# Plot frequency dataframe with seaborn heatmap
flatui = ["#3d77d4", "#f0b05d"]
fig, ax = plt.subplots(figsize=(18,10))
p = sns.heatmap(lut_2_df, linewidths=0.1, annot=False, cbar=True, 
                ax=ax, cmap=sns.color_palette(flatui), 
                cbar_kws={'orientation': 'vertical'})

# Manually specify colorbar labelling after it's been generated
colorbar = p.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['0', '1'])

plt.title('2D Look-Up Table')
plt.xlabel('degree of node')
# plt.xlabel('binned empirical variance of edge orientation')
plt.ylabel('binned pairwise KL distance')
plt.show()


# save LUT to file
lut_2_df.sort_index(inplace=True)
print("\n\nLUT:\n\n")
print(lut_2_df)

def get_lut(x):
    kl_dist = x.name
    x = x.to_list()
    nonzero_id = np.nonzero(x)[0]

    if len(nonzero_id) == 0:
        return [kl_dist, 0, 0]
    elif len(nonzero_id) == 1:
        return [kl_dist, nonzero_id[0], nonzero_id[0] + 1]
    else:
        return [kl_dist, nonzero_id[0], nonzero_id[-1] + 1]

lut_to_save = []
lut_2_df.apply(lambda x: lut_to_save.append(get_lut(x)), axis=0)

print("LUT to save:\n", lut_to_save)

with open("output/kl_dist_degree/kl_degree.lut", "w") as outfile:
# with open("output/kl_dist_vs_emp_var/kl_empvar.lut", "w") as outfile:
    for row in lut_to_save:
        lut_row = str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n"
        outfile.write(lut_row)