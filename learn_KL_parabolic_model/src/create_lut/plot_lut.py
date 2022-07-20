import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_lut_from_file(filename):
    
    lut = []
    lut_file = open(filename, "r")
    for line in lut_file.readlines():

        elements = line.split(" ")
        bin_num = elements[0]
        bin_min = elements[1]
        bin_max = elements[2].split("\n")[0]
    
        lut.append([int(bin_num), int(bin_min), int(bin_max)])
    
    print(lut)
    # LUT dimension variables: degree of node
    df = pd.DataFrame(index=np.arange(0,100,1).tolist(), columns=np.arange(0,15,1).tolist())
    # LUT dimension variables: empirical variance of edge orientation for a given node
    # df = pd.DataFrame(index=np.arange(0,100,1).tolist(), columns=np.arange(0,28,1).tolist())

    # create dataframe 'matrix' of bins equivalent to lut
    for i in range(len(lut)):        
        df.loc[lut[i][1]:lut[i][2], lut[i][0]] = 1

    df = df.fillna(0)
    df = df.iloc[::-1]
    print(df)

    # Plot frequency dataframe with seaborn heatmap
    flatui = ["#3d77d4", "#f0b05d"]
    fig, ax = plt.subplots(figsize=(15,9))
    p = sns.heatmap(df, linewidths=0.1, annot=False, cbar=True, 
                    ax=ax, cmap=sns.color_palette(flatui), 
                    cbar_kws={'orientation': 'vertical'})

    # Manually specify colorbar labelling after it's been generated
    colorbar = p.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])

    plt.title('2D Look-Up Table')
    # plt.xlabel('degree of node')
    plt.xlabel('binned empirical variance of edge orientation')
    plt.ylabel('binned pairwise KL distance')
    plt.show()


# plot_lut_from_file("output/kl_dist_degree/kl_degree.lut")
plot_lut_from_file("learn_KL_linear_model/output/empvar/empvar_relaxed.lut")