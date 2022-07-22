import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import argparse


def plot_pvals(df, plane, inputDir):
    fig, ax = plt.subplots()
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 0.1))

    key = 'pvals_' + plane
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    ax.hist(df[key], bins=50, density=False)
    plt.xlabel("p-value distribution from chi2 fit in" + plane + "plane for extracted candidates")
    plt.ylabel("Frequency")
    plt.savefig(inputDir + "/p_value_distribution_" + plane + ".png", dpi=300)


parser = argparse.ArgumentParser(description='extract track candidates')
parser.add_argument('-i', '--input', help='input directory path')
args = parser.parse_args()
inputDir = args.input

df = pd.read_csv(inputDir + "/iteration_1/candidates/pvals.csv")

plot_pvals(df, 'xy', inputDir)
plot_pvals(df, 'zr', inputDir)