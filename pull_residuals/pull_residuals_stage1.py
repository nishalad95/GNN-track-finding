import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import random
import argparse
from utilities import helper as h
import pprint
import os
import glob
import time
import random
from scipy import stats
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


def plot_pull_residual(pull_data, labels, truth, axis_lims, bins, bw):
    sns.set()
    for p, l, a, b, w in zip(pull_data, labels, axis_lims, bins, bw):
        df = pd.DataFrame({l: p, 'truth': truth})
        correct = df.loc[df['truth'] == 1]
        incorrect = df.loc[df['truth'] == 0]

        # plot histogram
        fig, ax1 = plt.subplots(figsize=(8, 6))
        sns.distplot(correct[l], hist=True, kde=False, bins=b, ax=ax1, color="orange", norm_hist=False,
                            hist_kws={"lw":2, "label": "truth 1"})
        ax2 = ax1.twinx()
        sns.distplot(correct[l], hist=False, kde=True, ax=ax2, color="orange", 
                            kde_kws={"lw": 2, "label": "KDE bw: " + str(w), "clip": a, "bw_adjust": w})
        
        ax1.set_ylabel("Frequency")
        ax2.set_ylabel("PDF")
        ax1.grid(False)
        ax2.grid(False)
        plt.xlim(a)

        # combine legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc=2)

        # FWHM
        kde_curve = ax2.lines[0]
        x = kde_curve.get_xdata()
        y = kde_curve.get_ydata()
        halfmax = y.max() / 2
        maxpos = y.argmax()
        leftpos = (np.abs(y[:maxpos] - halfmax)).argmin()
        rightpos = (np.abs(y[maxpos:] - halfmax)).argmin() + maxpos
        fullwidthathalfmax = x[rightpos] - x[leftpos]
        ax2.hlines(halfmax, x[leftpos], x[rightpos], color='purple', ls=':')
        ax2.text(x[maxpos], halfmax, f'{fullwidthathalfmax:.3f}\n', color='purple', ha='center', va='center')

        plt.title('Stage 1: Pull residual for parameter ' + l)
        plt.show()



def main():

    parser = argparse.ArgumentParser(description='Convert trackml csv to GNN')
    parser.add_argument('-i', '--inputDir', help="Full directory path of graph networks")
    args = parser.parse_args()
    inputDir = args.inputDir

    # read in subgraph data
    subGraphs = []
    i = 0
    subgraph_path = "_subgraph.gpickle"
    path = inputDir + str(i) + subgraph_path
    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        subGraphs.append(sub)
        i += 1
        path = inputDir + str(i) + subgraph_path

    # extract the state and covariance metadata
    truth = []
    pull_a, pull_b = [], []
    pull_c, pull_tau = [], []
    pull_theta_1, pull_theta_2 = [], []
    for i, subGraph in enumerate(subGraphs):
        num_nodes = len(subGraph.nodes())

        if num_nodes > 1:
            for node in subGraph.nodes(data=True):
                node_num = node[0]
                node_attr = node[1]

                track_state_estimates = node_attr["track_state_estimates"]
                node_keys = list(track_state_estimates.keys())
                for j in range(len(track_state_estimates) - 1):
                    for k in range(j):
                        if j == k: continue

                        n1, n2 = node_keys[j], node_keys[k]
                        sv1 = track_state_estimates[n1]["edge_state_vector"]
                        cov1 = track_state_estimates[n1]["edge_covariance"]
                        sv2 = track_state_estimates[n2]["edge_state_vector"]
                        cov2 = track_state_estimates[n2]["edge_covariance"]

                        jv1 = np.array(track_state_estimates[n1]["joint_vector"])
                        jcov1 = track_state_estimates[n1]["joint_vector_covariance"]
                        jv2 = np.array(track_state_estimates[n2]["joint_vector"])
                        jcov2 = track_state_estimates[n2]["joint_vector_covariance"]

                        theta_1 = track_state_estimates[n1]["theta"]
                        theta2_1 = track_state_estimates[n1]["theta2"]
                        var_theta_1 = track_state_estimates[n1]["variance_theta"]

                        theta_2 = track_state_estimates[n2]["theta"]
                        theta2_2 = track_state_estimates[n2]["theta2"]
                        var_theta_2 = track_state_estimates[n2]["variance_theta"]

                        # calculate the pull residuals
                        diffs = sv1 - sv2
                        sum_cov = cov1 + cov2
                        jdiffs = jv1 - jv2
                        jsum_cov = jcov1 + jcov2

                        pa = diffs[0] / np.sqrt(sum_cov[0][0])
                        pb = diffs[1] / np.sqrt(sum_cov[1][1])
                        pc = diffs[2] / np.sqrt(sum_cov[2][2])
                        ptau = jdiffs[2] / np.sqrt(jsum_cov[2][2])

                        pull_a.append(pa)
                        pull_b.append(pb)
                        pull_c.append(pc)
                        pull_tau.append(ptau)

                        diff_theta_1 = theta_1 - theta_2
                        diff_theta_2 = theta2_1 - theta2_2
                        sum_cov_theta = var_theta_1 + var_theta_2
                        ptheta_1 = diff_theta_1 / np.sqrt(sum_cov_theta)
                        ptheta_2 = diff_theta_2 / np.sqrt(sum_cov_theta)
                        pull_theta_1.append(ptheta_1)
                        pull_theta_2.append(ptheta_2)

                        # store MC truth
                        t1 = subGraph.nodes[node_num]["truth_particle"]
                        t2 = subGraph.nodes[n1]["truth_particle"]
                        t3 = subGraph.nodes[n2]["truth_particle"]
                        if (t1 == t2) and (t2 == t3) and (t1 == t3): truth.append(1)
                        else: truth.append(0)


    # # pull residuals for [a, b, tau]
    # pull_data = [np.array(pull_a), np.array(pull_b), np.array(pull_tau)]
    # labels = ["a", "b", "tau"]
    # axis_lims = [(-0.25, 0.25), (-0.5, 0.5), (-50, 50)]
    # bins = [5000, 5000, 10000]
    # bw = [0.05, 0.05, 0.001]
    # plot_pull_residual(pull_data, labels, truth, axis_lims, bins, bw)

    # # pull residuals for theta
    pull_data = [np.array(pull_theta_1), np.array(pull_theta_2)]
    labels = ["theta1", "theta2"]
    axis_lims = [(-1000, 1000), (-1000, 1000)]
    bins = [5000, 10000]
    bw = [0.01, 0.01]
    plot_pull_residual(pull_data, labels, truth, axis_lims, bins, bw)


if __name__ == "__main__":    
    main()