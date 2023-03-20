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



#Calculating the Gaussian PDF values given Gaussian parameters and random variable X
def gaus(X,C,X_mean,sigma):
    return C*exp(-(X-X_mean)**2/(2*sigma**2))


def plot_pull_residual(pull_data, labels, truth, axis_lims, bins):
    for p, l, a, b in zip(pull_data, labels, axis_lims, bins):
        df = pd.DataFrame({l: p, 'truth': truth})
        correct = df.loc[df['truth'] == 1]
        incorrect = df.loc[df['truth'] == 0]

        # creating histogram from data
        correct_numpy=correct.to_numpy(dtype ='float32')
        x_data=correct_numpy[:,0]
        hist, bin_edges = np.histogram(x_data)
        hist=hist/sum(hist)
        n = len(hist)
        x_hist=np.zeros((n),dtype=float) 
        for ii in range(n):
            x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
        
        # compute the Gaussian least-square fitting process
        y_hist=hist
        mean = sum(x_hist*y_hist)/sum(y_hist)                  
        sigma = sum(y_hist*(x_hist-mean)**2)/sum(y_hist) 
        param_optimised,param_covariance_matrix = curve_fit(gaus,x_hist,y_hist,p0=[max(y_hist),mean,sigma],maxfev=5000)

        #print fit Gaussian parameters
        print("Fit parameters: ")
        print("=====================================================")
        print("C = ", param_optimised[0], "+-",np.sqrt(param_covariance_matrix[0,0]))
        print("X_mean =", param_optimised[1], "+-",np.sqrt(param_covariance_matrix[1,1]))
        print("sigma = ", param_optimised[2], "+-",np.sqrt(param_covariance_matrix[2,2]))
        print("\n")

        # plot histogram
        fig, ax1 = plt.subplots(figsize=(8, 6))
        plt.hist(x_data, bins=b, color="orange", label="truth 1", histtype='bar')
        plt.xlabel("Data: " + l)
        plt.ylabel("Frequency")

        # plot Gaussian fit 
        ax2 = ax1.twinx()
        x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),10000)
        ax2.plot(x_hist_2,gaus(x_hist_2,*param_optimised),'r:',label='Gaussian fit')
        ax2.set_ylim(ymin=0)

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

        # combine legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc=2)
        
        # ax1.set_xlim((a[0], a[1]))
        plt.ylabel("PDF")
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

                        # store MC truth
                        t1 = subGraph.nodes[node_num]["truth_particle"]
                        t2 = subGraph.nodes[n1]["truth_particle"]
                        t3 = subGraph.nodes[n2]["truth_particle"]
                        if (t1 == t2) and (t2 == t3) and (t1 == t3): truth.append(1)
                        else: truth.append(0)



    pull_data = [np.array(pull_a), np.array(pull_b), np.array(pull_tau)]
    labels = ["a", "b", "tau"]
    axis_lims = [(-0.25, 0.25), (-0.5, 0.5), (-50, 50)]
    bins = [5000, 5000, 500]
    # fit_params = [[1, mean, sigma], [1, mean, sigma], [1, mean, sigma]]
    plot_pull_residual(pull_data, labels, truth, axis_lims, bins)


if __name__ == "__main__":    
    main()