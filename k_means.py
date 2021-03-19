from math import *
from random import *
import csv
import os
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



def cluster(coords, iters=10, num_clusters=1):
    '''
    Executes the k-means clustering algorithm on a set of specified data points
    & outputs each cluster centre coordinates

    Params:
    --------
        coords        (list)    : list of floats containing the x data
        iters         (int)     : number of iterations to perform the algorithm
        num_clusters  (int)     : number of clusters

    Returns:
    --------
        cluster_labels (list)   : maps each coordinate in the dataset to a cluster index
    '''
    coords = np.asarray(coords)
    num_points = len(coords)

    # random points initialized to be initial cluster centres
    cluster_centre = np.array([coords[np.random.choice(num_points)]
                               for i in range(num_clusters)])
    cluster_labels = np.zeros(len(coords))

    n = 0
    while n < iters:

        # coords = coords.transpose()
        distances = np.empty((0, num_points))

        for i in range(num_clusters):
            distances = np.concatenate((distances, [np.sqrt(
                (coords[0] - cluster_centre[i][0])**2 + (coords[1] - cluster_centre[i][1])**2)]), axis=0)
                        
        # assign each data point to the nearest centre
        cluster_labels = np.argmin(distances, axis=1)

        coords = coords.transpose()

        for i in range(num_clusters):
            cluster = coords[np.where(
                np.array([cluster_labels == i]).transpose())[0]]
            # update cluster centre
            cluster_size = len(cluster)
            if cluster_size > 0:
                new_centre = np.sum(cluster, axis=0) / cluster_size
                cluster_centre[i] = new_centre
            if (n == iters - 1):
                print(
                    "Cluster {} is centred at ({:.3f}, {:.3f}) and has {} points".format(
                        i, *cluster_centre[i], cluster_size))

        n += 1

    return cluster_labels