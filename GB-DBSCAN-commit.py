import time
import matplotlib.pyplot as plt
import numpy as np
import pyflann
from scipy import io
from scipy.spatial import distance
from munkres import Munkres
import random
import scipy
from collections import Counter
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform


def calculate_center_and_radius(gb):
    """
    Calculate the center and radius of the granular ball.
    params:
    - gb: the granular ball
    returns:
    - radius：the radius of the granular ball
    - center：the center of the granular ball
    """
    dataGb = line[gb]
    center = dataGb.mean(axis=0)
    radius = np.max((((dataGb - center) ** 2).sum(axis=1) ** 0.5))
    return radius, center


def getDensity(Radii, gbNum):
    """
    Calculate the density of granular balls.
    params:
    - Radii: the Radii of granular balls
    - gbNum: the number of granular balls
    returns:
    - density: the densities of granular balls
    """
    density = np.zeros(shape=gbNum)
    for i in range(0, gbNum):
        density[i] = Radii[i]
    return density


def Select_CB_and_NCB(Density, density_threshold):
    """
    Partitioning granular balls into Core-GBs and Non-Core-GBs based on the density threshold.
     params:
    - Density: the densities of granular balls
    - density_threshold: the density threshold
    returns:
    - CBS: Core-GBs
    - NCBS: Non-Core-GBs
    """
    CBS = np.where(Density <= density_threshold)[0]
    NCBS = np.where(Density > density_threshold)[0]
    return CBS, NCBS


def cluster_CBS(CBS, Centers, cbs_radius):
    """
    Perform clustering on Core-GBs and merge them into corresponding clusters.
    params:
    - CBS: the Core-GBs
    - Centers: the centers of granular balls
    - cbs_radius: the radii of the Core-GBs
    returns:
    - cluster: the result after clustering the Core-GBs
    """
    Centers = np.array(Centers)
    CBS_N = len(CBS)
    unvisited = [i for i in range(CBS_N)]
    cluster = [-1 for i in range(CBS_N)]
    K = -1
    while len(unvisited) > 0:
        p = unvisited[0]
        unvisited.remove(p)
        neighbors = []
        for i in range(CBS_N):
            if i != p:
                dis = ((Centers[CBS[i]] - Centers[CBS[p]]) ** 2).sum(axis=0) ** 0.5
                if dis <= (cbs_radius[i] + cbs_radius[p]):
                    neighbors.append(i)
        K = K + 1
        cluster[p] = K
        for pi in neighbors:
            if pi in unvisited:
                unvisited.remove(pi)
                neighbors_pi = []
                for j in range(CBS_N):
                    if j != pi:
                        dis_pi = ((Centers[CBS[j]] - Centers[CBS[pi]]) ** 2).sum(axis=0) ** 0.5
                        if dis_pi <= (cbs_radius[j] + cbs_radius[pi]):
                            neighbors_pi.append(j)

                for t in neighbors_pi:
                    if t not in neighbors:
                        neighbors.append(t)
            if cluster[pi] == -1:
                cluster[pi] = K
    return cluster


def cluster_NCBS(NCBS, point_cluster, gb_list, CBS_Centers):
    """
    Assigning Non-Core-GBs to their respective clusters.
    params:
    - NCBS: the Non-Core-GBs
    - point_cluster: the result before processing Non-Core-GBs
    - gb_list: the granular balls
    - CBS_Centers: the centers of the Core-GBs
    returns:
    - point_cluster: The result after processing Non-Core-GBs, i.e., the final result of the clustering algorithm
    """
    NCBS_N = len(NCBS)
    for i in range(NCBS_N):
        varity = []
        var_cluster = []
        No_cluster = []
        for j in range(gb_list[NCBS[i]].shape[0]):
            if point_cluster[gb_list[NCBS[i]][j]] != -1:
                varity.append(gb_list[NCBS[i]][j])
                var_cluster.append(point_cluster[gb_list[NCBS[i]][j]])
            if point_cluster[gb_list[NCBS[i]][j]] == -1:
                No_cluster.append(gb_list[NCBS[i]][j])
        if len(varity) != 0:
            c = distance.cdist(line[No_cluster], line[varity])
            near_point = np.argmin(c, axis=1)
            for i in range(len(near_point)):
                point_cluster[No_cluster[i]] = point_cluster[varity[near_point[i]]]
        else:
            center_ncb = line[gb_list[NCBS[i]]].mean(axis=0)
            c2 = distance.cdist([center_ncb], CBS_Centers)
            near_cb = np.argmin(c2)
            point_cluster[gb_list[NCBS[i]]] = point_cluster[gb_list[CBS[near_cb]]]

    return point_cluster


def draw_all_point(point_cluster, line):
    """
    Visualize the clustering results
    params:
    - point_cluster: The result of each data object after clustering
    - line: initial dataset
    returns:
    """
    plt.figure()
    # using user-defined colors in the main function
    # for i in range(point_cluster.shape[0]):
    #     plt.plot(line[i][0], line[i][1], marker='o', markersize=4.0, color=dic_colors[point_cluster[i]])

    # using an official color library
    x = line[:, 0].tolist()
    y = line[:, 1].tolist()
    C = point_cluster.tolist()
    plt.scatter(x, y, c=C, marker='o')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('result')
    plt.show()


if __name__ == '__main__':
    dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                  2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8),
                  6: (0, 0.5, 0.7), 7: (0.8, 0.8, 0.8),
                  8: (0.6, 0, 0), 9: (0, 0.6, 0),
                  10: (1, 0, .8), 11: (0, 1, .8),
                  12: (1, 1, .8), 13: (0.4, 0, .8),
                  14: (0, 0.4, .8), 15: (0.4, 0.4, .8),
                  16: (1, 0.4, .8), 17: (1, 0, 1),
                  18: (1, 0, .8), 19: (.8, 0.2, 0), 20: (0, 0.7, 0),
                  21: (0.9, 0, .8), 22: (.8, .8, 0.1),
                  23: (.8, 0.5, .8), 24: (0, .1, .8),
                  25: (0.9, 0, .8), 26: (.8, .8, 0.1),
                  27: (.8, 0.5, .8), -1: (0, 0, 0),
                  28: (.6, 0.2, 0), 29: (0.2, .6, 0),
                  30: (0, 0.3, .6), 31: (.6, .6, 0),
                  32: (.6, 0, .6), 33: (0, .6, .6),
                  34: (0, 0.9, 0.1), 35: (0.6, 0.6, 0.6),
                  36: (0.2, 0.7, 0), 37: (0, 0.4, 0.5),
                  38: (1, 0.5, .6), 39: (0, .9, .9),
                  40: (1, 1, .6), 41: (0.4, 0, .6),
                  42: (0, 0.4, .6), 43: (0.8, 0.4, .6),
                  44: (0.3, 0.4, .6), 45: (1, 0, 1),
                  46: (1, 0, .6), 47: (.6, 0.2, 0.1), 48: (0, 0.9, 0.2),
                  49: (0.5, 0, .6), 50: (.6, .6, 0.1),
                  51: (.6, 0.5, .6), 52: (0, .1, .6),
                  53: (0.9, 0, .6), 54: (.6, .6, 0.1),
                  55: (.6, 0.5, .6), 56: (.9, 0, 0), 57: (0, .9, 0),
                  58: (0, 0, .9), 59: (.9, .9, 0),
                  }
    np.set_printoptions(threshold=np.inf)

    # txt format
    # line = np.loadtxt('Datasets/t4.txt')

    # mat format
    dataset = io.loadmat('Datasets/sn.mat')
    line = np.array(dataset['sn']).astype(float)

    startTime = time.time()
    pyflann.set_distance_type(distance_type='euclidean')
    flann = pyflann.FLANN()

    # The value of K is adaptive based on the size of the dataset and generally does not require adjustment
    K = int(np.ceil(np.sqrt(line.shape[0])) * 0.3)

    nearest_neighbors, dists = flann.nn(
        line, line, num_neighbors=K, algorithm="kmeans", branching=32, iterations=7, checks=-1)
    n = line.shape[0]
    Point = np.ones(n).astype(int)
    Point = -1 * Point
    gb_list = []
    for i in range(0, n):
        if Point[i] == -1:
            gb_list.append(nearest_neighbors[i])
            Point[nearest_neighbors[i]] = 1
    Radii = []
    Centers = []
    CBS_Centers = []
    for gb in gb_list:
        radius, center = calculate_center_and_radius(gb)
        Radii.append(radius)
        Centers.append(center)
    gbNum = len(gb_list)
    Density = getDensity(Radii, gbNum)
    sortDensity = np.sort(Density)

    # Set a parameter Ratio, which represents the proportion of Core-GBs
    Ratio = 0.92
    density_threshold = sortDensity[int(gbNum * Ratio) - 1]

    # Partitioning Core-GBs and Non-Core-GBs
    CBS, NCBS = Select_CB_and_NCB(Density, density_threshold)

    for i in range(len(CBS)):
        CBS_Centers.append(Centers[CBS[i]])
    cbs_radius = []
    ncbs_radius = []
    for i in CBS:
        cbs_radius.append(Radii[i])
    for i in NCBS:
        ncbs_radius.append(Radii[i])
    cluster = cluster_CBS(CBS, Centers, cbs_radius)
    point_cluster = np.ones(line.shape[0])
    point_cluster = -1 * point_cluster
    for i1 in range(CBS.shape[0]):
        for j1, cb_point_index in enumerate(gb_list[CBS[i1]]):
            point_cluster[cb_point_index] = cluster[i1]
    point_cluster = cluster_NCBS(NCBS, point_cluster, gb_list, CBS_Centers)
    endTime = time.time()
    times = endTime - startTime
    print('runtime：%s s ' % times)

    # clustering result visualization
    draw_all_point(point_cluster, line)
