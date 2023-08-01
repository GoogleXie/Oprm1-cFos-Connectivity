# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from collections import OrderedDict
import pandas as pd
import time
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA

from brainrender import Scene
from brainrender.actors import Points, PointsDensity
import allensdk
from rich import print
from myterial import orange
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import matplotlib.cm as cm
from numpy import ogrid
from PIL import Image
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.cbook import get_sample_data
from matplotlib.pyplot import imread as imread
from mpl_toolkits import mplot3d

from allensdk.core.reference_space_cache import ReferenceSpaceCache
import scipy
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, make_pipeline
from numpy import linalg as LA

import time
from tqdm import tqdm
from nctpy.utils import matrix_normalization
from nctpy.energies import get_control_inputs, integrate_u
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from nctpy.utils import matrix_normalization
from nctpy.energies import sim_state_eq
from nctpy.metrics import ave_control
from nctpy.energies import get_control_inputs, integrate_u
import networkx.algorithms.node_classification as node_classification
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform
import random
import powerlaw
import math

def create_csv_path(title, parent, suffix):
    ext_name = title + suffix
    return os.path.join(parent, ext_name)
# Press the green button in the gutter to run the script.

def corr_to_edgelist(group2v, col0 = "roi 1", col1 = "roi 2", col2 = "corr"):
    links = group2v.stack()
    links.to_csv(outpath3)
    links = pd.read_csv(outpath3)
    links.rename(columns={links.columns[0]: col0}, inplace=True)
    links.rename(columns={links.columns[1]: col1}, inplace=True)
    links.rename(columns={links.columns[2]: col2}, inplace=True)
    links = pd.DataFrame(links)
    links.sort_values(by = ['roi 1', 'roi 2'])
    return links

def generate_network(group2v, roi_list, roi_color):
    links = corr_to_edgelist(group2v)
    for roi in roi_list:
        links = links.loc[(links["roi 1"] != roi) | (links["roi 2"] != roi)]
    links_pos = links.loc[links["corr"] > 0].reset_index()
    links_neg = links.loc[links["corr"] < 0].reset_index()
    links_neg["corr"] = - links_neg["corr"]

    G1 = nx.from_pandas_edgelist(links_pos, 'roi 1', 'roi 2', edge_attr='corr')
    # G2 = nx.from_pandas_edgelist(links_neg, 'roi 1', 'roi 2', edge_attr='corr')
    weight = nx.get_edge_attributes(G1, 'corr').values()
    layout = nx.spring_layout(G1, weight='corr', k=0.8, iterations=500)
    weight = [i * 4 for i in weight]
    roi_color_sorted = list()
    for node in G1.nodes:
        roi_color_sorted.append(roi_color[node])
    nx.draw(G1, layout, node_color= roi_color_sorted, node_size=500, width=list(weight), with_labels=True,
                font_color='black',
                font_weight='bold', cmap=plt.cm.Blues,
                edge_color=list(weight), edge_cmap=plt.cm.Oranges)
    return G1

def generate_network_for_gomory_hu_tree(group2v, roi_list, roi_color):
    links = corr_to_edgelist(group2v)
    for roi in roi_list:
        links = links.loc[(links["roi 1"] != roi) | (links["roi 2"] != roi)]
    links_pos = links.loc[links["corr"] > 0].reset_index()
    links_neg = links.loc[links["corr"] < 0].reset_index()
    links_neg["corr"] = - links_neg["corr"]

    G1 = nx.from_pandas_edgelist(links_pos, 'roi 1', 'roi 2', edge_attr='corr')
    # G2 = nx.from_pandas_edgelist(links_neg, 'roi 1', 'roi 2', edge_attr='corr')
    weight = nx.get_edge_attributes(G1, 'corr').values()

    weight = [i * 4 for i in weight]
    roi_color_sorted = list()
    G1 = nx.gomory_hu_tree(G1, capacity = 'corr')
    #print(G1.edges)
    layout = nx.spring_layout(G1, weight='corr', k=0.8, iterations=500)
    for node in G1.nodes:
        roi_color_sorted.append(roi_color[node])
        #print(node)
    #nx.draw(G1, node_size=500, with_labels=True,font_color='black',font_weight='bold')

    nx.draw(G1, layout, node_color= roi_color_sorted, node_size=500, with_labels=True,
                font_color='black',
                font_weight='bold', cmap=plt.cm.Blues)

    return G1

def generate_abs_distance_network(group2v, roi_list):
    links = group2v.stack()
    links.to_csv(outpath3)
    links = pd.read_csv(outpath3)
    links.rename(columns={links.columns[0]: "roi 1"}, inplace=True)
    links.rename(columns={links.columns[1]: "roi 2"}, inplace=True)
    links.rename(columns={links.columns[2]: "corr"}, inplace=True)


    links = pd.DataFrame(links)
    for roi in roi_list:
        links = links.loc[(links["roi 1"] != roi) | (links["roi 2"] != roi)]
    links_pos = links.loc[links["corr"] > 0].reset_index()
    links_neg = links.loc[links["corr"] < 0].reset_index()
    links_neg["corr"] = - links_neg["corr"]
    # use inverse min max / mean normalization scalar function to calculate the distance between nodes with regard to their correlation coefficient
    #remove outliers by quantile
    links_pos = pd.concat([links_pos,links_neg], axis = 0)
    distance = links_pos["corr"]
    distance = distance.apply(lambda x: np.log(1 / x))
    #q_low = distance.quantile(0.05)
    #q_hi = distance.quantile(0.95)
    #distance = distance[(distance < q_hi)&(distance>q_low)]
    #distance = (distance - distance.min())/(distance.max() - distance.min())
    #links_pos["corr"] = (links_pos["corr"] - links_pos["corr"].mean())/links_pos["corr"].std()
    #print(distance)
    #print(links_pos)
    links_pos["corr"] = distance

    G1 = nx.from_pandas_edgelist(links_pos, 'roi 1', 'roi 2', edge_attr='corr')
    # G2 = nx.from_pandas_edgelist(links_neg, 'roi 1', 'roi 2', edge_attr='corr')
    #print(nx.get_edge_attributes(G1, 'corr'))
    weight = nx.get_edge_attributes(G1, 'corr').values()
    layout = nx.spring_layout(G1, weight='corr', k=0.8, iterations=500)
    weight = [i * 4 for i in weight]
    #print('# of rois: ',len(roi_list))
    '''
    nx.draw(G1, layout, node_color=range(len(roi_list)), node_size=500, width=list(weight), with_labels=True,
                font_color='black',
                font_weight='bold', cmap=plt.cm.Blues,
                edge_color=list(weight), edge_cmap=plt.cm.Oranges)
    '''
    return G1

def generate_distance_network_n(group2v, roi_list):
    links = group2v.stack()
    links.to_csv(outpath3)
    links = pd.read_csv(outpath3)
    links.rename(columns={links.columns[0]: "roi 1"}, inplace=True)
    links.rename(columns={links.columns[1]: "roi 2"}, inplace=True)
    links.rename(columns={links.columns[2]: "corr"}, inplace=True)


    links = pd.DataFrame(links)
    for roi in roi_list:
        links = links.loc[(links["roi 1"] != roi) | (links["roi 2"] != roi)]
    links_pos = links.loc[links["corr"] > 0].reset_index()
    links_neg = links.loc[links["corr"] < 0].reset_index()
    links_neg["corr"] = - links_neg["corr"]
    # use inverse min max / mean normalization scalar function to calculate the distance between nodes with regard to their correlation coefficient
    #remove outliers by quantile

    distance = links_neg["corr"]
    distance = distance.apply(lambda x: np.log(1 / x))
    #q_low = distance.quantile(0.05)
    #q_hi = distance.quantile(0.95)
    #distance = distance[(distance < q_hi)&(distance>q_low)]
    #distance = (distance - distance.min())/(distance.max() - distance.min())
    #links_pos["corr"] = (links_pos["corr"] - links_pos["corr"].mean())/links_pos["corr"].std()
    #print(distance)
    #print(links_pos)
    links_neg["corr"] = distance

    G1 = nx.from_pandas_edgelist(links_neg, 'roi 1', 'roi 2', edge_attr='corr')
    # G2 = nx.from_pandas_edgelist(links_neg, 'roi 1', 'roi 2', edge_attr='corr')
    #print(nx.get_edge_attributes(G1, 'corr'))
    weight = nx.get_edge_attributes(G1, 'corr').values()
    layout = nx.spring_layout(G1, weight='corr', k=0.8, iterations=500)
    weight = [i * 4 for i in weight]
    #print('# of rois: ',len(roi_list))
    '''
    #draw the network graph if needed
    nx.draw(G1, layout, node_color=range(len(roi_list)), node_size=500, width=list(weight), with_labels=True,
                font_color='black',
                font_weight='bold', cmap=plt.cm.Blues,
                edge_color=list(weight), edge_cmap=plt.cm.Oranges)
    '''
    return G1

def generate_distance_network(group2v, roi_list):
    links = group2v.stack()
    links.to_csv(outpath3)
    links = pd.read_csv(outpath3)
    links.rename(columns={links.columns[0]: "roi 1"}, inplace=True)
    links.rename(columns={links.columns[1]: "roi 2"}, inplace=True)
    links.rename(columns={links.columns[2]: "corr"}, inplace=True)
    links = pd.DataFrame(links)
    #select roi specified by roi_list
    #print(roi_list)
    links = links.loc[links["roi 1"].isin(roi_list) & links["roi 2"].isin(roi_list)]
    #separate out pos vs. neg correlation
    for roi in roi_list:

        links = links.loc[(links["roi 1"] != roi) | (links["roi 2"] != roi)]
    links_pos = links.loc[links["corr"] > 0].reset_index()
    links_neg = links.loc[links["corr"] < 0].reset_index()
    links_neg["corr"] = - links_neg["corr"]
    # use inverse min max / mean normalization scalar function to calculate the distance between nodes with regard to their correlation coefficient
    #remove outliers by quantile

    distance = links_pos["corr"]
    distance = distance.apply(lambda x: np.log(1 / x))
    #q_low = distance.quantile(0.05)
    #q_hi = distance.quantile(0.95)
    #distance = distance[(distance < q_hi)&(distance>q_low)]
    #distance = (distance - distance.min())/(distance.max() - distance.min())
    #links_pos["corr"] = (links_pos["corr"] - links_pos["corr"].mean())/links_pos["corr"].std()
    #print(distance)
    #print(links_pos)
    links_pos["corr"] = distance

    G1 = nx.from_pandas_edgelist(links_pos, 'roi 1', 'roi 2', edge_attr='corr')
    # G2 = nx.from_pandas_edgelist(links_neg, 'roi 1', 'roi 2', edge_attr='corr')
    #print(nx.get_edge_attributes(G1, 'corr'))
    weight = nx.get_edge_attributes(G1, 'corr').values()
    layout = nx.spring_layout(G1, weight='corr', k=0.8, iterations=500)
    weight = [i * 4 for i in weight]
    #print('# of rois: ',len(roi_list))
    '''
    #draw the network graph if needed
    nx.draw(G1, layout, node_color=range(len(roi_list)), node_size=500, width=list(weight), with_labels=True,
                font_color='black',
                font_weight='bold', cmap=plt.cm.Blues,
                edge_color=list(weight), edge_cmap=plt.cm.Oranges)
    '''
    return G1


#Test Distribution between two correlation matrix, make sure two matrix has same layout
def Corr_Matrix_Two_Sample_Kol_Test(corr_1, corr_2, message):
    distribution1 = corr_to_edgelist(corr_1).iloc[:,2]
    #print(distribution1)
    distribution2 = corr_to_edgelist(corr_2).iloc[:,2]
    print("Kolmogorov-Smirnov test for: ", message," results:")
    print(scipy.stats.kstest(distribution1,distribution2))


def get_PCA(set1, set2, labels, measure):
    x = set1
    x1 = set2
    x[measure] = labels[0]
    x1[measure] = labels[1]
    print(x, x1)
    x = pd.DataFrame(x)
    x1 = pd.DataFrame(x1)
    x = pd.concat([x, x1], axis=0, ignore_index=True)
    # x = x.iloc[:, :-1]
    print(measure, labels,x)
    x.iloc[:, :-1] = StandardScaler().fit_transform(x.iloc[:, :-1])

    for colidx in range(x.shape[1]):
        if x.iloc[:, colidx].isnull().values.any():
            x.iloc[:, colidx] = x.iloc[:, colidx].fillna(x.iloc[:, colidx].mean())
    # print(x)
    pca_x = PCA(n_components=2)
    principalComponents_x = pca_x.fit_transform(x.iloc[:, :-1])
    principal_x_Df = pd.DataFrame(data=principalComponents_x
                                  , columns=['principal component 1', 'principal component 2'])
    #print('PCA result: ', principal_x_Df)
    x = pd.concat([x, principal_x_Df], axis=1)
    # print(x)
    print('Explained variation per principal component: {}'.format(pca_x.explained_variance_ratio_))
    print('PCA table: ')
    print('####################################')
    print('####################################')
    print('####################################')
    print(x)
    return x

def get_general_PCA(set1, set2, set3, set4, labels, measure):
    x = set1
    x1 = set2
    x2 = set3
    x3 = set4
    x[measure] = labels[0]
    x1[measure] = labels[1]
    x2[measure] = labels[2]
    x3[measure] = labels[3]
    x = pd.DataFrame(x)
    x1 = pd.DataFrame(x1)
    x2 = pd.DataFrame(x2)
    x3 = pd.DataFrame(x3)
    x = pd.concat([x, x1, x2, x3], axis=0, ignore_index=True)
    # x = x.iloc[:, :-1]
    print(measure, labels,x)
    x.iloc[:, :-1] = StandardScaler().fit_transform(x.iloc[:, :-1])

    for colidx in range(x.shape[1]):
        if x.iloc[:, colidx].isnull().values.any():
            x.iloc[:, colidx] = x.iloc[:, colidx].fillna(x.iloc[:, colidx].mean())
    # print(x)
    pca_x = PCA(n_components=2)
    principalComponents_x = pca_x.fit_transform(x.iloc[:, :-1])
    principal_x_Df = pd.DataFrame(data=principalComponents_x
                                  , columns=['principal component 1', 'principal component 2'])
    #print('PCA result: ', principal_x_Df)
    x = pd.concat([x, principal_x_Df], axis=1)
    # print(x)
    print('Explained variation per principal component: {}'.format(pca_x.explained_variance_ratio_))
    print('PCA table: ')
    print('####################################')
    print('####################################')
    print('####################################')
    #print(x)
    return x

def generate_PCA(set1, set2, set11, set12, set21, set22, set31, set32, title1,
                 title2, title3, title4, outpath, title, labels, measure, colors = ['cornflowerblue', 'darkorange'], method = 'one'):
    outpath_pca = os.path.join(outpath, 'PCA Analysis '+title+'.tif')
    colors = colors
    if method == 'one':
        x1 = get_PCA(set1, set2, labels, measure)
        x2 = get_PCA(set11, set12, labels, measure)
        x3 = get_PCA(set21, set22, labels, measure)
        x4 = get_PCA(set31, set32, labels, measure)
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('A112G cFos Expression PCA by ' + measure, fontsize=24)

        for label, color in zip(labels, colors):
            # group 1
            data00 = (x1.loc[x1[measure] == label])[['principal component 1', 'principal component 2']]
            axs[0, 0].scatter(data00['principal component 1'], data00['principal component 2'], c=color, s=50,
                              label=label)
            axs[0, 0].set_title(title1, size=20)
            data00 = data00.values
            hull = ConvexHull(data00)
            for j in hull.simplices:
                axs[0, 0].plot(data00[j, 0], data00[j, 1], color=color)
            # group 2
            data10 = (x2.loc[x2[measure] == label])[['principal component 1', 'principal component 2']]
            axs[1, 0].scatter(data10['principal component 1'], data10['principal component 2'], c=color, s=50)
            axs[1, 0].set_title(title2, size=20)
            data10 = data10.values
            hull = ConvexHull(data10)
            for j in hull.simplices:
                axs[1, 0].plot(data10[j, 0], data10[j, 1], color=color)
            # group 3
            data01 = (x3.loc[x3[measure] == label])[['principal component 1', 'principal component 2']]
            axs[0, 1].scatter(data01['principal component 1'], data01['principal component 2'], c=color, s=50)
            axs[0, 1].set_title(title3, size=20)
            data01 = data01.values
            hull = ConvexHull(data01)
            for j in hull.simplices:
                axs[0, 1].plot(data01[j, 0], data01[j, 1], color=color)
            # group 4
            data11 = (x4.loc[x4[measure] == label])[['principal component 1', 'principal component 2']]
            axs[1, 1].scatter(data11['principal component 1'], data11['principal component 2'], c=color, s=50)
            axs[1, 1].set_title(title4, size=20)
            data11 = data11.values
            hull = ConvexHull(data11)
            for j in hull.simplices:
                axs[1, 1].plot(data11[j, 0], data11[j, 1], color=color)
    else:
        x1 = get_general_PCA(set1,set2,set11,set12, labels, measure)
        x2 = get_general_PCA(set21,set22,set31,set32, labels, measure)
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('A112G cFos Expression PCA by ' + measure, fontsize=24)

        for label, color in zip(labels, colors):
            # group 1
            data00 = (x1.loc[x1[measure] == label])[['principal component 1', 'principal component 2']]
            axs[0].scatter(data00['principal component 1'], data00['principal component 2'], c=color, s=50,
                              label=label)
            axs[0].set_title(title1, size=20)
            data00 = data00.values
            hull = ConvexHull(data00)
            for j in hull.simplices:
                axs[0].plot(data00[j, 0], data00[j, 1], color=color)

            # group 3
            data01 = (x2.loc[x2[measure] == label])[['principal component 1', 'principal component 2']]
            axs[1].scatter(data01['principal component 1'], data01['principal component 2'], c=color, s=50)
            axs[1].set_title(title3, size=20)
            data01 = data01.values
            hull = ConvexHull(data01)
            for j in hull.simplices:
                axs[1].plot(data01[j, 0], data01[j, 1], color=color)

    #fig.savefig(outpath_pca,transparent= True, dpi = 500)
    #plt.figure()
    #plt.figure(figsize=(10, 10))
    #plt.xlabel('Principal Component - 1', fontsize=20)
    #plt.ylabel('Principal Component - 2', fontsize=20)
    #pcatitle1 = "Principal Component Analysis of A112G: " + title1
    #plt.title(pcatitle1, fontsize=20)




        '''
        axs[1, 0].scatter((x2.loc[x2['Genotype'] == label])['principal component 1'],
                          (x2.loc[x2['Genotype'] == label])['principal component 2'], c=color, s=50)
        axs[1, 0].set_title(title2)

        axs[0, 1].scatter((x3.loc[x3['Genotype'] == label])['principal component 1'],
                          (x3.loc[x3['Genotype'] == label])['principal component 2'], c=color, s=50)
        axs[0, 1].set_title(title3)

        axs[1, 1].scatter((x4.loc[x4['Genotype'] == label])['principal component 1'],
                          (x4.loc[x4['Genotype'] == label])['principal component 2'], c=color, s=50)
        axs[1, 1].set_title(title4)
        '''
    fig.legend(prop={'size': 15})
    plt.show()
    fig.savefig(outpath_pca, transparent=True, dpi=500)


def generate_connectome_matrix(roi_list,method = 'avginterregion', get_cre_experiment = False):
    roi_list = roi_list
    mcc = MouseConnectivityCache()
    all_experiments = mcc.get_experiments(dataframe=True)
    structure_tree = mcc.get_structure_tree()
    # structures = structure_tree.get_structures_by_name(['Primary visual area', 'Hypothalamus'])

    structures = structure_tree.get_structures_by_acronym(roi_list)
    # experiments = mcc.get_experiments(cre=False, injection_structure_ids=[structures[:]['id']])
    #print(structures)

    struclist = list()
    connectome = pd.DataFrame(columns=roi_list)

    emptyrow = pd.DataFrame(columns=roi_list)
    #print(connectome)
    for i in range(len(roi_list)):
        struclist.append(structures[i]['id'])

    #print('structure ids: ', struclist)
    for i in range(len(roi_list)):
        print('getting structure: ',structures[i]['acronym'], structures[i]['id'])
        #print([structures[i]['id'] for i in structures[:]])
        # if want no cre simply add cre = get_cre_experiment in get_experiments
        experiments = mcc.get_experiments(cre = get_cre_experiment, injection_structure_ids=[structures[i]['id']])
        if len(experiments) != 0:
            experiment_ids = [e['id'] for e in experiments]
            pm = mcc.get_projection_matrix(experiment_ids=experiment_ids,
                                           projection_structure_ids=struclist,
                                           hemisphere_ids=[3],  # both hemisphere
                                           parameter='projection_density')

            # print(pm['matrix'])
            pmlabel = [e['label'] for e in pm['columns']]

            midx = [structures[i]['acronym'] for e in pm['rows']]
            # print(midx)
            cm = pd.DataFrame(pm['matrix'], columns=pmlabel, index=midx)
            #print(cm)
            connectome = pd.concat([connectome, cm], axis=0, sort=True)

        else:
            emptyrow = emptyrow.reindex([roi_list[i]])
            connectome = pd.concat([connectome, emptyrow], axis = 0, sort=True)
    # print('raw connectivity matrix with rois injected experiments: ',connectome)
    if method == 'avginterregion':
        #connectmin = min(connectome[connectome > 0].min(skipna=True))
        #connectome = connectome.fillna(0.01*connectmin)
        avgconnectome = connectome.groupby(level=0).mean()
        #print(avgconnectome)
        avgconnectome.to_csv(connectome_outpath)
        return avgconnectome
'''
#following min_control_energy function modified based on
# -Brynildsen JK, Mace KD, Cornblath EJ, Weidler C, Pasqualetti F, Bassett DS, Blendy JA (2020) Gene coexpression patterns predict opiate-induced brain-state transitions. Proceedings of the National Academy of Science
def min_control_energy(A, T, B, x0, xf, nor):
    # compute minimum control energy
    # A: system adjacency matrix: NxN
    # B: control input matrix: Nxk
    # x0: initial state: Nx1
    # xf: final state: Nx1
    # T: control horizon: 1x1
    # nor: normalization boolean
    n = A.len()
    if nor == 1:
        egval, egvec = LA.eigvals(A)
        A = np.divide(A,((egval[-1] + 1) - np.identity(n)))
        z1 = np.concatenate((A, -0.5*(np.matmul(B,B.T))), axis = 1)
        z2 = np.concatenate((np.zeros(A.shape), -A.T), axis = 1)
        AT = np.concatenate((z1,z2), axis = 0)
    E = scipy.linalg.expm(np.matmul(AT,A))

    E12 = E[0:n-1, n:(2*n-1)]
    E11 = E[0:n-1, 0:n-1]
    p0 = np.matmul(scipy.linalg.pinv(E12),(xf - np.matmul(E11,x0)))
    n_err = np.linalg.norm(np.matmul(E12,p0) - (xf - np.matmul(E11,x0)))

    nstep = 1000;
    t = np.linspace(0, T, nstep+1)
    v0 = np.concatenate((x0, p0), axis = 1)
    v = np.zeros(2*n,len(t))
    Et = scipy.linalg.expm(np.matmul(AT,T)/(len(t)-1))
    v[:,0] = v0

    for i in range(len(t)):
        v[:, i+1] = np.matmul(Et,v[:, i])
    x = v[0:n-1,:]
    u = -0.5*np.matmul(B.T, v[n:(2*n-1), :])

    print(n_err)
    print(np.linalg.norm(x[:,-1]-xf))
    u = u.T
    x = x.T
    return x, u, n_err
'''


def min_control_energy(outpath0, outpathf, roi_list, tracker, receptor_density = [], blunt_region = [], method = 'mean', title = 'nan', brho = 1):
    connectome = generate_connectome_matrix(roi_list)
    #print(connectome)
    M = len(connectome.index)
    N = len(connectome.columns)
    ran = pd.DataFrame(np.abs(np.random.randn(M, N)) * 1e-8, columns=connectome.columns, index=connectome.index)
    # print(ran)
    connectome.loc[connectome.isnull().any(axis=1)] = ran.loc[connectome.isnull().any(axis=1)]
    connectomeA = connectome.to_numpy()
    #print(connectomeA)
    print('method: ', method)
    if method == 'mean':
        x0 = pd.read_csv(outpath0, index_col=0)
        xf = pd.read_csv(outpathf, index_col=0)
        x0 = pd.DataFrame(x0).mean(axis=0)
        x0 = x0[roi_list]
        #print('state0: ', x0)
        x0 = x0.to_numpy()


        xf = pd.DataFrame(xf).mean(axis=0)
        xf = xf[roi_list]
        #print('statef: ', xf)
        xf = xf.to_numpy()
        # print(xf)

        system = 'continuous'
        A_norm = matrix_normalization(A=connectomeA, c=1, system=system)
        B = np.eye(len(roi_list))  # uniform full control set
        '''

            #simulated neural activity
            system = 'discrete'

            print(A_norm.shape)
            T = 20  # time horizon
            U = np.zeros((len(roi_list), T))  # the input to the system
            U[:, 0] = 1  # impulse, 1 input at the first time point delivered to all nodes

            x = sim_state_eq(A_norm=A_norm, B=B, x0=x0, U=U, system=system)
            f, ax = plt.subplots(1, 1)
            ax.plot(x.T)
            ax.set_ylabel('Simulated neural activity')
            ax.set_xlabel('Time')
            plt.show()

            ac = ave_control(A_norm=A_norm, system=system)
            ac = pd.DataFrame(ac, index=roi_list)
            print('avg control ability: ', ac)
            '''

        # set parameters

        T = 1  # time horizon
        rho = 1  # mixing parameter for state trajectory constraint
        S = np.zeros((len(roi_list), len(roi_list)))  # nodes in state trajectory to be constrained
        # print("constrain roi matrix: ", S)
        # get the state trajectory (x) and the control inputs (u)
        x, u, n_err = get_control_inputs(A_norm=A_norm, T=T, B=B, x0=x0, xf=xf, system=system, rho=rho, S=S)
        '''
        f, ax = plt.subplots(1, 2, figsize=(6, 3))
        # plot control signals for initial state
        ax[0].plot(u)
        ax[0].set_title('control signals')

        # plot state trajectory for initial state
        ax[1].plot(x)
        ax[1].set_title('state trajectory (neural activity)')

        for cax in ax.reshape(-1):
            cax.set_ylabel("activity")
            cax.set_xlabel("time (arbitary units)")
            cax.set_xticks([0, x.shape[0]])
            cax.set_xticklabels([0, T])

        f.tight_layout()
        f.savefig('plot_xu.png', dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.show()
        '''
        # integrate control inputs to get control energy
        node_energy = integrate_u(u)
        # print('node energy =', node_energy)
        # summarize nodal energy to get control energy
        energy0 = np.sum(node_energy)
        print('energy = {:.2F}'.format(np.round(energy0, 2)))
        minenergy_list = []
        B = np.eye(len(roi_list))
        for id in range(len(roi_list)):
            # nodes in state trajectory to be constrained
            S = np.zeros((len(roi_list), len(roi_list)))
            S[id, id] = 1
            # print("constrain roi: ", S[id])
            x, u, n_err = get_control_inputs(A_norm=A_norm, T=T, B=B, x0=x0, xf=xf,system=system, rho=rho, S=S)
            node_energy = integrate_u(u)
            # print('node energy =', node_energy)
            energy = np.sum(node_energy)
            # print('energy = {:.2F}'.format(np.round(energy, 2)))
            minenergy_list.append(energy)
    else:
        if len(blunt_region) >= 1:
            x0 = pd.read_csv(outpath0, index_col=0)
            xf = pd.read_csv(outpathf, index_col=0)
            x0 = pd.DataFrame(x0)
            xf = pd.DataFrame(xf)
            energybluntlist = []
            pairidx = []
            T = 1

            for i in range(x0.shape[0]):
                for j in range(xf.shape[0]):
                    row0 = x0.iloc[i, :]
                    rowf = xf.iloc[j, :]
                    row0 = row0[roi_list]
                    # print('state0: ', row0)
                    row0 = row0.to_numpy()
                    rowf = rowf[roi_list]
                    # print('statef: ', rowf)
                    rowf = rowf.to_numpy()
                    # print(xf)
                    pair = str(i) + '_' + str(j)
                    pairidx.append(pair)
                    print('calculating pair of conversion: ', pair, ' |   in: ',title, ' | at round: ', tracker)
                    system = 'continuous'
                    A_norm = matrix_normalization(A=connectomeA, c=1, system=system)
                    B = np.eye(len(roi_list))  # uniform full control set
                    S = np.zeros((len(roi_list), len(roi_list)))
                    for region in blunt_region:
                        #print('blunt region ', region)
                        id = roi_list.index(region)
                        B[id, id] = 0
                    x, u, n_err = get_control_inputs(A_norm=A_norm, T=T, B=B, x0=row0, xf=rowf, system=system, rho=brho, S=S)
                    energy_blunt = integrate_u(u)
                    energy_blunt = np.sum(energy_blunt)
                    energybluntlist.append(energy_blunt)
            energybluntlist = pd.DataFrame(energybluntlist, index=pairidx, columns=[title])
            return energybluntlist
        else:
            x0 = pd.read_csv(outpath0, index_col=0)
            xf = pd.read_csv(outpathf, index_col=0)
            x0 = pd.DataFrame(x0)
            xf = pd.DataFrame(xf)
            energy0list = []
            energy0list1 = []
            energy0list2 = []
            energy0list3 = []
            pairidx = []
            min_control_energy_list1 = pd.DataFrame(index=roi_list)
            S = np.zeros((len(roi_list), len(roi_list)))  # nodes in state trajectory to be constrained
            for i in range(x0.shape[0]):
                for j in range(xf.shape[0]):
                    row0 = x0.iloc[i, :]
                    rowf = xf.iloc[j, :]
                    row0 = row0[roi_list]
                    # print('state0: ', row0)
                    row0 = row0.to_numpy()
                    rowf = rowf[roi_list]
                    # print('statef: ', rowf)
                    rowf = rowf.to_numpy()
                    # print(xf)
                    pair = str(i) + '_' + str(j)
                    pairidx.append(pair)
                    print('calculating pair of conversion: ', pair, ' |   in: ',title,' | at round: ', tracker)
                    system = 'continuous'
                    A_norm = matrix_normalization(A=connectomeA, c=1, system=system)
                    B = np.eye(len(roi_list))  # uniform full control set
                    T = 1  # time horizon
                    rho = 1  # mixing parameter for state trajectory constraint
                    # set parameters
                    if tracker == 'receptor':
                        for receptori in range(receptor_density.shape[0]):
                            B = np.eye(len(roi_list))
                            for rd in range(len(roi_list)):
                                B[rd, rd] = receptor_density.iloc[receptori,rd]
                            print('Receptor '+str(receptori)+' Density incorporated with B matrix: ')
                            print(B)
                            x, u, n_err = get_control_inputs(A_norm=A_norm, T=T, B=B, x0=row0, xf=rowf, system=system,
                                                             rho=rho,
                                                             S=S)
                            node_energy = integrate_u(u)
                            energy0 = np.sum(node_energy)
                            if receptori == 0:
                                energy0list1.append(energy0)
                            elif receptori == 1:
                                energy0list2.append(energy0)
                            else:
                                energy0list3.append(energy0)
                            #print('mu: ', energy0list1)
                            print('energy = {:.2F}'.format(np.round(energy0, 2)))
                        minenergy_list = []
                        min_control_energy_list1 = pd.DataFrame()
                    else:
                        # print("constrain roi matrix: ", S)
                        # get the state trajectory (x) and the control inputs (u)
                        x, u, n_err = get_control_inputs(A_norm=A_norm, T=T, B=B, x0=row0, xf=rowf, system=system,
                                                         rho=rho,
                                                         S=S)

                        # integrate control inputs to get control energy
                        node_energy = integrate_u(u)
                        # print('node energy =', node_energy)
                        # summarize nodal energy to get control energy
                        energy0 = np.sum(node_energy)
                        energy0list.append(energy0)
                        print('energy = {:.2F}'.format(np.round(energy0, 2)))
                        minenergy_list = []
                        for id in range(len(roi_list)):
                            # nodes in state trajectory to be constrained
                            S = np.zeros((len(roi_list), len(roi_list)))
                            # B = np.eye(len(roi_list))
                            S[id, id] = 1
                            # print("constrain roi: ", S[id])
                            x, u, n_err = get_control_inputs(A_norm=A_norm, T=T, B=B, x0=row0, xf=rowf, system=system,
                                                             rho=rho, S=S)
                            node_energy = integrate_u(u)
                            # print('node energy =', node_energy)
                            energy = np.sum(node_energy)
                            # print('energy = {:.2F}'.format(np.round(energy, 2)))
                            minenergy_list.append(energy)
                        minenergy_list = pd.DataFrame(minenergy_list, index=roi_list, columns=[pair])
                        min_control_energy_list1 = pd.concat([min_control_energy_list1, minenergy_list], axis=1)
            x0 = x0[roi_list]
            x0 = x0.to_numpy()
            xf = xf[roi_list]
            xf = xf.to_numpy()
            min_control_energy_list1['group'] = title
            print('min control energy by each pair:')
            print(min_control_energy_list1)
            print('min energy by each pair')
            #energy0list = pd.DataFrame(energy0list, index=pairidx, columns=[title])
            if tracker == 'receptor':
                energy0list = energy0list1 +[123]+ energy0list3 + [123]+ energy0list3
                energy0list = pd.DataFrame(energy0list, columns=[title])
                print(energy0list)
                return min_control_energy_list1, energy0list
            else:
                energy0list = pd.DataFrame(energy0list, columns=[title])
                print(energy0list)
                return min_control_energy_list1, energy0list

    if len(blunt_region) >= 1:
        S = np.zeros((len(roi_list), len(roi_list)))
        B = np.eye(len(roi_list))
        for region in blunt_region:
            id = roi_list.index(region)
            S[id, id] = 1
        x, u, n_err = get_control_inputs(A_norm=A_norm, T=T, B=B, x0=x0, xf=xf, system=system, rho=brho, S=S)
        energy_blunt = integrate_u(u)
        energy_blunt1 = np.sum(energy_blunt)
        return energy_blunt1-energy0, energy0
    else:
        # return minenergy_list - energy0, energy0
        # following returns with percentage wise min control energy
        return minenergy_list - energy0, energy0

def get_distance_from_set(corr, gene, roi_list):
    data1 = pd.DataFrame(columns=['Genotype', 'ROI', 'Distance'])
    y = pd.DataFrame(columns = roi_list)
    avg_shortest_path_list1 = {}
    for n, (dist, path) in corr:
        x1 = pd.DataFrame([gene], columns=['Genotype'])
        n1 = pd.DataFrame([n], columns=['ROI'])
        d1 = pd.DataFrame(dist.values(), columns=['Distance'])
        x1 = pd.concat([x1,n1, d1], axis = 1)
        y1 = pd.DataFrame(dist.values(), index = roi_list, columns = [n])
        y1 = y1.transpose()
        x1['Genotype'] = x1['Genotype'][0]
        x1['ROI'] = x1['ROI'][0]
        data1 = pd.concat([data1, x1], axis = 0, ignore_index=True)
        y = pd.concat([y,y1], axis = 0)
        avg_shortest_path1 = sum(dist.values()) / len(dist)
        avg_shortest_path_list1[n] = avg_shortest_path1
        avg_shortest_path_list1 = dict(sorted(avg_shortest_path_list1.items()))
    return avg_shortest_path_list1, data1, y

def get_closet_path(corraa, corrgg, roi_list, title):
    any_title = "Avg Shortest Path Passing through region AA vs. GG "+ title
    corraa = nx.all_pairs_dijkstra(generate_distance_network(corraa, roi_list),
                                                    weight='corr')
    corrgg = nx.all_pairs_dijkstra(generate_distance_network(corrgg, roi_list),
                                                    weight='corr')
    avg_shortest_path_list1, data1, tdata1 = get_distance_from_set(corraa, 'AA', roi_list)
    avg_shortest_path_list2, data2, tdata2 = get_distance_from_set(corrgg, 'GG', roi_list)
    avg_shortest_path_list1_f = pd.DataFrame(list(avg_shortest_path_list1.values()), index=roi_list)
    avg_shortest_path_list2_f = pd.DataFrame(list(avg_shortest_path_list2.values()), index=roi_list)
    avg_st_path = pd.concat([avg_shortest_path_list1_f,avg_shortest_path_list2_f], axis = 1)
    data = pd.concat([data1,data2], axis = 0, ignore_index=True)
    tdata = pd.concat([tdata1, tdata2], axis = 1)
    print(data)
    model = ols('Distance ~ C(Genotype) + C(ROI) + C(Genotype):C(ROI)',data = data).fit()
    result = sm.stats.anova_lm(model, type=2)
    print('Two way Anova for ', any_title)
    print(result)
    return data, tdata, avg_st_path
    #print(avg_shortest_path_list1.values())
    #print('Komologov Test for ', any_title)
    #print(scipy.stats.kstest(list(avg_shortest_path_list1.values()), list(avg_shortest_path_list2.values())))

    '''
    x = np.arange(len(avg_shortest_path_list1))
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(any_title)
    plt.bar(x - 0.2, list(avg_shortest_path_list1.values()), 0.4, label = 'AA')
    plt.bar(x + 0.2, list(avg_shortest_path_list2.values()), 0.4, label = 'GG')

    plt.xticks(x, list(avg_shortest_path_list1.keys()))
    plt.xlabel("ROIs")
    plt.ylabel("Avg Shortest Path length")
    plt.legend()
    plt.show()
    '''

def get_sim_rank_matrix(corr, roi_list):
    G = generate_distance_network(corr, roi_list)
    simrank_G = nx.simrank_similarity(G)
    edgelist_simrank_G = []
    for roi1 in simrank_G:
        for item in simrank_G[roi1].items():
            edgelist_simrank_G.append([roi1, item[0], item[1]])

    simrank_matrix = pd.DataFrame(edgelist_simrank_G, columns=['roi', 'roi1', 'similarity'])
    simrank_matrix = pd.pivot(simrank_matrix, index= 'roi', columns='roi1', values='similarity')
    return simrank_matrix


def remove_label(axs, shape):
    for i in range(shape[0]):
        for j in range(shape[1]):
            axs[i, j].set_ylabel('')
            axs[i, j].set_xlabel('')

def group_closeness_centrality(corr, roi_list, exp_group):
    cortex = ['ACAd', 'ACAv', 'AId', 'AIv','CLA','ILA','PL']
    striatum = ['ACB','CP']
    pallidum = ['BST','GPe', 'GPi', 'PALv']
    amygdala = ['BLA','CEA']
    thalamus = ['LH','MH','PVT']
    midbrain = ['PAG','SNc','SNr','VTA']
    Groups = ['cortex','striatum','pallidum','amygdala','thalamus','midbrain']
    closeness_cortex = nx.group_closeness_centrality(generate_distance_network(corr, roi_list), cortex, weight = 'corr')
    closeness_striatum = nx.group_closeness_centrality(generate_distance_network(corr, roi_list), striatum, weight='corr')
    closeness_pallidum = nx.group_closeness_centrality(generate_distance_network(corr, roi_list), pallidum, weight='corr')
    closeness_amygdala = nx.group_closeness_centrality(generate_distance_network(corr, roi_list), amygdala, weight='corr')
    closeness_thalamus = nx.group_closeness_centrality(generate_distance_network(corr, roi_list), thalamus, weight='corr')
    closeness_midbrain = nx.group_closeness_centrality(generate_distance_network(corr, roi_list), midbrain, weight='corr')
    closeness = pd.DataFrame([closeness_cortex, closeness_striatum, closeness_pallidum, closeness_amygdala, closeness_thalamus, closeness_midbrain], index = Groups, columns=[exp_group])
    return closeness

def plot_group_closeness(closeness, sex, out_path):
    region_groups = ['cortex', 'striatum', 'pallidum', 'amygdala', 'thalamus', 'midbrain']
    close_x = np.arange(len(region_groups))
    fig_name = "Group Closeness " + sex
    close_graph_path = os.path.join(out_path,fig_name)
    close_width = 0.2
    close_multipler = 0
    fig, ax = plt.subplots(constrained_layout=True)
    plt.title('Group Closeness '+sex)
    for j in range(len(closeness.columns)):
        offset = close_width * close_multipler
        closeg = ax.bar(close_x + offset, closeness.iloc[:, j], close_width, label=closeness.columns[j])
        ax.bar_label(closeg, padding=4)
        print(closeness.iloc[:, j])
        close_multipler += 1

    for bars in ax.containers:
        print(bars)
    ax.set_ylabel('Weighted Distance')
    close_x = close_x + close_width
    ax.set_xticks(close_x + 0.2, region_groups)
    ax.legend(loc='upper left')
    plt.show()
    fig.savefig(close_graph_path, transparent= True, dpi = 500)

def generate_network_role(corr, roi_list, method):
    G = generate_distance_network(corr,roi_list)
    intoxication = ['CP','GPe','GPi','ACB']
    withdrawal = ['BST','PALv','CEA']
    anticipation = ['ACAd','ACAv','PL','ILA','AId','AIv']
    plan_control = ['ACAd','ACAv','PL','ILA']
    reward_motivation = ['ACB','VTA']
    emotional_learning = ['BLA','CEA']
    #print(G.nodes)
    if method == 'reinforcement':
        plan_control_selected = [plan_control[i] for i in random.sample(range(0, len(plan_control)), 1)]
        reward_motivation_selected = [reward_motivation[i] for i in random.sample(range(0, len(reward_motivation)), 1)]
        emotional_learning_selected = [emotional_learning[i] for i in random.sample(range(0, len(emotional_learning)), 1)]
        selection = plan_control_selected + reward_motivation_selected + emotional_learning_selected
        print('selected: ', selection)
        G.nodes[''.join(plan_control_selected)]['label'] = 'plan_control_decision'
        G.nodes[''.join(reward_motivation_selected)]['label'] = 'reward_motivation'
        G.nodes[''.join(emotional_learning_selected)]['label'] = 'Emotional_learning'
        selection = selection + ['CP','DG']
        G.nodes['CP']['label'] = 'Habit_learning'
        G.nodes['DG']['label'] = 'Context_memory'
        return G, G.nodes, selection
    else:
        intoxication_selected = [intoxication[i] for i in random.sample(range(0, len(intoxication)), 1)]
        anticipation_selected = [anticipation[i] for i in random.sample(range(0, len(anticipation)), 1)]
        withdrawal_selected = [withdrawal[i] for i in random.sample(range(0, len(withdrawal)), 1)]
        G.nodes[''.join(intoxication_selected)]['label'] = 'Binge/Intoxication'
        G.nodes[''.join(anticipation_selected)]['label'] = 'Anticipation/Preoccupation'
        G.nodes[''.join(withdrawal_selected)]['label'] = 'Withdrawal'
        selection= intoxication_selected + withdrawal_selected + anticipation_selected
        print('selected: ', selection)
        return G, G.nodes, selection


def assign_network_color(Graph):
    roi_color = {'VTA': "darkcyan", 'ACB':"chocolate", 'ACAd': "dodgerblue", 'ACAv': "dodgerblue", 'AId': "dodgerblue", 'AIv': "dodgerblue", 'PL': "dodgerblue",
                'ILA': "dodgerblue", 'BLA':"crimson", 'BST': "violet", 'CEA':"crimson", 'CLA': "dodgerblue", 'CP':"chocolate", 'DG':"gold", 'GPe': "violet", 'GPi': "violet", 'LH':"orange", 'LHA':"gold", 'MH':"orange", 'PAG': "darkcyan", 'PALv': "violet", 'PVT':"orange",
                'SNc': "darkcyan",'SNr': "darkcyan"}
    for node in roi_color.keys():
        #print(node)
        Graph.nodes[node]["color"] = roi_color[node]
        #print(Graph.nodes[node])

def get_constraint(corr, roi_list, name):
    constraint = dict(sorted(nx.algorithms.structuralholes.constraint(generate_distance_network(corr, roi_list),
                                                         weight='corr').items()))
    constraint = pd.DataFrame(list(constraint.values()), index = list(constraint.keys()),columns = [name])
    return constraint
def get_effective_size(corr, roi_list, genotype, treatment, sex):
    name = genotype + ' ' + treatment+ ' ' + sex
    effective_size = dict(sorted(nx.algorithms.structuralholes.effective_size(generate_distance_network(corr, roi_list), weight = 'corr').items()))
    effective_size = pd.DataFrame(list(effective_size.values()), index = list(effective_size.keys()), columns = [name])
    #effective_size['Genotype'] = genotype
    #effective_size['Treatment'] = treatment
    #effective_size['Sex'] = sex
    #effective_size.reset_index(inplace=True)
    #effective_size = effective_size.rename(columns={'index': 'ROI'})
    return effective_size

def get_clustering(corr, roi_list, name):
    clustering = dict(sorted(nx.clustering(generate_distance_network(corr, roi_list),
                                                                      weight='corr').items()))
    clustering = pd.DataFrame(list(clustering.values()), index=list(clustering.keys()), columns=[name])
    return clustering

def get_upper(corr):
    #corr is dataframe
    corr = corr.values
    upper = np.triu_indices(corr.shape[0], k=1)
    return corr[upper]

#following function is modified based on -https://towardsdatascience.com/how-to-measure-similarity-between-two-correlation-matrices-ce2ea13d8231
def Compare_Matrix_Perm_Spearmanr(corr1, corr2, title, outpath, number_iteration = 5000):
    title1 = title + '.tif'
    outpath_matrix_compare = os.path.join(outpath,title1)
    #note the spearmanr test here the rho is good but P value is not good since our corr coefficient is not independent from one another
    #need permutation test
    print("Bare comparison without permutation test: ", scipy.stats.spearmanr(get_upper(corr1),get_upper(corr2)))
    true_rho, _ = scipy.stats.spearmanr(get_upper(corr1),get_upper(corr2))
    rho = []
    corr_id = list(corr1.columns)
    corr2_v = get_upper(corr2)
    for iter in range(number_iteration):
        np.random.shuffle(corr_id)
        r,_ = scipy.stats.spearmanr(get_upper(corr1.loc[corr_id, corr_id]),corr2_v)
        rho.append(r)
    permuted_p = ((np.sum(np.abs(true_rho) <= np.abs(rho))) + 1) / (number_iteration + 1)
    fig, ax = plt.subplots()
    plt.hist(rho, bins=20, color='lightgrey')
    ax.axvline(true_rho, color='r', linestyle='--')
    ax.set(title=f"Permuted Spearmanr Test p= {permuted_p:.3f}"+" "+title, ylabel="counts", xlabel="rho")
    plt.show()
    #fig.savefig(outpath_matrix_compare, transparent= True, dpi = 500)

def bootstrap(sample, n):
    boot_sample = []
    mean = sample.mean()
    for i in range(n):
        y = random.sample(sample.tolist(), len(sample))
        norm_y = y / mean
        boot_sample.append(norm_y)
    return boot_sample
def bootstrap_dataframe(dataframe, out_sample_size):
    #print('original: ',dataframe)
    boot_dataframe = pd.DataFrame(columns = dataframe.columns)
    for i in range(out_sample_size):
        x = dataframe.iloc[np.random.randint(dataframe.shape[0], size = dataframe.shape[0]), :]
        x = x[dataframe.columns].mean()
        x = pd.DataFrame(x, columns = [i])
        #print(x.T)
        boot_dataframe = pd.concat([boot_dataframe,x.T], axis = 0)
    print(boot_dataframe)
    return boot_dataframe

def reform_min_energy_boot(energy_path, temporal_path, genotype, sex):
    min_control_energy_aa_m = pd.read_csv(energy_path)
    min_control_energy_aa_m = min_control_energy_aa_m.drop(columns=['group'])
    min_control_energy_aa_m.rename(columns={min_control_energy_aa_m.columns[0]: "roi"}, inplace=True)
    #min_control_energy_aa_m['roi'] = range(23)
    min_control_energy_aa_m.set_index(['roi'], inplace=True)
    print('roi: ', min_control_energy_aa_m)
    min_control_energy_aa_m = min_control_energy_aa_m.stack()
    min_control_energy_aa_m.to_csv(temporal_path)
    min_control_energy_aa_m = pd.read_csv(temporal_path)
    min_control_energy_aa_m.rename(
        columns={min_control_energy_aa_m.columns[1]: "pair", min_control_energy_aa_m.columns[2]: "energy"},
        inplace=True)
    min_control_energy_aa_m['genotype'] = genotype
    min_control_energy_aa_m['sex'] = sex
    return min_control_energy_aa_m

def test_scale_free(g,title): #https://stackoverflow.com/questions/49908014/how-can-i-check-if-a-network-is-scale-free
    '''
    degree_sorted = sorted([d for n,d in g.degree()],reverse = True)
    fit = powerlaw.Fit(degree_sorted, xmin = 1)
    fig = fit.plot_pdf(color='b', linewidth=2)
    fig.set_title(title)
    fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig)
    #R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    #print(R, p)
    plt.figure(figsize=(10, 6))
    plt.show()

    fit.distribution_compare('power_law', 'lognormal')
    fig4 = fit.plot_ccdf(linewidth=3, color='black')
    fit.power_law.plot_ccdf(ax=fig4, color='r', linestyle='--')  # powerlaw
    fit.lognormal.plot_ccdf(ax=fig4, color='g', linestyle='--')  # lognormal
    fit.stretched_exponential.plot_ccdf(ax=fig4, color='b', linestyle='--')  # stretched_exponential
    '''
    k = []
    Pk = []
    logk = []
    logPk = []

    for node in list(g.nodes()):
        degree = g.degree(nbunch=node)
        try:
            pos = k.index(degree)
        except ValueError as e:
            k.append(degree)
            Pk.append(1)
        else:
            Pk[pos] += 1

    # get a double log representation
    for i in range(len(k)):
        logk.append(math.log10(k[i]))
        logPk.append(math.log10(Pk[i]))

    order = np.argsort(logk)
    logk_array = np.array(logk)[order]
    logPk_array = np.array(logPk)[order]
    plt.plot(logk_array, logPk_array, ".")
    plt.title(title)
    m, c = np.polyfit(logk_array, logPk_array, 1)
    plt.plot(logk_array, m * logk_array + c, "-")
    print(m)
    return m
    #plt.show()



if __name__ == '__main__':

    path = "F:\Count Summary"

    path2 = "F:\Count Summary\Corr"
    path3 = "F:\Count Summary\ptimage"
    path_analysis = "F:\Count Summary\Analysis"
    badimagefile = "F:\Count Summary\Bad Image.csv"
    labelfile = "F:\Count Summary\label.csv"
    connectome_outpath = "F:\Count Summary\connectome.csv"
    dir_list = os.listdir(path)
    Meta = pd.DataFrame()
    outpath1 = os.path.join(path, 'avgMetaSummary.csv')
    outpath2 = os.path.join(path, 'avgMetaSummarywithlabel.csv')
    outpath3 = os.path.join(path, 'testlist.csv')
    outpath_node_classification1 = os.path.join(path_analysis, 'node_classification_opioid_reinforcement.csv')
    outpath_node_classification2 = os.path.join(path_analysis, 'node_classification_drug_addiction.csv')
    outpath_aa_acute_m = os.path.join(path,'aa acute male cell density.csv')
    outpath_gg_acute_m = os.path.join(path, 'gg acute male cell density.csv')
    outpath_aa_chronic_m = os.path.join(path, 'aa chronic male cell density.csv')
    outpath_gg_chronic_m = os.path.join(path, 'gg chronic male cell density.csv')
    outpath_aa_acute_f = os.path.join(path, 'aa acute female cell density.csv')
    outpath_gg_acute_f = os.path.join(path, 'gg acute female cell density.csv')
    outpath_aa_chronic_f = os.path.join(path, 'aa chronic female cell density.csv')
    outpath_gg_chronic_f = os.path.join(path, 'gg chronic female cell density.csv')

    outpath_aa_acute_m_corr = os.path.join(path2, 'corr aa acute male cell density.csv')
    outpath_gg_acute_m_corr = os.path.join(path2, 'corr gg acute male cell density.csv')
    outpath_aa_chronic_m_corr = os.path.join(path2, 'corr aa chronic male cell density.csv')
    outpath_gg_chronic_m_corr = os.path.join(path2, 'corr gg chronic male cell density.csv')
    outpath_aa_acute_f_corr = os.path.join(path2, 'corr aa acute female cell density.csv')
    outpath_gg_acute_f_corr = os.path.join(path2, 'corr gg acute female cell density.csv')
    outpath_aa_chronic_f_corr = os.path.join(path2, 'corr aa chronic female cell density.csv')
    outpath_gg_chronic_f_corr = os.path.join(path2, 'corr gg chronic female cell density.csv')
    roi_list = ['VTA', 'ACB', 'ACAd', 'ACAv', 'AId', 'AIv', 'PL',
                'ILA', 'BLA', 'BST', 'CEA', 'CLA', 'CP', 'DG', 'GPe','GPi', 'LH', 'LHA', 'MH', 'PAG', 'PALv', 'PVT', 'SNc',
                'SNr']
    roi_list1 = list(roi_list)
    roi_list1.remove('SNc')
    cortex_regions = ['ACAd', 'ACAv', 'AId', 'AIv', 'CLA', 'ILA', 'PL']
    striatum_regions = ['ACB', 'CP']
    pallidum_regions = ['BST', 'GPe', 'GPi', 'PALv']
    amygdala_regions = ['BLA', 'CEA']
    thalamus_regions = ['LH', 'MH', 'PVT']
    midbrain_regions = ['PAG', 'SNc', 'SNr', 'VTA']
    Groups_regions = ['cortex', 'striatum', 'pallidum', 'amygdala', 'thalamus', 'midbrain']
    roi_color = {'VTA': "darkcyan", 'ACB': "chocolate", 'ACAd': "dodgerblue", 'ACAv': "dodgerblue", 'AId': "dodgerblue",
                 'AIv': "dodgerblue", 'PL': "dodgerblue",
                 'ILA': "dodgerblue", 'BLA': "crimson", 'BST': "violet", 'CEA': "crimson", 'CLA': "dodgerblue",
                 'CP': "chocolate", 'DG': "gold", 'GPe': "violet", 'GPi': "violet", 'LH': "orange", 'LHA': "gold",
                 'MH': "orange", 'PAG': "darkcyan", 'PALv': "violet", 'PVT': "orange",
                 'SNc': "darkcyan", 'SNr': "darkcyan"}
    roi_list.sort()
    roi_list1.sort()
    roi_color = dict(sorted(roi_color.items()))
    #print(roi_color)


    num_bootstrap_size = 8
    '''
    for (root, dirs, file) in os.walk(path):
        for folder in dirs:
            if folder.__contains__("Analyzed Result"):
                regionid = folder.split(' ')[0]
                groupid = root.split('Final Overlays ')[1]
                print("Extracting Result in experimental group: ", groupid, ", within roi: ",regionid)
                subpath = os.path.join(root,folder)
                print("loading...  ",subpath)
                roilist = pd.DataFrame(columns=['ID','ROI'])
                for fs1 in os.listdir(subpath):
                    if fs1.__contains__("ROI Area"):
                        roifilepath = os.path.join(subpath, fs1)
                        roifile = pd.read_csv(roifilepath)
                        #print(roifile.iloc[0,1].astype('int'))
                        dd = pd.DataFrame([[fs1.split(' ROI')[0]] + [ roifile.iloc[0,1].astype('double')]],columns=['ID','ROI um^2'])
                        roilist = pd.concat([roilist, dd], ignore_index=True)
                print("Loading ROI Areas: ...")
                #print(roilist)
                roilist = roilist.sort_values(by=['ID'])
                roilist1 = pd.concat([roilist.iloc[:, 0].str.split(' ', expand=True), roilist.iloc[:, 1:]], axis=1)
                #print(roilist1)
                for fs in os.listdir(subpath):
                    if fs.__contains__("Summary"):
                        summaryfilepath = os.path.join(subpath,fs)
                        #print(summaryfilepath)
                        print("opening...  ",fs)
                        summary = pd.read_csv(summaryfilepath)
                        df = pd.DataFrame(summary)
                        df = df.sort_values(by = ['Slice'])
                        dt = pd.concat([df.iloc[:,0].str.split(' ', expand=True),df.iloc[:,1:]], axis=1)
                        dt = dt.dropna(axis = 0, how = 'all')
                        dt = dt.reset_index(drop=True)
                        dt.columns.values[1] = 'BrainID'
                        dt.columns.values[3] = 'SliceID'
                        dt = pd.concat([dt, roilist['ROI um^2']],axis = 1)
                        roilist2 = pd.DataFrame(roilist['ROI um^2'].div(1000000))
                        roilist2 = roilist2.rename(columns = {'ROI um^2':'ROI mm^2'})
                        dt = pd.concat([dt, roilist2], axis=1)
                        dt['#Cell Density/mm^2'] = dt['Count']/dt['ROI mm^2']
                        #print(dt)
                        dt['group id'] = groupid
                        dt['roi id'] = regionid

                        #print(dt)

                        Meta = pd.concat([Meta,dt], axis = 0, ignore_index= True)



    #print(Meta['#Cell Density/mm^2'])
    Meta = Meta.dropna(subset = ['BrainID'])
    Meta = Meta.astype({'BrainID': int, 'SliceID': int})
    print(Meta.dtypes)

    #Meta = Meta.iloc[].astype('int')
    
    bdids = pd.read_csv(badimagefile)
    bdids = pd.DataFrame(bdids)
    print(bdids.dtypes)
    c = pd.concat([Meta,bdids]).duplicated(subset = ['BrainID','SliceID','group id','roi id',])
    print('bad image id matching: ', c.value_counts([True]))
    Meta = pd.concat([Meta,bdids]).drop_duplicates(subset = ['group id', 'roi id', 'BrainID','SliceID'],keep=False)

    Meta = Meta.dropna(subset = ['ROI um^2','ROI mm^2','#Cell Density/mm^2','group id','roi id'], how='all').reset_index(drop = True)
    outpath = os.path.join(path,'Meta.csv')
    Meta.to_csv(outpath)
    Meta = Meta.loc[:,['group id','BrainID','roi id','Count','Total Area','Average Size','%Area','Perim.','ROI um^2','ROI mm^2','#Cell Density/mm^2']]
    avgMeta = Meta.groupby(['group id','BrainID','roi id'])['Count','Total Area','Average Size','%Area','Perim.','ROI um^2','ROI mm^2','#Cell Density/mm^2'].mean()

    label = pd.read_csv(labelfile)
    label = pd.DataFrame(label)
    avgMeta.to_csv(outpath1)
    avgMeta1 = pd.read_csv(outpath1)
    avgMeta1 = pd.DataFrame(avgMeta1)
    temproi = avgMeta1.loc[:,"roi id"]
    #print(temproi)
    avgMeta1 = pd.merge(avgMeta1,label, how= 'outer', on=['group id', 'BrainID'])
    avgMeta1.to_csv(outpath2)

    '''

    sigma_list = pd.DataFrame()
    omega_list = pd.DataFrame()
    omega_diff_list = pd.DataFrame()
    min_energy_list = pd.DataFrame()
    scale_list = pd.DataFrame()


for i in range(50):

    measure = '#Cell Density/mm^2'
    avgmeasure_path = os.path.join(path,'cfos percent change over baseline.csv')
    avgMeta1 = pd.read_csv(outpath2)
    avgMeta1 = pd.DataFrame(avgMeta1)
    avgmeasure = pd.DataFrame()

#AA SAL M
    aa_male_sal = avgMeta1.loc[(avgMeta1['Genotype'] == 'AA') & (avgMeta1['Sex'] == 'M') & (avgMeta1['Treatment'] == 'SAL')]
    avg_aa_male_sal = aa_male_sal.groupby(['roi id']).mean().apply(list)
    avg_aa_male_sal = pd.DataFrame(avg_aa_male_sal.loc[:, [measure]].T).reset_index(drop = True)
    avg_aa_male_sal = avg_aa_male_sal.replace(0, 1)
#GG SAL M
    gg_male_sal = avgMeta1.loc[(avgMeta1['Genotype'] == 'GG') & (avgMeta1['Sex'] == 'M') & (avgMeta1['Treatment'] == 'SAL')]
    avg_gg_male_sal = gg_male_sal.groupby(['roi id']).mean().apply(list)
    avg_gg_male_sal = pd.DataFrame(avg_gg_male_sal.loc[:, [measure]].T).reset_index(drop=True)
    avg_gg_male_sal = avg_gg_male_sal.replace(0, 1)

#AA Acute M
    aa_male_acute = avgMeta1.loc[(avgMeta1['Genotype'] == 'AA') & (avgMeta1['Sex'] == 'M') & (avgMeta1['Treatment'] == 'Acute') ]
    aa_male_acute = aa_male_acute.loc[:,['group id','BrainID','roi id', measure]]
    aa_male_acute = pd.pivot(aa_male_acute, columns = ['roi id'], index= ['group id','BrainID'], values= [measure])
    aa_male_acute = aa_male_acute.droplevel(0,axis = 1).reset_index(drop = True)

    aa_male_acute = bootstrap_dataframe(aa_male_acute, num_bootstrap_size)
    aa_male_acute.to_csv(outpath_aa_acute_m)
    #aa_male_acute = aa_male_acute.div(avg_aa_male_sal.iloc[0], level = 1)
    aa_male_acute = (aa_male_acute - avg_aa_male_sal.iloc[0]).div(avg_aa_male_sal.iloc[0], level=1)
    avgmeasure['AA Acute M'] = aa_male_acute.mean(axis=0)

    corr_aa_male_acute = aa_male_acute.corr()
    corr_aa_male_acute.to_csv(outpath_aa_acute_m_corr)


#GG Acute M
    gg_male_acute = avgMeta1.loc[(avgMeta1['Genotype'] == 'GG') & (avgMeta1['Sex'] == 'M') & (avgMeta1['Treatment'] == 'Acute') ]
    gg_male_acute = gg_male_acute.loc[:, ['group id','BrainID','roi id', measure]]
    gg_male_acute = pd.pivot(gg_male_acute, columns=['roi id'] , index= ['group id','BrainID'], values=[measure])
    gg_male_acute = gg_male_acute.droplevel(0, axis=1).reset_index(drop = True)

    gg_male_acute = bootstrap_dataframe(gg_male_acute, num_bootstrap_size)
    gg_male_acute.to_csv(outpath_gg_acute_m)
    #gg_male_acute = gg_male_acute.div(avg_gg_male_sal.iloc[0], level=1)
    gg_male_acute = (gg_male_acute - avg_gg_male_sal.iloc[0]).div(avg_gg_male_sal.iloc[0], level=1)
    avgmeasure['GG Acute M'] = gg_male_acute.mean(axis=0)

    corr_gg_male_acute = gg_male_acute.corr()
    corr_gg_male_acute.to_csv(outpath_gg_acute_m_corr)

#AA Chronic M
    aa_male_chronic = avgMeta1.loc[(avgMeta1['Genotype'] == 'AA') & (avgMeta1['Sex'] == 'M') & (avgMeta1['Treatment'] == 'Chronic') ]
    aa_male_chronic = aa_male_chronic.loc[:, ['group id','BrainID','roi id', measure]]
    aa_male_chronic = pd.pivot(aa_male_chronic, columns=['roi id'], index= ['group id','BrainID'], values=[measure])
    aa_male_chronic = aa_male_chronic.droplevel(0, axis=1).reset_index(drop=True)

    aa_male_chronic = bootstrap_dataframe(aa_male_chronic, num_bootstrap_size)
    aa_male_chronic.to_csv(outpath_aa_chronic_m)
    #aa_male_chronic = aa_male_chronic.div(avg_aa_male_sal.iloc[0], level=1)
    aa_male_chronic = (aa_male_chronic - avg_aa_male_sal.iloc[0]).div(avg_aa_male_sal.iloc[0], level=1)
    avgmeasure['AA Chronic M'] = aa_male_chronic.mean(axis=0)

    corr_aa_male_chronic = aa_male_chronic.corr()
    corr_aa_male_chronic.to_csv(outpath_aa_chronic_m_corr)

#GG Chronic M
    gg_male_chronic = avgMeta1.loc[(avgMeta1['Genotype'] == 'GG') & (avgMeta1['Sex'] == 'M') & (avgMeta1['Treatment'] == 'Chronic') ]
    gg_male_chronic = gg_male_chronic.loc[:, ['group id','BrainID','roi id', measure]]
    gg_male_chronic = pd.pivot(gg_male_chronic, columns=['roi id'], index= ['group id','BrainID'], values=[measure])
    gg_male_chronic = gg_male_chronic.droplevel(0, axis=1).reset_index(drop=True)

    gg_male_chronic = bootstrap_dataframe(gg_male_chronic, num_bootstrap_size)
    gg_male_chronic.to_csv(outpath_gg_chronic_m)
    #gg_male_chronic = gg_male_chronic.div(avg_gg_male_sal.iloc[0], level=1)
    gg_male_chronic = (gg_male_chronic - avg_gg_male_sal.iloc[0]).div(avg_gg_male_sal.iloc[0], level=1)
    avgmeasure['GG Chronic M'] = gg_male_chronic.mean(axis=0)

    corr_gg_male_chronic = gg_male_chronic.corr()
    corr_gg_male_chronic.to_csv(outpath_gg_chronic_m_corr)

#AA SAL F
    aa_female_sal = avgMeta1.loc[(avgMeta1['Genotype'] == 'AA') & (avgMeta1['Sex'] == 'F') & (avgMeta1['Treatment'] == 'SAL')]
    avg_aa_female_sal = aa_female_sal.groupby(['roi id']).mean().apply(list)
    avg_aa_female_sal = pd.DataFrame(avg_aa_female_sal.loc[:, [measure]].T).reset_index(drop=True)
    avg_aa_female_sal = avg_aa_female_sal.replace(0, 1)

#GG SAL F
    gg_female_sal = avgMeta1.loc[(avgMeta1['Genotype'] == 'GG') & (avgMeta1['Sex'] == 'F') & (avgMeta1['Treatment'] == 'SAL')]
    avg_gg_female_sal = gg_female_sal.groupby(['roi id']).mean().apply(list)
    avg_gg_female_sal = pd.DataFrame(avg_gg_female_sal.loc[:, [measure]].T).reset_index(drop=True)
    avg_gg_female_sal = avg_gg_female_sal.replace(0, 1)

#AA Acute F
    aa_female_acute = avgMeta1.loc[(avgMeta1['Genotype'] == 'AA') & (avgMeta1['Sex'] == 'F') & (avgMeta1['Treatment'] == 'Acute')]
    aa_female_acute = aa_female_acute.loc[:, ['group id','BrainID','roi id', measure]]
    aa_female_acute = pd.pivot(aa_female_acute, columns=['roi id'], index= ['group id','BrainID'], values=[measure])
    aa_female_acute = aa_female_acute.droplevel(0, axis=1).reset_index(drop=True)

    aa_female_acute = bootstrap_dataframe(aa_female_acute, num_bootstrap_size)
    aa_female_acute.to_csv(outpath_aa_acute_f)
    #aa_female_acute = aa_female_acute.div(avg_aa_female_sal.iloc[0], level=1)
    aa_female_acute = (aa_female_acute - avg_aa_female_sal.iloc[0]).div(avg_aa_female_sal.iloc[0], level=1)
    avgmeasure['AA Acute F'] = aa_female_acute.mean(axis=0)

    corr_aa_female_acute = aa_female_acute.corr()
    corr_aa_female_acute.to_csv(outpath_aa_acute_f_corr)

#GG Acute F
    gg_female_acute = avgMeta1.loc[(avgMeta1['Genotype'] == 'GG') & (avgMeta1['Sex'] == 'F') & (avgMeta1['Treatment'] == 'Acute')]
    gg_female_acute = gg_female_acute.loc[:, ['group id','BrainID','roi id', measure]]
    gg_female_acute = pd.pivot(gg_female_acute, columns=['roi id'], index= ['group id','BrainID'], values=[measure])
    gg_female_acute = gg_female_acute.droplevel(0, axis=1).reset_index(drop=True)

    gg_female_acute = bootstrap_dataframe(gg_female_acute, num_bootstrap_size)
    gg_female_acute.to_csv(outpath_gg_acute_f)
    #gg_female_acute = gg_female_acute.div(avg_gg_female_sal.iloc[0], level=1)
    gg_female_acute = (gg_female_acute - avg_gg_female_sal.iloc[0]).div(avg_gg_female_sal.iloc[0], level=1)
    avgmeasure['GG Acute F'] = gg_female_acute.mean(axis=0)

    corr_gg_female_acute = gg_female_acute.corr()
    corr_gg_female_acute.to_csv(outpath_gg_acute_f_corr)

#AA Chronic F
    aa_female_chronic = avgMeta1.loc[(avgMeta1['Genotype'] == 'AA') & (avgMeta1['Sex'] == 'F') & (avgMeta1['Treatment'] == 'Chronic')]
    aa_female_chronic = aa_female_chronic.loc[:, ['group id','BrainID','roi id', measure]]
    aa_female_chronic = pd.pivot(aa_female_chronic, columns=['roi id'], index= ['group id','BrainID'], values=[measure])
    aa_female_chronic = aa_female_chronic.droplevel(0, axis=1).reset_index(drop=True)

    aa_female_chronic = bootstrap_dataframe(aa_female_chronic, num_bootstrap_size)
    aa_female_chronic.to_csv(outpath_aa_chronic_f)
    #aa_female_chronic = aa_female_chronic.div(avg_aa_female_sal.iloc[0], level=1)
    aa_female_chronic = (aa_female_chronic - avg_aa_female_sal.iloc[0]).div(avg_aa_female_sal.iloc[0], level=1)
    avgmeasure['AA Chronic F'] = aa_female_chronic.mean(axis=0)

    corr_aa_female_chronic = aa_female_chronic.corr()
    corr_aa_female_chronic.to_csv(outpath_aa_chronic_f_corr)

## GG Chronic F
    gg_female_chronic = avgMeta1.loc[(avgMeta1['Genotype'] == 'GG') & (avgMeta1['Sex'] == 'F') & (avgMeta1['Treatment'] == 'Chronic')]
    gg_female_chronic = gg_female_chronic.loc[:, ['group id','BrainID','roi id', measure]]
    gg_female_chronic = pd.pivot(gg_female_chronic, columns=['roi id'], index= ['group id','BrainID'], values=[measure])
    gg_female_chronic = gg_female_chronic.droplevel(0, axis=1).reset_index(drop=True)

    gg_female_chronic = bootstrap_dataframe(gg_female_chronic, num_bootstrap_size)
    gg_female_chronic.to_csv(outpath_gg_chronic_f)
    #gg_female_chronic = gg_female_chronic.div(avg_gg_female_sal.iloc[0], level=1)
    gg_female_chronic = (gg_female_chronic - avg_gg_female_sal.iloc[0]).div(avg_gg_female_sal.iloc[0], level=1)
    avgmeasure['GG Chronic F'] = gg_female_chronic.mean(axis=0)

    corr_gg_female_chronic = gg_female_chronic.corr()
    corr_gg_female_chronic.to_csv(outpath_gg_chronic_f_corr)


    print(avgmeasure)
    avgmeasure.to_csv(avgmeasure_path)

##SAL
    sal_group = avgMeta1.loc[(avgMeta1['Treatment'] == 'SAL')]
    avg_sal_group = sal_group.groupby(['roi id']).mean().apply(list)
    avg_sal_group = pd.DataFrame(avg_sal_group.loc[:, ['#Cell Density/mm^2']].T).reset_index(drop=True)
    avg_sal_group = avg_sal_group.replace(0, 1)

## Acute
    acute_group = avgMeta1.loc[(avgMeta1['Treatment'] == 'Acute')]
    acute_group = acute_group.loc[:, ['group id','BrainID','roi id', '#Cell Density/mm^2']]
    acute_group = pd.pivot(acute_group, columns=['roi id'], index=['group id', 'BrainID'],
                                 values=['#Cell Density/mm^2'])
    acute_group = acute_group.droplevel(0, axis=1).reset_index(drop=True)
    acute_group = (acute_group - avg_sal_group.iloc[0]).div(avg_sal_group.iloc[0], level=1)
    corr_acute_group = acute_group.corr()


## Chronic
    chronic_group = avgMeta1.loc[(avgMeta1['Treatment'] == 'Chronic')]
    chronic_group = chronic_group.loc[:, ['group id', 'BrainID', 'roi id', '#Cell Density/mm^2']]
    chronic_group = pd.pivot(chronic_group, columns=['roi id'], index=['group id', 'BrainID'],
                           values=['#Cell Density/mm^2'])
    chronic_group = chronic_group.droplevel(0, axis=1).reset_index(drop=True)
    chronic_group = (chronic_group - avg_sal_group.iloc[0]).div(avg_sal_group.iloc[0], level=1)
    corr_chronic_group = chronic_group.corr()



#scale free network test
    '''
    scale_free_path = os.path.join(path_analysis,"scale_free.csv")
    test_scale = [test_scale_free(generate_distance_network(corr_aa_male_acute, roi_list), 'AA Male Acute'),
    test_scale_free(generate_distance_network(corr_gg_male_acute, roi_list),'GG Male Acute'),
    test_scale_free(generate_distance_network(corr_aa_male_chronic, roi_list),'AA Male Chronic'),
    test_scale_free(generate_distance_network(corr_gg_male_chronic, roi_list),'GG Male Chronic'),

    test_scale_free(generate_distance_network(corr_aa_female_acute, roi_list), 'AA Female Acute'),
    test_scale_free(generate_distance_network(corr_gg_female_acute, roi_list), 'GG Female Acute'),
    test_scale_free(generate_distance_network(corr_aa_female_chronic, roi_list), 'AA Female Chronic'),
    test_scale_free(generate_distance_network(corr_gg_female_chronic, roi_list), 'GG Female Chronic')]
    test_scale = pd.DataFrame(test_scale).T
    print(test_scale)

    scale_list = pd.concat([scale_list,test_scale],axis = 0)
    scale_list.to_csv(scale_free_path)
    '''
    '''
    #preprocessing receptor density
    receptor_density = pd.read_csv("E:\TRL\A112G\ISH data from allen\Count Summary\Summary of all regions.csv")
    receptor_density = pd.DataFrame(receptor_density)
    receptor_density = receptor_density.drop(columns = 'SNc')
    receptor_density = receptor_density.iloc[0:3,:]
    receptor_density = receptor_density.iloc[:,1:]
    print(receptor_density)
    
    #oprm1_receptor_density = oprm1_receptor_density
    for i in range(receptor_density.shape[0]):
        receptor_density.iloc[i,:] = (receptor_density.iloc[i,:] - receptor_density.iloc[i,:].min()) / (receptor_density.iloc[i,:].max() - receptor_density.iloc[i,:].min())
    print(receptor_density)
    '''

    #print(oprd1_receptor_density)
    #print(oprk1_receptor_density)
    '''
    # create PCA analysis plot
    #pca_t = str(i)
    generate_PCA(aa_male_acute, gg_male_acute, aa_male_chronic, gg_male_chronic,
                 aa_female_acute, gg_female_acute, aa_female_chronic, gg_female_chronic,
                 "AA vs. GG Male Acute", "AA vs. GG Male Chronic", "AA vs. GG Female Acute", "AA vs. GG Female Chronic",
                 path_analysis, title='by Genotype and treatment', colors= ['cornflowerblue','darkorange', 'blue','orange'] ,labels=['AA Acute', 'GG Acute', 'AA Chronic', 'GG Chronic'], measure='Genotype Treatment', method='two')
    
    pca_t = str(i)
    path_pca = os.path.join(path_analysis,pca_t)
    print('Generate PCA Plot # ', i)
    print('//============================//')
    print('//============================//')
    print('//============================//')
    print('//============================//')
    generate_PCA(aa_male_acute, gg_male_acute, aa_male_chronic, gg_male_chronic,
                 aa_female_acute, gg_female_acute, aa_female_chronic, gg_female_chronic,
                 "AA vs. GG Male Acute", "AA vs. GG Male Chronic", "AA vs. GG Female Acute", "AA vs. GG Female Chronic",
                 path_analysis, title='by Genotype' + str(i), labels=['AA', 'GG'], measure='Genotype')
    
    # dont run together with previous
    generate_PCA(aa_male_acute, aa_male_chronic, gg_male_acute, gg_male_chronic,
                 aa_female_acute, aa_female_chronic, gg_female_acute, gg_female_chronic,
                 "AA Male Acute vs. Chronic", "GG Male Acute vs. Chronic", "AA Female Acute vs. Chronic",
                 "GG Female Acute vs. Chronic",
                 path_analysis, title='by Treatment', labels=['Acute', 'Chronic'], measure='Treatment')

    '''
    # calculating weighted by receptor density, mu delta kappa data concatenate by rows
    min_energy_boot_path = os.path.join(path_analysis, 'Minimum Energy boot correlation.csv')
    min_control_energy_aa_m, min_aa_m = min_control_energy(outpath_aa_acute_m, outpath_aa_chronic_m, roi_list1,
                                                           tracker=i, method='mean',
                                                           title='AA Male')
    min_control_energy_gg_m, min_gg_m = min_control_energy(outpath_gg_acute_m, outpath_gg_chronic_m, roi_list1,
                                                           tracker=i, method='mean',
                                                           title='GG Male')
    min_control_energy_aa_f, min_aa_f = min_control_energy(outpath_aa_acute_f, outpath_aa_chronic_f, roi_list1,
                                                           tracker=i, method='mean',
                                                           title='AA Female')
    min_control_energy_gg_f, min_gg_f = min_control_energy(outpath_gg_acute_f, outpath_gg_chronic_f, roi_list1,
                                                           tracker=i, method='mean',
                                                           title='GG Female')

    min_boot = [min_aa_m, min_gg_m, min_aa_f, min_gg_f]
    min_boot = pd.DataFrame(min_boot)
    min_energy_list = pd.concat([min_energy_list, min_boot], axis=1)

    min_energy_list.to_csv(min_energy_boot_path)


    # small world coefficient
    print('calculating small world sigma round: ', i)
    print('//-------------------------------//')
    print('//-------------------------------//')
    print('//-------------------------------//')
    print('//-------------------------------//')
    print('//-------------------------------//')
    sigma_list_path = os.path.join(path_analysis, 'small world sigma correlation.csv')

    sigma_aa_male_acute = nx.sigma(generate_distance_network(corr_aa_male_acute, roi_list), niter=23)
    sigma_gg_male_acute = nx.sigma(generate_distance_network(corr_gg_male_acute, roi_list), niter=23)
    sigma_aa_male_chronic = nx.sigma(generate_distance_network(corr_aa_male_chronic, roi_list), niter=23)
    sigma_gg_male_chronic = nx.sigma(generate_distance_network(corr_gg_male_chronic, roi_list), niter=23)

    sigma_aa_female_acute = nx.sigma(generate_distance_network(corr_aa_female_acute, roi_list), niter=23)
    sigma_gg_female_acute = nx.sigma(generate_distance_network(corr_gg_female_acute, roi_list), niter=23)
    sigma_aa_female_chronic = nx.sigma(generate_distance_network(corr_aa_female_chronic, roi_list), niter=23)
    sigma_gg_female_chronic = nx.sigma(generate_distance_network(corr_gg_female_chronic, roi_list), niter=23)
    #below for calculating sigma in each state
    #sigma = [sigma_aa_male_acute, sigma_gg_male_acute, sigma_aa_male_chronic, sigma_gg_male_chronic,sigma_aa_female_acute, sigma_gg_female_acute, sigma_aa_female_chronic, sigma_gg_female_chronic]
    #below for calculating sigma change during dependency formation to correlate with network control theory
    sigma = [sigma_aa_male_chronic - sigma_aa_male_acute, sigma_gg_male_chronic - sigma_gg_male_acute, sigma_aa_female_chronic - sigma_aa_female_acute, sigma_gg_female_chronic - sigma_gg_female_acute]
    sigma = pd.DataFrame(sigma)
    sigma_list = pd.concat([sigma_list, sigma], axis=1)
    sigma_list.to_csv(sigma_list_path)
    print('Calcualting Small Worldness Coefficient sigma, this may take 10-20 min: ', sigma)
    '''
    
    # small world coefficient
    print('calculating small world omega round: ', i)
    print('//-------------------------------//')
    print('//-------------------------------//')
    print('//-------------------------------//')
    print('//-------------------------------//')
    print('//-------------------------------//')
    omega_list_path = os.path.join(path_analysis, 'small world omega.csv')
    omega_diff_list_path = os.path.join(path_analysis, 'small world omega diff.csv')

    omega_aa_male_acute = nx.omega(generate_distance_network(corr_aa_male_acute, roi_list), niter=23)

    omega_gg_male_acute = nx.omega(generate_distance_network(corr_gg_male_acute, roi_list), niter=23)

    omega_aa_male_chronic = nx.omega(generate_distance_network(corr_aa_male_chronic, roi_list), niter=23)

    omega_gg_male_chronic = nx.omega(generate_distance_network(corr_gg_male_chronic, roi_list), niter=23)


    omega_aa_female_acute = nx.omega(generate_distance_network(corr_aa_female_acute, roi_list), niter=23)

    omega_gg_female_acute = nx.omega(generate_distance_network(corr_gg_female_acute, roi_list), niter=23)

    omega_aa_female_chronic = nx.omega(generate_distance_network(corr_aa_female_chronic, roi_list), niter=23)

    omega_gg_female_chronic = nx.omega(generate_distance_network(corr_gg_female_chronic, roi_list), niter=23)

    omega = [omega_aa_male_acute,omega_gg_male_acute,omega_aa_male_chronic,omega_gg_male_chronic,omega_aa_female_acute,omega_gg_female_acute,omega_aa_female_chronic,omega_gg_female_chronic]
    omega = pd.DataFrame(omega).T
    omega_diff = [omega_aa_male_chronic - omega_aa_male_acute, omega_gg_male_chronic - omega_gg_male_acute,
                  omega_aa_female_chronic - omega_aa_female_acute, omega_gg_female_chronic - omega_gg_female_acute]
    omega_diff = pd.DataFrame(omega_diff).T
    print(omega_list)

    omega_list = pd.concat([omega_list, omega], axis=0)
    omega_diff_list = pd.concat([omega_diff_list,omega_diff],axis=0)
    omega_list.to_csv(omega_list_path)
    omega_diff_list.to_csv(omega_diff_list_path)
    #Scale Free network Test





    
    # get network clustering
    clustering_path = os.path.join(path_analysis, "Clustering Coefficient\clustering.csv")
    clustering_acute_group = get_clustering(corr_acute_group, roi_list, "acute morphine")
    clustering_chronic_group = get_clustering(corr_chronic_group, roi_list, "chronic morphine")
    clustering_aa_acute_male = get_clustering(corr_aa_male_acute, roi_list, "aa acute male")
    clustering_gg_acute_male = get_clustering(corr_gg_male_acute, roi_list, "gg acute male")
    clustering_aa_chronic_male = get_clustering(corr_aa_male_chronic, roi_list, "aa chronic male")
    clustering_gg_chronic_male = get_clustering(corr_gg_male_chronic, roi_list, "gg chronic male")

    clustering_aa_acute_female = get_clustering(corr_aa_female_acute, roi_list, "aa acute female")
    clustering_gg_acute_female = get_clustering(corr_gg_female_acute, roi_list, "gg acute female")
    clustering_aa_chronic_female = get_clustering(corr_aa_female_chronic, roi_list, "aa chronic female")
    clustering_gg_chronic_female = get_clustering(corr_gg_female_chronic, roi_list, "gg chronic female")
    clustering = pd.concat(
        [clustering_acute_group, clustering_chronic_group, clustering_aa_acute_male,
         clustering_gg_acute_male,
         clustering_aa_chronic_male, clustering_gg_chronic_male, clustering_aa_acute_female,
         clustering_gg_acute_female,
         clustering_aa_chronic_female, clustering_gg_chronic_female], axis=1)
    clustering.to_csv(clustering_path)


    
    #get network effective size
    effective_size_path = os.path.join(path_analysis, "Structural Holes\effective size.csv")
    #effective_size_acute_group = get_effective_size(corr_acute_group, roi_list, "acute morphine")
    #effective_size_chronic_group = get_effective_size(corr_chronic_group, roi_list, "chronic morphine")
    effective_size_aa_acute_male = get_effective_size(corr_aa_male_acute, roi_list, "aa","acute","male")
    effective_size_gg_acute_male = get_effective_size(corr_gg_male_acute, roi_list, "gg","acute","male")
    effective_size_aa_chronic_male = get_effective_size(corr_aa_male_chronic, roi_list, "aa","chronic","male")
    effective_size_gg_chronic_male = get_effective_size(corr_gg_male_chronic, roi_list, "gg","chronic","male")

    effective_size_aa_acute_female = get_effective_size(corr_aa_female_acute, roi_list, "aa","acute","female")
    effective_size_gg_acute_female = get_effective_size(corr_gg_female_acute, roi_list, "gg","acute","female")
    effective_size_aa_chronic_female = get_effective_size(corr_aa_female_chronic, roi_list, "aa","chronic","female")
    effective_size_gg_chronic_female = get_effective_size(corr_gg_female_chronic, roi_list, "gg","chronic","female")
    effective_size = pd.concat(
        [effective_size_aa_acute_male, effective_size_gg_acute_male,
         effective_size_aa_chronic_male, effective_size_gg_chronic_male, effective_size_aa_acute_female, effective_size_gg_acute_female,
         effective_size_aa_chronic_female, effective_size_gg_chronic_female], axis=1)
    effective_size.to_csv(effective_size_path)

    print(effective_size)
    #maov = MANOVA.from_formula('effective_size + ROI ~ Genotype + Treatment + Sex', data=effective_size)
    #print(maov.mv_test())

    
    
    #get network constraint
    
    constraint_path = os.path.join(path_analysis,"Structural Holes\contraint.csv")
    constraint_acute_group = get_constraint(corr_acute_group,roi_list,"acute morphine")
    constraint_chronic_group = get_constraint(corr_chronic_group,roi_list,"chronic morphine")
    constraint_aa_acute_male = get_constraint(corr_aa_male_acute,roi_list,"aa acute male")
    constraint_gg_acute_male = get_constraint(corr_gg_male_acute,roi_list,"gg acute male")
    constraint_aa_chronic_male = get_constraint(corr_aa_male_chronic, roi_list, "aa chronic male")
    constraint_gg_chronic_male = get_constraint(corr_gg_male_chronic, roi_list, "gg chronic male")

    constraint_aa_acute_female = get_constraint(corr_aa_female_acute, roi_list, "aa acute female")
    constraint_gg_acute_female = get_constraint(corr_gg_female_acute, roi_list, "gg acute female")
    constraint_aa_chronic_female = get_constraint(corr_aa_female_chronic, roi_list, "aa chronic female")
    constraint_gg_chronic_female = get_constraint(corr_gg_female_chronic, roi_list, "gg chronic female")
    constraint = pd.concat([constraint_acute_group,constraint_chronic_group, constraint_aa_acute_male,constraint_gg_acute_male,
                            constraint_aa_chronic_male,constraint_gg_chronic_male,constraint_aa_acute_female,constraint_gg_acute_female,
                            constraint_aa_chronic_female,constraint_gg_chronic_female], axis = 1)
    constraint.to_csv(constraint_path)
    
    #generate connectivity map
    outpath_connectivity = "D:\Count Summary\Connectivity Graph"

    #aa acute male
    outpath_connectivity_aa_acute_male = os.path.join(outpath_connectivity,"AA Acute Male connectivity network.tif")
    fig = plt.figure()
    plt.title("AA Male Acute Morphine mediated cFos Connectivity Network", fontsize = 20)
    G = generate_network(corr_aa_male_acute, roi_list, roi_color)
    plt.show()
    fig.savefig(outpath_connectivity_aa_acute_male, transparent= True, dpi = 500)
    # gg acute male
    outpath_connectivity_gg_acute_male = os.path.join(outpath_connectivity, "GG Acute Male connectivity network.tif")
    fig = plt.figure()
    plt.title("GG Male Acute Morphine mediated cFos Connectivity Network", fontsize = 20)
    G = generate_network(corr_gg_male_acute, roi_list, roi_color)
    plt.show()
    fig.savefig(outpath_connectivity_gg_acute_male, transparent=True, dpi=500)
    # aa acute female
    outpath_connectivity_aa_acute_female = os.path.join(outpath_connectivity, "AA Acute Female connectivity network.tif")
    fig = plt.figure()
    plt.title("AA Female Acute Morphine mediated cFos Connectivity Network", fontsize = 20)
    G = generate_network(corr_aa_female_acute, roi_list, roi_color)
    plt.show()
    fig.savefig(outpath_connectivity_aa_acute_female, transparent=True, dpi=500)
    # gg acute female
    outpath_connectivity_gg_acute_female = os.path.join(outpath_connectivity, "GG Acute Female connectivity network.tif")
    fig = plt.figure()
    plt.title("GG Female Acute Morphine mediated cFos Connectivity Network", fontsize = 20)
    G = generate_network(corr_gg_female_acute, roi_list, roi_color)
    plt.show()
    fig.savefig(outpath_connectivity_gg_acute_female, transparent=True, dpi=500)

    # aa chronic male
    outpath_connectivity_aa_chronic_male = os.path.join(outpath_connectivity, "AA Chronic Male connectivity network.tif")
    fig = plt.figure()
    plt.title("AA Male Chronic Morphine mediated cFos Connectivity Network", fontsize = 20)
    G = generate_network(corr_aa_male_chronic, roi_list, roi_color)
    plt.show()
    fig.savefig(outpath_connectivity_aa_chronic_male, transparent=True, dpi=500)
    # gg chronic male
    outpath_connectivity_gg_chronic_male = os.path.join(outpath_connectivity, "GG Chronic Male connectivity network.tif")
    fig = plt.figure()
    plt.title("GG Male Chronic Morphine mediated cFos Connectivity Network", fontsize = 20)
    G = generate_network(corr_gg_male_chronic, roi_list, roi_color)
    plt.show()
    fig.savefig(outpath_connectivity_gg_chronic_male, transparent=True, dpi=500)
    # aa chronic female
    outpath_connectivity_aa_chronic_female = os.path.join(outpath_connectivity,
                                                        "AA Chronic Female connectivity network.tif")
    fig = plt.figure()
    plt.title("AA Female Chronic Morphine mediated cFos Connectivity Network", fontsize = 20)
    G = generate_network(corr_aa_female_chronic, roi_list, roi_color)
    plt.show()
    fig.savefig(outpath_connectivity_aa_chronic_female, transparent=True, dpi=500)
    # gg chronic female
    outpath_connectivity_gg_chronic_female = os.path.join(outpath_connectivity,
                                                        "GG Chronic Female connectivity network.tif")
    fig = plt.figure()
    plt.title("GG Female Chronic Morphine mediated cFos Connectivity Network", fontsize = 20)
    G = generate_network(corr_gg_female_chronic, roi_list, roi_color)
    plt.show()
    fig.savefig(outpath_connectivity_gg_chronic_female, transparent=True, dpi=500)
    
    #classify nodes by their role in opioid dependency
    mode_classification_reinforcement_path = os.path.join(path_analysis,'mode classification reinforcement path.csv')
    prediction_list = pd.DataFrame()
    for j in range(50):
        Acute_G, Acute_G_nodes, Acute_G_selected = generate_network_role(corr_acute_group, roi_list, 'reinforcement')
        prediction_acute = node_classification.harmonic_function(Acute_G)
        prediction_acute = pd.DataFrame(prediction_acute, index=Acute_G_nodes, columns=['Acute'])
        for i in Acute_G_selected:
            prediction_acute.loc[i, 'Acute'] = '**' + prediction_acute.loc[i, 'Acute'] + '**'
        Chronic_G, Chronic_G_nodes, Chronic_G_selected = generate_network_role(corr_chronic_group, roi_list,
                                                                               'reinforcement')
        prediction_chronic = node_classification.harmonic_function(Chronic_G)
        prediction_chronic = pd.DataFrame(prediction_chronic, index=Chronic_G_nodes, columns=['Chronic'])
        for i in Chronic_G_selected:
            prediction_chronic.loc[i, 'Chronic'] = '**' + prediction_chronic.loc[i, 'Chronic'] + '**'
        prediction = pd.concat([prediction_acute, prediction_chronic], axis=1)
        #print(prediction)
        prediction_list = pd.concat([prediction_list,prediction], axis = 0)
    prediction_list = prediction_list.reset_index()
    prediction_list = prediction_list.sort_values(by= 'index',axis = 0)
    print(prediction_list)
    mode_prediciton_list_reinforcement = pd.DataFrame(prediction_list.groupby(['index'])['Acute','Chronic'].agg(pd.Series.mode))
    print(mode_prediciton_list_reinforcement)
    prediction_list.to_csv(outpath_node_classification1)
    mode_prediciton_list_reinforcement.to_csv(mode_classification_reinforcement_path)
    
    
    #random blunt 7 regions
    mode_classification_addiction_path = os.path.join(path_analysis, 'mode classification drug addiction.csv')
    prediction_list = pd.DataFrame()
    for j in range(50):
        Acute_G, Acute_G_nodes, Acute_G_selected = generate_network_role(corr_acute_group, roi_list, 'drug addiction')
        prediction_acute = node_classification.harmonic_function(Acute_G)
        prediction_acute = pd.DataFrame(prediction_acute, index=Acute_G_nodes, columns=['Acute'])
        for i in Acute_G_selected:
            prediction_acute.loc[i, 'Acute'] = '**' + prediction_acute.loc[i, 'Acute'] + '**'
        Chronic_G, Chronic_G_nodes, Chronic_G_selected = generate_network_role(corr_chronic_group, roi_list,
                                                                               'drug addiction')
        prediction_chronic = node_classification.harmonic_function(Chronic_G)
        prediction_chronic = pd.DataFrame(prediction_chronic, index=Chronic_G_nodes, columns=['Chronic'])
        for i in Chronic_G_selected:
            prediction_chronic.loc[i, 'Chronic'] = '**' + prediction_chronic.loc[i, 'Chronic'] + '**'
        prediction = pd.concat([prediction_acute, prediction_chronic], axis=1)
        # print(prediction)
        prediction_list = pd.concat([prediction_list, prediction], axis=0)
    prediction_list = prediction_list.reset_index()
    prediction_list = prediction_list.sort_values(by='index', axis=0)
    print(prediction_list)
    mode_prediciton_list_reinforcement = pd.DataFrame(
        prediction_list.groupby(['index'])['Acute', 'Chronic'].agg(pd.Series.mode))
    print(mode_prediciton_list_reinforcement)
    prediction_list.to_csv(outpath_node_classification2)
    mode_prediciton_list_reinforcement.to_csv(mode_classification_addiction_path)

    
    #compute graph edit distance between networks
    time_out_graph_edit = 120

    graph_edit_distance_aa_acute = nx.graph_edit_distance(generate_distance_network(corr_aa_male_acute, roi_list), generate_distance_network(corr_aa_female_acute, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_gg_acute = nx.graph_edit_distance(generate_distance_network(corr_gg_male_acute, roi_list),generate_distance_network(corr_gg_female_acute, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_aa_chronic = nx.graph_edit_distance(generate_distance_network(corr_aa_male_chronic, roi_list),
                                    generate_distance_network(corr_aa_female_chronic, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_gg_chronic = nx.graph_edit_distance(generate_distance_network(corr_gg_male_chronic, roi_list),
                                    generate_distance_network(corr_gg_female_chronic, roi_list), timeout= time_out_graph_edit)

    graph_edit_distance_sex = [graph_edit_distance_aa_acute, graph_edit_distance_gg_acute, graph_edit_distance_aa_chronic, graph_edit_distance_gg_chronic]
    print('graph edit path between sex: ', graph_edit_distance_sex)


    graph_edit_distance_aa_male = nx.graph_edit_distance(generate_distance_network(corr_aa_male_acute, roi_list), generate_distance_network(corr_aa_male_chronic, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_gg_male = nx.graph_edit_distance(generate_distance_network(corr_gg_male_acute, roi_list),generate_distance_network(corr_gg_male_chronic, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_aa_female = nx.graph_edit_distance(generate_distance_network(corr_aa_female_acute, roi_list),
                                    generate_distance_network(corr_aa_female_chronic, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_gg_female = nx.graph_edit_distance(generate_distance_network(corr_gg_female_acute, roi_list),
                                    generate_distance_network(corr_gg_female_chronic, roi_list), timeout= time_out_graph_edit)
    
    graph_edit_distance_treatment = [graph_edit_distance_aa_male, graph_edit_distance_gg_male, graph_edit_distance_aa_female, graph_edit_distance_gg_female]
    print('graph edit path between treatment: ',graph_edit_distance_treatment)
    
    

    graph_edit_distance_acute_male = nx.graph_edit_distance(generate_distance_network(corr_aa_male_acute, roi_list),
                                                                  generate_distance_network(corr_gg_male_acute, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_chronic_male = nx.graph_edit_distance(generate_distance_network(corr_aa_male_chronic, roi_list),
                                                                  generate_distance_network(corr_gg_male_chronic, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_acute_female = nx.graph_edit_distance( generate_distance_network(corr_aa_female_acute, roi_list),generate_distance_network(corr_gg_female_acute, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_chronic_female = nx.graph_edit_distance(generate_distance_network(corr_aa_female_chronic, roi_list),generate_distance_network(corr_gg_female_chronic, roi_list), timeout= time_out_graph_edit)
    graph_edit_distance_genotype = [graph_edit_distance_acute_male, graph_edit_distance_chronic_male,
                                     graph_edit_distance_acute_female, graph_edit_distance_chronic_female]
    count = 0
    print('graph edit path between genotype: ',graph_edit_distance_genotype)
    
    #get group closeness
    closeness_path = os.path.join(path_analysis, "Group Closeness\group closeness.csv")
    close1 = group_closeness_centrality(corr_aa_male_acute, roi_list, 'aa_male_acute')
    close2 = group_closeness_centrality(corr_gg_male_acute, roi_list, 'gg_male_acute')
    close3 = group_closeness_centrality(corr_aa_male_chronic, roi_list, 'aa_male_chronic')
    close4 = group_closeness_centrality(corr_gg_male_chronic, roi_list, 'gg_male_chronic')

    close5 = group_closeness_centrality(corr_aa_female_acute, roi_list, 'aa_female_acute')
    close6 = group_closeness_centrality(corr_gg_female_acute, roi_list, 'gg_female_acute')
    close7 = group_closeness_centrality(corr_aa_female_chronic, roi_list, 'aa_female_chronic')
    close8 = group_closeness_centrality(corr_gg_female_chronic, roi_list, 'gg_female_chronic')
    closeness_M = pd.concat([close1, close2, close3, close4], axis = 1)
    closeness_F = pd.concat([close5, close6, close7, close8], axis=1)

    closeness = pd.concat([closeness_M,closeness_F], axis=1)
    closeness.to_csv(closeness_path)
    plot_group_closeness(closeness_M,'Male', path_analysis)
    plot_group_closeness(closeness_F, 'Female', path_analysis)
    
    
    #get commmunicability betweenness centrality -https://doi.org/10.48550/arXiv.0905.4102
        #for male
    comm_between_centrality1 = nx.communicability_betweenness_centrality(generate_distance_network(corr_aa_male_acute, roi_list))
    comm_between_centrality1 = pd.DataFrame.from_dict(comm_between_centrality1, orient='index',columns=['aa_male_acute'])
    comm_between_centrality2 = nx.communicability_betweenness_centrality(
        generate_distance_network(corr_gg_male_acute, roi_list))
    comm_between_centrality2 = pd.DataFrame.from_dict(comm_between_centrality2, orient='index',
                                                      columns=['gg_male_acute'])
    comm_between_centrality3 = nx.communicability_betweenness_centrality(
        generate_distance_network(corr_aa_male_chronic, roi_list))
    comm_between_centrality3 = pd.DataFrame.from_dict(comm_between_centrality3, orient='index',
                                                      columns=['aa_male_chronic'])
    comm_between_centrality4 = nx.communicability_betweenness_centrality(
        generate_distance_network(corr_gg_male_chronic, roi_list))
    comm_between_centrality4 = pd.DataFrame.from_dict(comm_between_centrality4, orient='index',
                                                      columns=['gg_male_chronic'])
        #for female
    comm_between_centrality5 = nx.communicability_betweenness_centrality(
        generate_distance_network(corr_aa_female_acute, roi_list))
    comm_between_centrality5 = pd.DataFrame.from_dict(comm_between_centrality5, orient='index',
                                                      columns=['aa_female_acute'])
    comm_between_centrality6 = nx.communicability_betweenness_centrality(
        generate_distance_network(corr_gg_female_acute, roi_list))
    comm_between_centrality6 = pd.DataFrame.from_dict(comm_between_centrality6, orient='index',
                                                      columns=['gg_female_acute'])
    comm_between_centrality7 = nx.communicability_betweenness_centrality(
        generate_distance_network(corr_aa_female_chronic, roi_list))
    comm_between_centrality7 = pd.DataFrame.from_dict(comm_between_centrality7, orient='index',
                                                      columns=['aa_female_chronic'])
    comm_between_centrality8 = nx.communicability_betweenness_centrality(
        generate_distance_network(corr_gg_female_chronic, roi_list))
    comm_between_centrality8 = pd.DataFrame.from_dict(comm_between_centrality8, orient='index',
                                                      columns=['gg_female_chronic'])

    comm_between_centrality = pd.concat([comm_between_centrality1, comm_between_centrality2, comm_between_centrality3, comm_between_centrality4,
                                         comm_between_centrality5, comm_between_centrality6, comm_between_centrality7, comm_between_centrality8], axis = 1)
    #print(comm_between_centrality)
    centrality_save = os.path.join(path_analysis, 'Communicability Betweenness Centrality Pos.csv')
    comm_between_centrality.to_csv(centrality_save)

    comm_save = os.path.join(path_analysis,'Communicability Betweenness Centrality Pos.tif')
    fig = plt.figure()
    plt.title('Communicability Betweenness Centrality by genotype, treatment, sex', fontsize = 20)
    sn.heatmap(comm_between_centrality, cmap='coolwarm', xticklabels=True, yticklabels=True)
    plt.show()
    fig.savefig(comm_save, transparent= True, dpi = 500)
    
    #get between regional Similarity Rank
    simrank_aa_male_acute = get_sim_rank_matrix(corr_aa_male_acute, roi_list)
    simrank_gg_male_acute = get_sim_rank_matrix(corr_gg_male_acute, roi_list)
    simrank_aa_male_chronic = get_sim_rank_matrix(corr_aa_male_chronic, roi_list)
    simrank_gg_male_chronic = get_sim_rank_matrix(corr_gg_male_chronic, roi_list)
    simrank_aa_female_acute = get_sim_rank_matrix(corr_aa_female_acute, roi_list)
    simrank_gg_female_acute = get_sim_rank_matrix(corr_gg_female_acute, roi_list)
    simrank_aa_female_chronic = get_sim_rank_matrix(corr_aa_female_chronic, roi_list)
    simrank_gg_female_chronic = get_sim_rank_matrix(corr_gg_female_chronic, roi_list)
    #compare AA vs. GG
    Compare_Matrix_Perm_Spearmanr(simrank_aa_male_acute,simrank_gg_male_acute, 'Pos Similarity Rank Male Acute AA vs. GG', path_analysis)
    Compare_Matrix_Perm_Spearmanr(simrank_aa_male_chronic, simrank_gg_male_chronic, 'Pos Similarity Rank Male Chronic AA vs. GG',
                                  path_analysis)
    Compare_Matrix_Perm_Spearmanr(simrank_aa_female_acute, simrank_gg_female_acute, 'Pos Similarity Rank Female Acute AA vs. GG',
                                  path_analysis)
    Compare_Matrix_Perm_Spearmanr(simrank_aa_female_chronic, simrank_gg_female_chronic,
                                  'Pos Similarity Rank Female Chronic AA vs. GG',
                                  path_analysis)
    #compare Acute vs. Chronic
    Compare_Matrix_Perm_Spearmanr(simrank_aa_male_acute, simrank_aa_male_chronic, 'Pos Similarity Rank Male AA Acute vs. Chronic',
                                  path_analysis)
    Compare_Matrix_Perm_Spearmanr(simrank_gg_male_acute, simrank_gg_male_chronic,
                                  'Pos Similarity Rank Male GG Acute vs. Chronic',
                                  path_analysis)
    Compare_Matrix_Perm_Spearmanr(simrank_aa_female_acute, simrank_aa_female_chronic,
                                  'Pos Similarity Rank Female AA Acute vs. Chronic',
                                  path_analysis)
    Compare_Matrix_Perm_Spearmanr(simrank_gg_female_acute, simrank_gg_female_chronic,
                                  'Pos Similarity Rank Female GG Acute vs. Chronic',
                                  path_analysis)
                                  
                                  
    simrank_path = os.path.join(path_analysis,'Similarity Rank after bootstrap')
    fig, axs = plt.subplots(2,4)
    fig.suptitle('Acute Male Pair-wise Region Similarity Rank based on Pos Corr Network', fontsize = 22)
    colormap = 'gist_stern'
    sn.heatmap(simrank_aa_male_acute, cmap=colormap, xticklabels=True, yticklabels=True, ax=axs[0, 0])
    axs[0, 0].set_title("Acute AA Male", fontsize = 20)
    sn.heatmap(simrank_aa_male_chronic, cmap=colormap, xticklabels=True, yticklabels=True, ax=axs[1, 0])
    axs[1, 0].set_title("Chronic AA Male", fontsize = 20)
    sn.heatmap(simrank_gg_male_acute, cmap=colormap, xticklabels=True, yticklabels=True, ax=axs[0, 1])
    axs[0, 1].set_title("Acute GG Male", fontsize = 20)
    sn.heatmap(simrank_gg_male_chronic, cmap=colormap, xticklabels=True, yticklabels=True, ax=axs[1, 1])
    axs[1, 1].set_title("Chronic GG Male", fontsize = 20)

    sn.heatmap(simrank_aa_female_acute, cmap=colormap, xticklabels=True, yticklabels=True, ax=axs[0, 2])
    axs[0, 2].set_title("Acute AA Female", fontsize = 20)
    sn.heatmap(simrank_aa_female_chronic, cmap=colormap, xticklabels=True, yticklabels=True, ax=axs[1, 2])
    axs[1, 2].set_title("Chronic AA Female", fontsize = 20)
    sn.heatmap(simrank_gg_female_acute, cmap=colormap, xticklabels=True, yticklabels=True, ax=axs[0, 3])
    axs[0, 3].set_title("Acute GG Female", fontsize = 20)
    sn.heatmap(simrank_gg_female_chronic, cmap=colormap, xticklabels=True, yticklabels=True, ax=axs[1, 3])
    axs[1, 3].set_title("Chronic GG Female", fontsize = 20)
    remove_label(axs,(2,4))
    plt.show()
    fig.savefig(simrank_path, transparent=True, dpi=500)

    


    # Network Control Theory
    #with bootstrapmethod and pair-wise calculation + blunt the cortical regions, this will take a day
    blunt = ['ACAd', 'ACAv', 'PL','ILA', 'CLA', 'PVT']

    min_energy_blunt_boot_path = os.path.join(path_analysis, 'Network Control\Minimum Energy blunt boot.csv')
    min_aa_m_blunt = min_control_energy(outpath_aa_acute_m, outpath_aa_chronic_m, roi_list1, blunt_region=cortex_regions,method = 'individual', title='AA Male')

    min_gg_m_blunt = min_control_energy(outpath_gg_acute_m, outpath_gg_chronic_m, roi_list1, blunt_region=cortex_regions,method = 'individual', title='GG Male')

    min_aa_f_blunt = min_control_energy(outpath_aa_acute_f, outpath_aa_chronic_f, roi_list1, blunt_region=cortex_regions,method = 'individual', title='AA Female')

    min_gg_f_blunt = min_control_energy(outpath_gg_acute_f, outpath_gg_chronic_f, roi_list1, blunt_region=cortex_regions,method = 'individual', title='GG Female')

    min_blunt_boot = pd.concat([min_aa_m_blunt,min_gg_m_blunt,min_aa_f_blunt,min_gg_f_blunt],axis = 1)
    print(min_blunt_boot)
    min_blunt_boot.to_csv(min_energy_blunt_boot_path)
    

    min_energy_blunt_boot_permutation_path = os.path.join(path_analysis, 'Network Control\Permuted Minimum Energy blunt boot.csv')
    p_min_blunt_boot_list = pd.DataFrame()
    p_index_list = pd.DataFrame()
    for i in range(50):
        randblunt = random.sample(range(0,len(roi_list1)),7)
        randbluntlist = [roi_list1[i] for i in randblunt]
        print('random picked blunt regions by: ',randbluntlist,' ......')
        p_min_aa_m_blunt = min_control_energy(outpath_aa_acute_m, outpath_aa_chronic_m, roi_list1, i,
                                            blunt_region=randbluntlist, method='individual', title='AA Male')

        p_min_gg_m_blunt = min_control_energy(outpath_gg_acute_m, outpath_gg_chronic_m, roi_list1, i,
                                            blunt_region=randbluntlist, method='individual', title='GG Male')

        p_min_aa_f_blunt = min_control_energy(outpath_aa_acute_f, outpath_aa_chronic_f, roi_list1,i,
                                            blunt_region=randbluntlist, method='individual', title='AA Female')

        p_min_gg_f_blunt = min_control_energy(outpath_gg_acute_f, outpath_gg_chronic_f, roi_list1,i,
                                            blunt_region=randbluntlist, method='individual', title='GG Female')
        p_min_blunt_boot = pd.concat([p_min_aa_m_blunt, p_min_gg_m_blunt, p_min_aa_f_blunt, p_min_gg_f_blunt], axis=1)
        p_min_blunt_boot = p_min_blunt_boot.mean(axis = 0).tolist()
        p_min_blunt_boot = pd.DataFrame(p_min_blunt_boot, index=['AA Male','GG Male','AA Female','GG Female']).transpose()
        p_min_blunt_boot['Blunt Region'] = ''.join(str(e) for e in randbluntlist)
        print('mean of min energy after blunt: ',randbluntlist, '  is:    ',p_min_blunt_boot)
        p_min_blunt_boot_list = pd.concat([p_min_blunt_boot_list,p_min_blunt_boot],axis = 0)
    p_min_blunt_boot_list.to_csv(min_energy_blunt_boot_permutation_path)

    
    #################
    
    
    #pair wise bootstrp conversion
    min_energy_blunt_boot_path = os.path.join(path_analysis, 'Network Control\Minimum Energy blunt boot.csv')
    #with bootstrap method and pair-wise calculation, this will take a day
    min_control_energy_boot_path_aa_m = os.path.join(path_analysis,'Network Control\Minimum Control Energy AA Male.csv')
    min_control_energy_boot_path_gg_m = os.path.join(path_analysis,
                                                     'Network Control\Minimum Control Energy GG Male.csv')
    min_control_energy_boot_path_aa_f = os.path.join(path_analysis,
                                                     'Network Control\Minimum Control Energy AA Female.csv')
    min_control_energy_boot_path_gg_f = os.path.join(path_analysis,
                                                     'Network Control\Minimum Control Energy GG Female.csv')

    min_energy_boot_path = os.path.join(path_analysis,'Network Control\Minimum Energy boot.csv')
    min_energy_stability_boot_path = os.path.join(path_analysis, 'Network Control\Minimum Energy Stability boot.csv')
    
    
    #calculating stability
    min_control_energy_aa_m, min_aa_a_m = min_control_energy(outpath_aa_acute_m, outpath_aa_acute_m, roi_list1, tracker =1,
                                                           receptor_density=receptor_density, method = 'individual', title='AA Acute Male')
    min_control_energy_gg_m, min_gg_a_m = min_control_energy(outpath_gg_acute_m, outpath_gg_acute_m, roi_list1,tracker =1,
                                                           receptor_density=receptor_density, method='individual', title='GG Acute Male')
    min_control_energy_aa_m, min_aa_c_m = min_control_energy(outpath_aa_chronic_m, outpath_aa_chronic_m, roi_list1, tracker=1,
                                                           receptor_density=receptor_density, method='individual',
                                                           title='AA Chronic Male')
    min_control_energy_gg_m, min_gg_c_m = min_control_energy(outpath_gg_chronic_m, outpath_gg_chronic_m, roi_list1, tracker=1,
                                                           receptor_density=receptor_density, method='individual',
                                                           title='GG Chronic Male')

    min_control_energy_aa_f, min_aa_a_f = min_control_energy(outpath_aa_acute_f, outpath_aa_acute_f, roi_list1,tracker =1,
                                                           receptor_density=receptor_density, method='individual', title='AA Acute Female')
    min_control_energy_gg_f, min_gg_a_f = min_control_energy(outpath_gg_acute_f, outpath_gg_acute_f, roi_list1,tracker =1,
                                                           receptor_density=receptor_density, method='individual', title='GG Acute Female')
    min_control_energy_aa_f, min_aa_c_f = min_control_energy(outpath_aa_chronic_f, outpath_aa_chronic_f, roi_list1,
                                                           tracker=1,
                                                           receptor_density=receptor_density, method='individual',
                                                           title='AA Chronic Female')
    min_control_energy_gg_f, min_gg_c_f = min_control_energy(outpath_gg_chronic_f, outpath_gg_chronic_f, roi_list1,
                                                           tracker=1,
                                                           receptor_density=receptor_density, method='individual',
                                                           title='GG Chronic Female')


    min_boot = pd.concat([min_aa_a_m, min_gg_a_m,min_aa_c_m, min_gg_c_m, min_aa_a_f, min_gg_a_f, min_aa_c_f, min_gg_c_f], axis=1)
    min_boot.to_csv(min_energy_stability_boot_path)

    
    #calculating weighted by receptor density, mu delta kappa data concatenate by rows
    min_control_energy_aa_m, min_aa_m = min_control_energy(outpath_aa_acute_m, outpath_aa_chronic_m, roi_list1, tracker ='receptor',
                                                           receptor_density=receptor_density, method = 'individual', title='AA Male')
    min_control_energy_gg_m, min_gg_m = min_control_energy(outpath_gg_acute_m, outpath_gg_chronic_m, roi_list1,tracker ='receptor',
                                                           receptor_density=receptor_density, method='individual', title='GG Male')
    min_control_energy_aa_f, min_aa_f = min_control_energy(outpath_aa_acute_f, outpath_aa_chronic_f, roi_list1,tracker ='receptor',
                                                           receptor_density=receptor_density, method='individual', title='AA Female')
    min_control_energy_gg_f, min_gg_f = min_control_energy(outpath_gg_acute_f, outpath_gg_chronic_f, roi_list1,tracker ='receptor',
                                                           receptor_density=receptor_density, method='individual', title='GG Female')
    
    #min_control_energy_aa_m = pd.DataFrame(min_control_energy_aa_m, index=roi_list1, columns=['AA Male'])
    min_control_energy_aa_m.to_csv(min_control_energy_boot_path_aa_m)
    min_control_energy_gg_m.to_csv(min_control_energy_boot_path_gg_m)
    min_control_energy_aa_f.to_csv(min_control_energy_boot_path_aa_f)
    min_control_energy_gg_f.to_csv(min_control_energy_boot_path_gg_f)
    min_boot = pd.concat([min_aa_m,min_gg_m,min_aa_f,min_gg_f],axis = 1)
    min_boot.to_csv(min_energy_boot_path)
    
    
    group2show = os.path.join(path_analysis,'Network Control\group2show.csv')
    min_control_energy_boot = os.path.join(path_analysis,'Network Control\Minimum Energy boot.tif')
    min_control_energy_aa_m_boot_path = os.path.join(path_analysis,'Network Control\Minimum Control Energy AA Male.csv')
    min_control_energy_gg_m_boot_path = os.path.join(path_analysis,
                                                     'Network Control\Minimum Control Energy GG Male.csv')
    min_control_energy_aa_f_boot_path = os.path.join(path_analysis,
                                                     'Network Control\Minimum Control Energy AA Female.csv')
    min_control_energy_gg_f_boot_path = os.path.join(path_analysis,
                                                     'Network Control\Minimum Control Energy GG Female.csv')
    
    # please check if the path to read and analyze the bootstrapped control energy is correct

    min_control_energy_aa_m = reform_min_energy_boot(min_control_energy_aa_m_boot_path,group2show,'AA','Male')

    min_control_energy_gg_m = reform_min_energy_boot(min_control_energy_gg_m_boot_path,group2show,'GG', 'Male')

    min_control_energy_aa_f = reform_min_energy_boot(min_control_energy_aa_f_boot_path, group2show, 'AA', 'Female')
    min_control_energy_gg_f = reform_min_energy_boot(min_control_energy_gg_f_boot_path, group2show, 'GG', 'Female')


    deno_min_energy_list = pd.DataFrame()
    deno_min_energy = pd.read_csv(min_energy_boot_path)
    for i in range(23):
        deno_min_energy_list = pd.concat([deno_min_energy_list, deno_min_energy], axis=0)
    print(deno_min_energy_list)

    min_control_energy_aa_m['energy'] = min_control_energy_aa_m['energy'].values / deno_min_energy_list['AA Male'].values
    min_control_energy_gg_m['energy'] = min_control_energy_gg_m['energy'].values / deno_min_energy_list['GG Male'].values
    min_control_energy_aa_f['energy'] = min_control_energy_aa_f['energy'].values / deno_min_energy_list['AA Female'].values
    min_control_energy_gg_f['energy'] = min_control_energy_gg_f['energy'].values / deno_min_energy_list['GG Female'].values
    min_control_energy = pd.concat([min_control_energy_aa_m, min_control_energy_gg_m, min_control_energy_aa_f, min_control_energy_gg_f], axis=0)
    min_control_energy.to_csv(r'D:\Count Summary\Analysis\\Network Control\min control energy boot.csv')
    print(min_control_energy)

    fig = plt.figure()
    fig.suptitle("Minimum Control Energy convert from Acute to Chronic by region after BootStrap", fontsize = 20)
    sn.set_theme(style="darkgrid")
    sn.lineplot(x='roi', y='energy', hue='genotype', style='sex', data=min_control_energy, ci='sd')
    plt.show()
    fig.savefig(min_control_energy_boot, transparent= True, dpi = 500)
    maov = MANOVA.from_formula('energy + roi ~ genotype + sex', data=min_control_energy)
    print(maov.mv_test())
    '''


    '''

    #reshape to input to prism for plotting
    min_control_energy_aa_m = pd.read_csv(min_control_energy_aa_m_boot_path)
    min_control_energy_gg_m = pd.read_csv(min_control_energy_gg_m_boot_path)
    min_control_energy_aa_f = pd.read_csv(min_control_energy_aa_f_boot_path)
    min_control_energy_gg_f = pd.read_csv(min_control_energy_gg_f_boot_path)
    min_control_energy_aa_m = min_control_energy_aa_m.drop(columns =['group'])
    min_control_energy_gg_m = min_control_energy_gg_m.drop(columns=['group'])
    min_control_energy_aa_f = min_control_energy_aa_f.drop(columns=['group'])
    min_control_energy_gg_f = min_control_energy_gg_f.drop(columns=['group'])
    min_control_energy_aa_m.rename(columns={min_control_energy_aa_m.columns[0]: "roi"}, inplace=True)
    min_control_energy_gg_m.rename(columns={min_control_energy_gg_m.columns[0]: "roi"}, inplace=True)
    min_control_energy_aa_f.rename(columns={min_control_energy_aa_f.columns[0]: "roi"}, inplace=True)
    min_control_energy_gg_f.rename(columns={min_control_energy_gg_f.columns[0]: "roi"}, inplace=True)
    min_control_energy_aa_m.set_index(['roi'], inplace=True)
    min_control_energy_gg_m.set_index(['roi'], inplace=True)
    min_control_energy_aa_f.set_index(['roi'], inplace=True)
    min_control_energy_gg_f.set_index(['roi'], inplace=True)
    min_control_energy_aa_m = min_control_energy_aa_m.transpose()
    min_control_energy_gg_m = min_control_energy_gg_m.transpose()
    min_control_energy_aa_f = min_control_energy_aa_f.transpose()
    min_control_energy_gg_f = min_control_energy_gg_f.transpose()
    
    path_min_control_energy = os.path.join(path_analysis,"Network Control\Minimum Control Energy.csv")
    path_min_control_energy_plot = os.path.join(path_analysis, "Network Control\Minimum Control Energy.tif")
    min_control_energy_aa_m, min_aa_m = min_control_energy(outpath_aa_acute_m, outpath_aa_chronic_m, roi_list1)
    min_control_energy_aa_m = pd.DataFrame(min_control_energy_aa_m, index=roi_list1, columns=['AA Male'])

    min_control_energy_gg_m, min_gg_m= min_control_energy(outpath_gg_acute_m, outpath_gg_chronic_m, roi_list1)
    min_control_energy_gg_m = pd.DataFrame(min_control_energy_gg_m,index=roi_list1, columns=['GG Male'])

    min_control_energy_aa_f, min_aa_f = min_control_energy(outpath_aa_acute_f, outpath_aa_chronic_f, roi_list1)
    min_control_energy_aa_f = pd.DataFrame(min_control_energy_aa_f,index=roi_list1, columns=['AA Female'])

    min_control_energy_gg_f, min_gg_f = min_control_energy(outpath_gg_acute_f, outpath_gg_chronic_f, roi_list1)
    min_control_energy_gg_f = pd.DataFrame(min_control_energy_gg_f,index=roi_list1, columns=['GG Female'])

    min_energy_list = [min_aa_m, min_gg_m, min_aa_f, min_gg_f]
    min_energy_list =  [min_energy_list,['AA Male', 'GG Male', 'AA Female', 'GG Female']]
    print('min energy put from acute to chronic: ', min_energy_list)
    min_control_energy_list = pd.concat([min_control_energy_aa_m,min_control_energy_gg_m,min_control_energy_aa_f,min_control_energy_gg_f], axis = 1)

    min_control_energy_list.to_csv(path_min_control_energy)


    print('min control energy: ', min_control_energy_list)

    fig, axs = plt.subplots(2,1)
    fig.suptitle('A112G Minimum Control Energy by Genotype, Sex')
    X_axis = np.arange(len(roi_list1))
    axs[0].bar(X_axis - 0.2,min_control_energy_aa_m['AA Male'].values, 0.4, label = 'AA Male')
    axs[0].bar(X_axis + 0.2,min_control_energy_gg_m['GG Male'].values, 0.4, label = 'GG Male')
    plt.xticks([],[])
    axs[1].bar(X_axis - 0.2, min_control_energy_aa_f['AA Female'].values, 0.4, label='AA Female')
    axs[1].bar(X_axis + 0.2, min_control_energy_gg_f['GG Female'].values, 0.4, label='GG Female')

    plt.xticks(X_axis, roi_list1)
    plt.xlabel("ROI controled")
    plt.ylabel("Minimum Control Energy")
    plt.legend()
    plt.show()
    
    # remove those regions and see if minimum energy reduce to similar level
    roi_list2 = roi_list
    # remove SNc because no connectivity data
    roi_list2.remove('ACAd')
    roi_list2.remove('ACAv')
    roi_list2.remove('AId')
    roi_list2.remove('AIv')
    roi_list2.remove('PL')
    roi_list2.remove('ILA')
    roi_list2.remove('CLA')
    roi_list2.remove('PVT')
    min_aa_m, c1 = min_control_energy(outpath_aa_acute_m, outpath_aa_chronic_m, roi_list2)

    min_gg_m, c2 = min_control_energy(outpath_gg_acute_m, outpath_gg_chronic_m, roi_list2)

    min_aa_f, c3 = min_control_energy(outpath_aa_acute_f, outpath_aa_chronic_f, roi_list2)

    min_gg_f, c4 = min_control_energy(outpath_gg_acute_f, outpath_gg_chronic_f, roi_list2)

    min_energy_list = [c1, c2, c3, c4]
    min_energy_list = [min_energy_list, ['AA Male', 'GG Male', 'AA Female', 'GG Female']]
    print('min energy put from acute to chronic: ', min_energy_list)
    
    
    
    #blunt all those cortical regions and see increase in minimum energy
    blunt_region = ['ACAd', 'ACAv', 'AId', 'AIv', 'PL','ILA', 'CLA']
    kmin_aa_m, c1 = min_control_energy(outpath_aa_acute_m, outpath_aa_chronic_m, roi_list1, blunt_region = cortex_regions)

    kmin_gg_m, c2 = min_control_energy(outpath_gg_acute_m, outpath_gg_chronic_m, roi_list1, blunt_region = cortex_regions)

    kmin_aa_f, c3 = min_control_energy(outpath_aa_acute_f, outpath_aa_chronic_f, roi_list1, blunt_region = cortex_regions)

    kmin_gg_f, c4 = min_control_energy(outpath_gg_acute_f, outpath_gg_chronic_f, roi_list1, blunt_region = cortex_regions)

    min_energy_list = [kmin_aa_m, kmin_gg_m, kmin_aa_f, kmin_gg_f]
    min_energy_list = [min_energy_list, ['AA Male', 'GG Male', 'AA Female', 'GG Female']]
    print('min energy put from acute to chronic: ', min_energy_list)
    
    
    #run  Kolmogorov-Smirnov test for each pair of condition
    # this is not proper as it only test distribution of frequency data, we have each column representing a category, 
    #it's also not frequency data of category no chi-square test, so permutation test needed
    Corr_Matrix_Two_Sample_Kol_Test(corr_aa_female_acute,corr_gg_female_acute, "AA Female Acute vs. GG Female Acute")
    Corr_Matrix_Two_Sample_Kol_Test(corr_aa_female_chronic, corr_gg_female_chronic, "AA Female chronic vs. GG Female chronic")
    Corr_Matrix_Two_Sample_Kol_Test(corr_aa_male_acute, corr_gg_male_acute, "AA male Acute vs. GG male Acute")
    Corr_Matrix_Two_Sample_Kol_Test(corr_aa_male_chronic, corr_gg_male_chronic, "AA male chronic vs. GG male chronic")
    
    #compare corr matrix by permutation spearman r test
    #AA vs. GG
    Compare_Matrix_Perm_Spearmanr(corr_aa_female_acute,corr_gg_female_acute, "AA Female Acute vs. GG Female Acute Correlation", path_analysis)
    Compare_Matrix_Perm_Spearmanr(corr_aa_female_chronic, corr_gg_female_chronic,
                                    "AA Female chronic vs. GG Female chronic Correlation", path_analysis)
    Compare_Matrix_Perm_Spearmanr(corr_aa_male_acute, corr_gg_male_acute, "AA male Acute vs. GG male Acute Correlation", path_analysis)
    Compare_Matrix_Perm_Spearmanr(corr_aa_male_chronic, corr_gg_male_chronic, "AA male chronic vs. GG male chronic Correlation", path_analysis)

    #Acute vs. Chronic
    Compare_Matrix_Perm_Spearmanr(corr_aa_female_acute, corr_aa_female_chronic,
                                  "AA Female Acute vs. Chronic", path_analysis)
    Compare_Matrix_Perm_Spearmanr(corr_gg_female_acute, corr_gg_female_chronic,
                                  "GG Female Acute vs. Chronic", path_analysis)
    Compare_Matrix_Perm_Spearmanr(corr_aa_male_acute, corr_aa_male_chronic,
                                  "AA Male Acute vs. Chronic", path_analysis)
    Compare_Matrix_Perm_Spearmanr(corr_gg_male_acute, corr_gg_male_chronic,
                                  "GG Male Acute vs. Chronic", path_analysis)
 
    #male vs. female
    Compare_Matrix_Perm_Spearmanr(corr_aa_male_acute, corr_aa_female_acute,"AA Acute male vs. female", path_analysis)
    Compare_Matrix_Perm_Spearmanr(corr_gg_male_acute, corr_gg_female_acute, "GG Acute male vs. female", path_analysis)
    Compare_Matrix_Perm_Spearmanr(corr_aa_male_chronic, corr_aa_female_chronic, "AA Chronic male vs. female", path_analysis)
    Compare_Matrix_Perm_Spearmanr(corr_gg_male_chronic, corr_gg_female_chronic, "GG Chronic male vs. female", path_analysis)
    
    ##allensdk plot

    
    experiments = mcc.get_experiments(cre=False,injection_structure_ids=struclist)
    print(experiments)
    experiment_ids = [e['id'] for e in experiments]
    print(experiments)
    #print("%d experiments with interested rois" % len(experiments))

    #structure_unionizes = mcc.get_structure_unionizes([e['id'] for e in experiments],is_injection=False,structure_ids=struclist, include_descendants=True)
    
    pm = mcc.get_projection_matrix(experiment_ids=experiment_ids,
                                   projection_structure_ids=struclist,
                                   hemisphere_ids=[3],  # both hemisphere
                                   parameter='projection_density')
    print(pm)
    
    output_dir = '.'
    reference_space_key = 'annotation/ccf_2017'
    resolution = 25
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=Path(output_dir) / 'manifest.json')
    tree = rspc.get_structure_tree(structure_graph_id=1)
    annotation, meta = rspc.get_annotation_volume()
    rsp = rspc.get_reference_space()
    # The file should appear in the reference space key directory
    os.listdir(Path(output_dir) / reference_space_key)
    #tree.get_structures_by_name(['Dorsal auditory area'])
    #brain_observatory_structures = rsp.structure_tree.get_structures_by_set_id([514166994])
    #brain_observatory_ids = [st['id'] for st in brain_observatory_structures]
    #brain_observatory_mask = rsp.make_structure_mask(brain_observatory_ids)
    #whole_cortex_mask = rsp.make_structure_mask([315])
    
    
    #get the image from allen mouse atlas
    reference_key = 4000
    imgid = "coronalbg"+str(reference_key)+".png"
    imgoutpath = os.path.join(path3, imgid)
    bgimg = rsp.get_slice_image(0, reference_key)

    #fig, ax = plt.subplots(figsize=(10, 10))

    Image.fromarray(bgimg).save(imgoutpath)


    newim = Image.open(imgoutpath)
    newim = newim.convert('RGBA')
    bgimg = newim.getdata()
    newbgimg = []
    for item in bgimg:
        #print(item)
        if item[0] < 1 and item[0] < 1 and item[0] < 1:
            newbgimg.append((255, 255, 255, 0))
        else:
            newbgimg.append(item)
    newim.putdata(newbgimg)
    plt.imshow(newim, interpolation='none')
    plt.show()
    newim.save(imgoutpath)
    '''
    '''
    #following function modified based on -https://stackoverflow.com/questions/72912011/plotting-2d-picture-on-3d-plot-in-python
    def plot_image(ax, image, axis, xlim, ylim, val, rstride=1, cstride=1, alf = 0.05):
        array = plt.imread(image)
        array = np.swapaxes(array, 0, 1)
        array = np.flip(array, 1)
        step_x, step_y = np.diff(xlim) / array.shape[0], np.diff(ylim) / array.shape[1]
        x_1 = np.arange(xlim[0], xlim[1], step_x)
        y_1 = np.arange(ylim[0], ylim[1], step_y)
        y_1, x_1 = np.meshgrid(y_1, x_1)
        vals = np.ones((array.shape[0], array.shape[1])) * val
        if axis == "x":
            ax.plot_surface(vals, x_1, y_1, rstride=rstride, cstride=cstride, facecolors=array, zorder=-1000, alpha = alf)
        elif axis == "y":
            ax.plot_surface(x_1, vals, y_1, rstride=rstride, cstride=cstride, facecolors=array, zorder=-1000, alpha = alf)
        elif axis == "z":
            ax.plot_surface(x_1, y_1, vals, rstride=rstride, cstride=cstride, facecolors=array, zorder=-1000, alpha = alf)

    fig = plt.figure(figsize=(10, 15))
    fig.add_subplot(111, projection="3d")
    ax = fig.gca(projection='3d')
    zval = [0,100,180,245,332,393]
    # Plot geometry in background
    #plot_image(plt.gca(), 'D:\Count Summary\ptimage\coronalbg3500.png','x', xlim=[-144, 144], ylim=[0, 487.528], val=-144)
    plot_image(plt.gca(), 'D:\Count Summary\ptimage\coronalbg3500.png','y', xlim=[0, 456], ylim=[0, 320],
               val=zval[0])
    plot_image(plt.gca(), 'D:\Count Summary\ptimage\coronalbg4000.png', 'y', xlim=[0, 456], ylim=[0, 320],
               val=zval[1])
    plot_image(plt.gca(), 'D:\Count Summary\ptimage\coronalbg4500.png', 'y', xlim=[0, 456], ylim=[0, 320],
               val=zval[2])
    plot_image(plt.gca(), 'D:\Count Summary\ptimage\coronalbg6130.png', 'y', xlim=[0, 456], ylim=[0, 320],
               val=zval[3])
    plot_image(plt.gca(), 'D:\Count Summary\ptimage\coronalbg8310.png', 'y', xlim=[0, 456], ylim=[0, 320],
               val=zval[4])
    plot_image(plt.gca(), 'D:\Count Summary\ptimage\coronalbg9830.png', 'y', xlim=[0, 456], ylim=[0, 320],
               val=zval[5])
    #plot_image(plt.gca(), 'D:\Count Summary\ptimage\coronalbg3500.png','z', xlim=[-144, 144], ylim=[-144, 144], val=0)

    #plt.show()
    

    center_parent_path = "E:\A112G Final Analysis\TransferData"
    center_sub_path = "annotation\ccf_2017\structure_centers.csv"
    center_coordinate_path = os.path.join(center_parent_path,center_sub_path)
    structre_id_path = "D:\Count Summary\structures from allen atlas.csv"

    def get_roi_coordinate(roi, structure_id_path, center_coordinate_path):
        sip = pd.read_csv(structure_id_path)
        sip = pd.DataFrame(sip)
        sip = sip.loc[sip['acronym'] == roi, 'id'].values
        #print('structure id ', sip)
        ccp = pd.read_csv(center_coordinate_path)
        ccp= pd.DataFrame(ccp)
        ccp = ccp.loc[ccp['structure_id'] == int(sip) ,['x','y','z']]
        return ccp.values

    # connectivity map

    node_position1 = {'VTA' , 'ACB', 'ACAd', 'ACAv', 'AId', 'AIv', 'PL',
                    'ILA', 'BLA', 'BST', 'CEA', 'CLA', 'CP', 'DG', 'GPe', 'LH', 'LHA', 'MH', 'PAG', 'PALv', 'PVT', 'SNc',
                    'SNr'}
    node_position = {}
    for roi in node_position1:
        node_position[roi] = np.divide(get_roi_coordinate(roi,structre_id_path, center_coordinate_path)[0], 25)

    print('node position in space: ', node_position)


    links = corr_gg_female_chronic.stack()
    links.to_csv(outpath3)
    links = pd.read_csv(outpath3)
    links.rename(columns={links.columns[0]: "roi 1"}, inplace=True)
    links.rename(columns={links.columns[1]: "roi 2"}, inplace=True)
    links.rename(columns={links.columns[2]: "corr"}, inplace=True)
    links = pd.DataFrame(links)
    for roi in roi_list:
        links = links.loc[(links["roi 1"] != roi) | (links["roi 2"] != roi)]
    links = links.reset_index()

    g = nx.Graph(dim = 3)
    g.add_nodes_from(roi_list)

    for idx in links.index:
        roi1 = links['roi 1'].iloc[idx]
        roi2 = links['roi 2'].iloc[idx]
        corr = links['corr'].iloc[idx]
        print('adding edge: ',roi1,roi2,corr)
        g.add_edge(roi1,roi2, weight = corr*2)

    edge_labels = dict([(roi1,roi2),d['weight']] for roi1, roi2, d in g.edges(data=True))
    #print(edge_labels)


    for key,value in node_position.items():
        x = value[0]
        y = value[1]
        z = value[2]
        ax.scatter(x,y,z,edgecolors = 'red',alpha = 0.9)
        ax.text(x,y,z, s = str(key), c = 'white')
    for i,j in enumerate(g.edges()):
        #print("edge: ",' ',j)
        x = np.array((node_position[j[0]][0], node_position[j[1]][0]))
        #print(node_position[j[0]][0], node_position[j[1]][0])
        y = np.array((node_position[j[0]][1],node_position[j[1]][1]))
        z = np.array((node_position[j[0]][2],node_position[j[1]][2]))
        weight = edge_labels[(j[0],j[1])]
        transparency = abs(weight)
        line_width = transparency * 6
        #print('x: ', x[0],' ', x[1],'y: ',y, 'z: ',z)
        if transparency <=1:
            if weight >= 0:
                print('adding to plot edge: ', j[0], ' to ', j[1], 'at position pair ', x, y, z)
                print('with positive line width = ', line_width)
                ax.plot(x, y, z, c='orange', linewidth=line_width, alpha=0.9)
            else:
                print('adding to plot edge: ', j[0], ' to ', j[1], 'at position pair ', x, y, z)
                print('with negative line width = ', line_width)
                ax.plot(x, y, z, c='blue', linewidth=line_width, alpha=0.9)

    print("started ploting, this may takes 20-30 minutes ...")
    plt.show()
    print("ploting finished.")
    '''



    '''
    
    #connectivity map
    group2v =  corr_aa_female_acute

    title_a_a_f = "AA Female Acute morphine connectivity graph"
    title_g_a_f = "GG Female Acute morphine connectivity graph"
    title_a_c_f = "AA Female Chronic morphine connectivity graph"
    title_g_c_f = "GG Female Chronic morphine connectivity graph"
    title_a_a_m = "AA Male Acute morphine connectivity graph"
    title_g_a_m = "GG Male Acute morphine connectivity graph"
    title_a_c_m = "AA Male Chronic morphine connectivity graph"
    title_g_c_m = "GG Male Chronic morphine connectivity graph"



    imagetitle = title_a_a_f + ".tif"
    network_outpath = "D:\Count Summary\Data Show"
    degree_centality_list_outpath = "D:\Count Summary\Analysis\degree centrality"

    rescale = 1.5
    #fig = plt.gcf()
    #plt.title(title_a_a_f)
    #image_save = os.path.join(network_outpath,imagetitle)
    #Degree Centrality and Betweenness Centrality Analysis Graph
    
    degree_centrality_list_path = os.path.join(degree_centality_list_outpath, "degree centrality.csv")
    between_degree_centrality_list_path = os.path.join(degree_centality_list_outpath, "between degree centrality.csv")
    between_edge_centrality_list_path = os.path.join(degree_centality_list_outpath, "between edge centrality.csv")


    degree_a_a_m = pd.DataFrame(nx.degree_centrality(generate_distance_network(corr_aa_male_acute, roi_list)), index=[0])
    degree_g_a_m = pd.DataFrame(nx.degree_centrality(generate_distance_network(corr_gg_male_acute, roi_list)), index=[1])
    degree_a_c_m = pd.DataFrame(nx.degree_centrality(generate_distance_network(corr_aa_male_chronic, roi_list)), index=[2])
    degree_g_c_m = pd.DataFrame(nx.degree_centrality(generate_distance_network(corr_gg_male_chronic, roi_list)), index=[3])

    #print(nx.degree_centrality(generate_network(corr_aa_female_acute, roi_list, roi_color)))
    degree_a_a_f = pd.DataFrame(nx.degree_centrality(generate_distance_network(corr_aa_female_acute, roi_list)), index = [4])
    degree_g_a_f = pd.DataFrame(nx.degree_centrality(generate_distance_network(corr_gg_female_acute, roi_list)),index = [5])
    degree_a_c_f = pd.DataFrame(nx.degree_centrality(generate_distance_network(corr_aa_female_chronic, roi_list)),index = [6])
    degree_g_c_f = pd.DataFrame(nx.degree_centrality(generate_distance_network(corr_gg_female_chronic, roi_list)),index = [7])

    degree_centrality_list = pd.concat([degree_a_a_m,degree_g_a_m,degree_a_c_m,degree_g_c_m,degree_a_a_f,degree_g_a_f,degree_a_c_f,degree_g_c_f], axis = 0)

    degree_centrality_list.to_csv(degree_centrality_list_path)

    between_degree_a_a_m = pd.DataFrame(nx.betweenness_centrality(generate_distance_network(corr_aa_male_acute, roi_list), weight = 'corr'), index=[0])
    between_degree_g_a_m = pd.DataFrame(nx.betweenness_centrality(generate_distance_network(corr_gg_male_acute, roi_list), weight = 'corr'), index=[1])
    between_degree_a_c_m = pd.DataFrame(nx.betweenness_centrality(generate_distance_network(corr_aa_male_chronic, roi_list), weight = 'corr'), index=[2])
    between_degree_g_c_m = pd.DataFrame(nx.betweenness_centrality(generate_distance_network(corr_gg_male_chronic, roi_list), weight = 'corr'), index=[3])

    #print(nx.degree_centrality(generate_network(corr_aa_female_acute, roi_list, roi_color)))
    between_degree_a_a_f = pd.DataFrame(nx.betweenness_centrality(generate_distance_network(corr_aa_female_acute, roi_list), weight = 'corr'), index = [4])
    between_degree_g_a_f = pd.DataFrame(nx.betweenness_centrality(generate_distance_network(corr_gg_female_acute, roi_list), weight = 'corr'),index = [5])
    between_degree_a_c_f = pd.DataFrame(nx.betweenness_centrality(generate_distance_network(corr_aa_female_chronic, roi_list), weight = 'corr'),index = [6])
    between_degree_g_c_f = pd.DataFrame(nx.betweenness_centrality(generate_distance_network(corr_gg_female_chronic, roi_list), weight = 'corr'),index = [7])

    between_degree_centrality_list = pd.concat([between_degree_a_a_m,between_degree_g_a_m,between_degree_a_c_m,between_degree_g_c_m,between_degree_a_a_f,between_degree_g_a_f,between_degree_a_c_f,between_degree_g_c_f], axis = 0)
    #print("between degree centrality: ", between_degree_centrality_list)
    between_degree_centrality_list.to_csv(between_degree_centrality_list_path)




    #edge betweenness centrality
    between_edge_degree_a_a_m = pd.DataFrame(
        nx.edge_betweenness_centrality(generate_distance_network(corr_aa_male_acute, roi_list), weight='corr'), index=[0])
    between_edge_degree_g_a_m = pd.DataFrame(
        nx.edge_betweenness_centrality(generate_distance_network(corr_gg_male_acute, roi_list), weight='corr'), index=[0])
    between_edge_degree_a_c_m = pd.DataFrame(
        nx.edge_betweenness_centrality(generate_distance_network(corr_aa_male_chronic, roi_list), weight='corr'), index=[0])
    between_edge_degree_g_c_m = pd.DataFrame(
        nx.edge_betweenness_centrality(generate_distance_network(corr_gg_male_chronic, roi_list), weight='corr'), index=[0])

    
    #all shortest path
    closet_path_save = os.path.join(path_analysis,"closet path.csv")
    tcloset_path_save = os.path.join(path_analysis, "transposed closet path.csv")
    avg_closet_path_save = os.path.join(path_analysis, "avg closet path.csv")
    closet_path_a_m, tcloset_path_a_m, avg_st_a_m = get_closet_path(corr_aa_male_acute, corr_gg_male_acute, roi_list, 'Acute Male')
    closet_path_c_m, tcloset_path_c_m, avg_st_c_m = get_closet_path(corr_aa_male_chronic, corr_gg_male_chronic, roi_list, 'Chronic Male')
    closet_path_a_f, tcloset_path_a_f, avg_st_a_f = get_closet_path(corr_aa_female_acute, corr_gg_female_acute, roi_list, 'Acute Female')
    closet_path_c_f, tcloset_path_c_f, avg_st_c_f = get_closet_path(corr_aa_female_chronic, corr_gg_female_chronic, roi_list, 'Chronic Female')

    closet_path = pd.concat([closet_path_a_m,closet_path_c_m,closet_path_a_f,closet_path_c_f], axis = 1)
    tcloset_path = pd.concat([tcloset_path_a_m, tcloset_path_c_m, tcloset_path_a_f, tcloset_path_c_f], axis=1)
    avg_st = pd.concat([avg_st_a_m, avg_st_c_m, avg_st_a_f,avg_st_c_f], axis = 1)
    avg_st.to_csv(avg_closet_path_save)
    #print(avg_st)
    tcloset_path.to_csv(tcloset_path_save)

    # print(nx.degree_centrality(generate_network(corr_aa_female_acute, roi_list, roi_color)))
    #between_edge_degree_a_a_f = pd.DataFrame(nx.edge_betweenness_centrality(generate_distance_network(corr_aa_female_acute, roi_list), weight='corr'), index=[0])
    #between_edge_degree_g_a_f = pd.DataFrame(nx.edge_betweenness_centrality(generate_distance_network(corr_gg_female_acute, roi_list), weight='corr'), index=[0])
    #between_edge_degree_a_c_f = pd.DataFrame(nx.edge_betweenness_centrality(generate_distance_network(corr_aa_female_chronic, roi_list), weight='corr'),index=[0])
    #between_edge_degree_g_c_f = pd.DataFrame(nx.edge_betweenness_centrality(generate_distance_network(corr_gg_female_chronic, roi_list), weight='corr'),index=[0])

    #plt.show()
    #fig.savefig(image_save, transparent= True, dpi = 500)
    #plt.close()
    
    #heatmaps
    correlationmatrix_path = os.path.join(path_analysis,'correlation matrix by genotype treatment sex.tif')
    fig, axs = plt.subplots(2,4)
    fig.suptitle('A112G cFos Expression Correlation Matrix by treatment, genotype, sex', fontsize=20)
    sn.heatmap(corr_aa_male_acute, cmap='coolwarm', xticklabels=True, yticklabels=True, ax= axs[0,0])
    axs[0, 0].set_title("Acute AA Male",fontsize=16)
    sn.heatmap(corr_aa_male_chronic, cmap='coolwarm', xticklabels=True, yticklabels=True, ax= axs[1,0])
    axs[1, 0].set_title("Chronic AA Male",fontsize=16)
    sn.heatmap(corr_gg_male_acute, cmap='coolwarm', xticklabels=True, yticklabels=True, ax=axs[0, 1])
    axs[0,1].set_title("Acute GG Male",fontsize=16)
    sn.heatmap(corr_gg_male_chronic, cmap='coolwarm', xticklabels=True, yticklabels=True, ax=axs[1, 1])
    axs[1, 1].set_title("Chronic GG Male",fontsize=16)

    sn.heatmap(corr_aa_female_acute, cmap='coolwarm', xticklabels=True, yticklabels=True, ax=axs[0,2])
    axs[0,2].set_title("Acute AA Female",fontsize=16)
    sn.heatmap(corr_aa_female_chronic, cmap='coolwarm', xticklabels=True, yticklabels=True, ax=axs[1,2])
    axs[1,2].set_title("Chronic AA Female",fontsize=16)
    sn.heatmap(corr_gg_female_acute, cmap='coolwarm', xticklabels=True, yticklabels=True, ax=axs[0,3])
    axs[0,3].set_title("Acute GG Female",fontsize=16)
    sn.heatmap(corr_gg_female_chronic, cmap='coolwarm', xticklabels=True, yticklabels=True, ax=axs[1,3])
    axs[1,3].set_title("Chronic GG Female",fontsize=16)
    remove_label(axs,(2,4))
    plt.show()
    fig.savefig(correlationmatrix_path, transparent= True, dpi = 500)

    


    #plt.savefig("D:\Count Summary\Data Show\Correlation matrix by genotype, sex.tif", transparent=True, dpi=500)
    sn.heatmap(corr_aa_male_acute, cmap='bwr', xticklabels=True, yticklabels=True, ax= axs[0,0])
    axs[0, 0].set_title("Acute Male")
    sn.heatmap(corr_aa_male_chronic, cmap='bwr', xticklabels=True, yticklabels=True, ax= axs[1,0])
    axs[1, 0].set_title("Chronic Male")
    sn.heatmap(corr_aa_female_acute, cmap='bwr', xticklabels=True, yticklabels=True, ax=axs[0,1])
    axs[0,1].set_title("Acute Female")
    sn.heatmap(corr_aa_female_chronic, cmap='bwr', xticklabels=True, yticklabels=True, ax=axs[1,1])
    axs[1,1].set_title("Chronic Female")
    '''