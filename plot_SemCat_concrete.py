#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:01:52 2022

@author: fm02
"""

# plot results of concat_indivisual_SemCat_confusion-matrix_slurm

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

sns.set_theme(context="notebook",
              style="white",
              font="sans-serif")

sns.set_style("ticks")

colors = sns.color_palette(["#FFBE0B",
                            "#FB5607",
                            "#FF006E",
                            "#8338EC",
                            "#3A86FF",
                            "#1D437F"
                            ])

kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']
kk2 = ['visual', 'hand', 'hear']

scores = []

for i in range(0, 18):
    with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/concabs_balanced_scores_{i}.P" , "rb") as f:
        scores.append(pickle.load(f))

# # create times array
times = np.arange(-300,900,4)

reorg = dict.fromkeys(["LD", "sd"])

reorg["LD"] = []
reorg["sd"] = []

for sub_score in scores:
    reorg["LD"].append(sub_score["LD"])
    reorg["sd"].append(sub_score["sd"])
    
del(scores)
scores = dict.fromkeys(reorg.keys())
for task in reorg:
    scores[task] = pd.DataFrame(reorg[task], index=range(0,18))
    
del(reorg)

p_clust = {}
t_clust = {}
clusters = {}
p_values = {}
H0 = {} 

for task in scores.keys():
    p_clust[task] = pd.DataFrame(index=range(300), columns=kkROI)
    for roi in scores[task].keys():
        # Reshape data to what is equivalent to (n_samples, n_space, n_time)
        data = np.vstack(scores[task][roi]).reshape(18, 1, 300)
        # Compute threshold from t distribution (this is also the default)
        threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
        t_clust[task], clusters[task], p_values[task], H0[task] = permutation_cluster_1samp_test(
            data-.5, n_jobs=1, threshold=threshold, adjacency=None,
            n_permutations="all")
        # Put the cluster data in a viewable format
        temp_p_clust = np.ones((1,300))
        for cl, p in zip(clusters[task], p_values[task]):
            temp_p_clust[cl] = p
        p_clust[task][roi] = temp_p_clust.T
        

times = np.arange(-300,900,4)

for task in p_clust.keys():
    for roi in p_clust[task].columns:
        print(f"{task, roi}: Decoding semantic category at timepoints: \
              {times[np.where(p_clust[task][roi] < 0.05)[0]]}")
        scores[task][roi].shape = (18, 300)

task = 'sd'
for i in range(0, 18):
    c = 0
    for roi in ['PVA']:
        sns.lineplot(x=times, y=scores2[i][task][roi], color=colors[c])
        plt.axvline(0, color="k");
        plt.title(f"{i, roi}")
        plt.axhline(0.5, color='k', linestyle='--')
        c+=1
        plt.show()
        
        
        
confmat = []
for sub in range(0, 18):
   with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/confusion_matrix_concrete_{sub}.P",
              'rb') as f:
        confmat.append(pickle.load(f))

new_mat = dict.fromkeys(['sd', 'ld'])

for task in new_mat.keys():
    new_mat[task] = dict.fromkeys(kkROI)
    for roi in new_mat[task].keys():
        new_mat[task][roi] = []
        for i in range(0, 18):
            new_mat[task][roi].append(confmat[i][task][roi])
        new_mat[task][roi] = np.array(new_mat[task][roi]).mean(axis=0)
        
        
times = np.arange(-300,900,4)



clusters = dict.fromkeys(kkROI)
clusters['lATL'] = [((int(np.where(times==236)[0])), \
                      (int(np.where(times==504)[0]))), \
                    ((int(np.where(times==588)[0])), \
                      (int(np.where(times==652)[0])))]
clusters['rATL'] = [((int(np.where(times==264)[0])), \
                          (int(np.where(times==488)[0])))]
clusters['AG'] = [((int(np.where(times==312)[0])), \
                          (int(np.where(times==372)[0])))]
clusters['PTC'] = [((int(np.where(times==208)[0])), \
                      (int(np.where(times==460)[0]))), \
                    ((int(np.where(times==468)[0])), \
                      (int(np.where(times==556)[0])))]
clusters['IFG'] = [((int(np.where(times==260)[0])), \
                      (int(np.where(times==496)[0])))]
clusters['PVA'] = [((int(np.where(times==64)[0])), \
                      (int(np.where(times==128)[0]))), \
                    ((int(np.where(times==156)[0])), \
                      (int(np.where(times==240)[0]))), \
                    ((int(np.where(times==252)[0])), \
                      (int(np.where(times==344)[0])))]
    
for roi in kkROI:
    i = 0
    for cluster in clusters[roi]:
        i += 1
        get_clust = np.stack(new_mat['sd'][roi][cluster[0]:cluster[1]]).mean(axis=0) # normalize over true values
            # for the true values check the proportions of each category

        ax=sns.heatmap(get_clust,
                    annot=True,
                    xticklabels=kk2,
                    yticklabels=kk2,
                    cmap="viridis",)
        ax.set(xlabel="Predicted", ylabel="True")
        ax.set_title(f"{roi}, cluster {str(i)}")
        plt.show()
    
 

