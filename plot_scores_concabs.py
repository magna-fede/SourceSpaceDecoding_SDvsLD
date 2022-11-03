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
    with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/abs_balanced_scores_{i}.P" , "rb") as f:
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
        #scores[task][roi].shape = (18, 300)

for task in ['LD', 'sd']:
    i = 0
    for roi in scores[task].keys():
        sns.lineplot(x=times, y=np.vstack(scores[task][roi]).mean(axis=0), color=colors[i])
        plt.fill_between(x=times, \
                          y1=(np.vstack(scores[task][roi]).mean(axis=0) - \
                              sem(np.vstack(scores[task][roi]))), \
                          y2=(np.vstack(scores[task][roi]).mean(axis=0) + \
                              sem(np.vstack(scores[task][roi]))), \
                          color=colors[i], alpha=.1)
        plt.axvline(0, color="k");
        mask = p_clust[task][roi] < 0.05
        mask = mask.values.reshape(300)
        first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
        last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions
        i+=1
        for start, stop in zip(first_vals, last_vals):
            plt.axvspan(times[start], times[stop], alpha=0.5,
                        label="Cluster based permutation p<.05",
                        color="green")
        # plt.legend();
        mask = stats.ttest_1samp(np.vstack(scores[task][roi]), .5)[1] < 0.05
        mask[0] = False # force first timepoint to be false otherwirse the axvspan might be reserved
                        # this is because first_vals requires
        first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
        last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions
    
        for start, stop in zip(first_vals, last_vals):
            plt.axvspan(times[start], times[stop], alpha=0.2,
                        label="Cluster based permutation p<.05",
                        color="yellow")    
        #plt.title(f"{task, roi} Semantic Category Decoding")
        plt.axhline(.5, color="k", linestyle="--", label="chance");
        # plt.legend();
        mask = p_clust[task][roi] < 0.05/6
        mask = mask.values.reshape(300)
        first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
        last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

        for start, stop in zip(first_vals, last_vals):
            plt.axvspan(times[start], times[stop], ymax=0.05, alpha=0.6, color="red",
                        label="Bonferroni-correction per 6 ROIs")
        plt.tight_layout()
        plt.savefig(f"//imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/Figures/{task}_{roi}_accuracy_balanced.png", format="png");
        plt.show();
 
 

