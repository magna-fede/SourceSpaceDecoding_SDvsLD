# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:04:54 2022

@author: fm02
"""
### Author: federica.magnabosco@mrc-cbu.cam.ac.uk
### Plot results from LDvsSD individual ROIs accuracy


import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

scores = []

for i in range(0, 18):
    with open(f"//imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/{i}_scores_concat.P" , 'rb') as f:
        scores.append(pickle.load(f))

# # create times array
times = np.arange(-300,900,4)

colors = sns.color_palette(['#FFBE0B',
                            '#FB5607',
                            '#FF006E',
                            '#8338EC',
                            '#3A86FF',
                            '#1D437F'
                            ])

reorg = dict.fromkeys(['ld', 'sd'])

reorg['ld'] = []
reorg['sd'] = []

for sub_score in scores:
    reorg['ld'].append(sub_score['ld'])
    reorg['sd'].append(sub_score['sd'])
    
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
p_clust

for task in scores.keys():
    p_clust[task] = pd.DataFrame(index=range(300), columns=kkROI)
    for roi in scores[task].keys():
        # Reshape data to what is equivalent to (n_samples, n_space, n_time)
        scores[task][roi].shape = (18, 1, 300)
        # Compute threshold from t distribution (this is also the default)
        threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
        t_clust[task], clusters[task], p_values[task], H0[task] = permutation_cluster_1samp_test(
            scores[task][roi]-.5, n_jobs=1, threshold=threshold, adjacency=None,
            n_permutations='all')
        # Put the cluster data in a viewable format
        temp_p_clust = np.ones((1,300))
        for cl, p in zip(clusters[task], p_values[task]):
            temp_p_clust[cl] = p
        p_clust[task][roi] = temp_p_clust.T
        

times = np.arange(-300,900,4)

for task in p_clust.keys():
    for roi in p_clust[task].columns:
        print(f'{task, roi}: Decoding semantic category at timepoints: \
              {times[np.where(p_clust[task][roi] < 0.05)[0]]}')
        scores[task][roi].shape = (18, 300)

for task in scores.keys():
    i = 0
    for roi in scores[task].keys():
        sns.lineplot(x=times, y=np.mean(scores[task][roi],0), color=colors[i])
        plt.fill_between(x=times, \
                          y1=(np.mean(scores[task][roi],0)-sem(scores[task][roi],0)), \
                          y2=(np.mean(scores[task][roi],0)+sem(scores[task][roi],0)), \
                          color=colors[i], alpha=.1)
        plt.axvline(0, color='k');
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
        mask = stats.ttest_1samp(scores[task][roi], .5)[1] < 0.05
        first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
        last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions
    
        for start, stop in zip(first_vals, last_vals):
            plt.axvspan(times[start], times[stop], alpha=0.2,
                        label="Cluster based permutation p<.05",
                        color="yellow")    
        #plt.title(f'{task, roi} Semantic Category Decoding')
        plt.axhline(.5, color='k', linestyle='--', label='chance');
        #plt.savefig(f'//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/Figures/{task}_{roi}_accuracy.png', format='png');
        # plt.legend();
        plt.show();

SD_mean = dict.fromkeys(kkROI)
for roi in kkROI:  
    SD_mean[roi] = np.mean(np.array([scores['mlk'][roi], 
                                  scores['frt'][roi], 
                                  scores['odr'][roi] ]),
                      axis=0)
    
    SD_mean[roi].shape = (18, 1, 300)


SDp_clust = pd.DataFrame(index=range(300),columns=kkROI)

for roi in kkROI:
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    SD_mean[roi].shape = (18, 1, 300)
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
    SDt_clust, SDclusters, SDp_values, H0 = permutation_cluster_1samp_test(
        SD_mean[roi]-.5, n_jobs=1, threshold=threshold, adjacency=None,
        n_permutations='all')
    # Put the cluster data in a viewable format
    temp_p_clust = np.ones((1,300))
    for cl, p in zip(SDclusters, SDp_values):
        temp_p_clust[cl] = p
    SDp_clust[roi] = temp_p_clust.T
    
i = 0    
for roi in kkROI:
    SD_mean[roi] = np.array(SD_mean[roi]).reshape((18,300))
    
    sns.lineplot(x=times, y=np.mean(SD_mean[roi], 0), color=colors[i])
    plt.fill_between(x=times, \
                      y1=(np.mean(SD_mean[roi],0)-sem(SD_mean[roi],0)), \
                      y2=(np.mean(SD_mean[roi],0)+sem(SD_mean[roi],0)), \
                      color=colors[i], alpha=.1)
    plt.axvline(0, color='k');
    plt.axhline(.5, color='k', linestyle='--', label='chance');
    # if len(times[np.where(SDp_clust[roi] < 0.05)]) > 0:
    #     for tp in times[np.where(SDp_clust[roi] < 0.05)]:
    #         plt.axvspan(tp,tp,
    #                 label="Cluster based permutation p<.05",
    #                 color="green", alpha=0.3)
    mask = SDp_clust[roi] < 0.05
    mask = mask.values.reshape(300)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions
    i+=1
    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                    label="Cluster based permutation p<.05",
                    color="green")
    plt.title('SD_avg Semantic Category Decoding'); 
    # plt.legend();
    mask = stats.ttest_1samp(SD_mean[roi], .5)[1] < 0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                    label="Cluster based permutation p<.05",
                    color="yellow")
    #plt.title(f'avg_SD {roi} Semantic Category Decoding')
    #plt.legend();
    #plt.savefig(f'//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/Figures/avgSD_{roi}_accuracy.png', format='png')
    plt.show();
    
# times[np.where(stats.ttest_1samp(SD_mean[roi], .2)[1] < 0.05)]
