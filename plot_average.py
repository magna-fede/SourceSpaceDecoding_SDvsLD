#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 18:26:45 2022

@author: fm02
"""
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import f_oneway

from mne.stats import permutation_cluster_test
from mne.stats import permutation_cluster_1samp_test
   
def rms(example):
    """Compute root mean square of each ROI.
    Input is a dataframe of length=n_vertices."""
    # first transform Series in np array of dimension n_vertics*timepoints
    
    return np.sqrt(np.mean(example**2))

    
with open("/imaging/hauk/users/fm02/first_output/evoked_responses/0804_evoked_AVGacrossROI.P", 'rb') as f:
      evoked = pickle.load(f)


colors = sns.color_palette(['#FFBE0B',
                            '#FB5607',
                            '#FF006E',
                            '#8338EC',
                            '#3A86FF',
                            '#1D437F'
                            ])


      
from scipy.stats import sem    


tasks = dict.fromkeys(['LD', 'MLK', 'FRT', 'ODR'])

kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

times = np.arange(-300,900,4)

for task in tasks:
    tasks[task] = dict.fromkeys(kkROI)
    for roi in kkROI:
        tasks[task][roi] = []

# for each participant, calculate average in each task, in each roi 
for i in range(18):
    for task in tasks:
        for roi in kkROI:
            #tasks[task][roi].append(np.apply_along_axis(rms, 0, evoked[task][i][roi][0]))
            tasks[task][roi].append(np.array(evoked[task][i][roi]).mean(0))
            #tasks[task][roi].append(np.array(evoked[task][i][roi]))
            
tasks['avg_SD'] = dict.fromkeys(kkROI)
for roi in kkROI:
    tasks['avg_SD'][roi] = []

for i in range(18):
    for roi in kkROI:
        # numpy array of size 3(tasks)*300(timepoints)
        all_sds = np.array([tasks['FRT'][roi][i], tasks['ODR'][roi][i], tasks['MLK'][roi][i]])
        # average across SD tasks, for each participant
        tasks['avg_SD'][roi].append(all_sds.mean(0))

for task in tasks.keys():
    for roi in kkROI:
        tasks[task][roi] = np.array(tasks[task][roi])
        tasks[task][roi].shape =(18, 1, 300) 


LDvsSD = dict.fromkeys(kkROI)

for roi in kkROI:
    t_obs, Lclusters, Lp_values, H0 = permutation_cluster_1samp_test(
        (tasks['LD'][roi] - tasks['avg_SD'][roi]),
        n_permutations=5000,
        n_jobs=1)
    Lp_clust = np.ones((1,300))
    for cl, p in zip(Lclusters, Lp_values):
        Lp_clust[cl] = p
    LDvsSD[roi] = Lp_clust.T

for task in tasks.keys():
    for roi in kkROI:
        tasks[task][roi].shape =(18, 300) 

for roi in kkROI:
    sns.lineplot(times, tasks['LD'][roi].mean(0), label='LD', color=colors[1])
    sns.lineplot(times, tasks['avg_SD'][roi].mean(0), label='average SD', color=colors[3])    
    plt.fill_between(x=times, \
                      y1=(tasks['LD'][roi].mean(0) - sem(tasks['LD'][roi],0)), \
                      y2=(tasks['LD'][roi].mean(0) + sem(tasks['LD'][roi],0)), \
                      color=colors[1], alpha=.1)
    plt.fill_between(x=times, \
                      y1=(tasks['avg_SD'][roi].mean(0) - sem(tasks['avg_SD'][roi],0)), \
                      y2=(tasks['avg_SD'][roi].mean(0) + sem(tasks['avg_SD'][roi],0)),                    
                      color=colors[3], alpha=.1)
    plt.axvline(0, color='k');
    plt.axhline(0, color='k', alpha=0.3, linewidth = 0.5);
    mask = (LDvsSD[roi] < 0.05).reshape(-1)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                    label="Cluster-based permutation p<.05",
                    color="green")
        
    # plt.legend();
    mask = stats.ttest_1samp(tasks['LD'][roi] - tasks['avg_SD'][roi], popmean=0)[1] < 0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                    label="uncorrected p<.05",
                    color="yellow")    
    plt.title(f'Flipped LD vs SD - {roi}')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(f'/imaging/hauk/users/fm02/final_dTtT/evoked/avg/evoked_LDvsSD_{roi}.png', format='png')
    plt.show();


SDvsSD = dict.fromkeys(kkROI)

for roi in kkROI:
    F_obs, SDclusters, SDp_values, H0 = permutation_cluster_test(
        ([tasks['MLK'][roi], tasks['ODR'][roi], tasks['FRT'][roi]]),
        n_permutations=5000,
        n_jobs=1)
    SDp_clust = np.ones(300)
    for cl, p in zip(SDclusters, SDp_values):
        SDp_clust[cl] = p
    SDvsSD[roi] = SDp_clust.T

for task in tasks.keys():
    for roi in kkROI:
        tasks[task][roi].shape =(18, 300) 

for roi in kkROI:
    sns.lineplot(times, tasks['MLK'][roi].mean(0), label='SD milk', color=colors[0])
    sns.lineplot(times, tasks['ODR'][roi].mean(0), label='SD odour', color=colors[2])
    sns.lineplot(times, tasks['FRT'][roi].mean(0), label='SD fruit', color=colors[4])
    plt.fill_between(x=times, \
                      y1=(tasks['MLK'][roi].mean(0) - sem(tasks['MLK'][roi], 0)), \
                      y2=(tasks['MLK'][roi].mean(0) + sem(tasks['MLK'][roi], 0)), \
                      color=colors[0], alpha=.1)
    plt.fill_between(x=times, \
                      y1=(tasks['ODR'][roi].mean(0) - sem(tasks['ODR'][roi], 0)), \
                      y2=(tasks['ODR'][roi].mean(0) + sem(tasks['ODR'][roi], 0)), \
                      color=colors[2], alpha=.1)
    plt.fill_between(x=times, \
                      y1=(tasks['FRT'][roi].mean(0) - sem(tasks['FRT'][roi], 0)), \
                      y2=(tasks['FRT'][roi].mean(0) + sem(tasks['FRT'][roi], 0)), \
                      color=colors[4], alpha=.1)
    plt.axvline(0, color='k');
    plt.axhline(0, color='k', alpha=0.3, linewidth = 0.5);
    mask = (SDvsSD[roi] < 0.05).reshape(-1)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                    label="Cluster based permutation p<.05",
                    color="green")
        
    # plt.legend();
    mask = f_oneway(tasks['MLK'][roi], tasks['ODR'][roi], tasks['FRT'][roi], axis =0)[1] <0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                    label="uncorrected p<.05",
                    color="yellow")    
    plt.title(f'Flipped SD vs SD - {roi}')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(f'/imaging/hauk/users/fm02/final_dTtT/evoked/avg/evoked_SDvsSD_{roi}.png', format='png')
    # plt.legend();
    plt.show();
           
 
for roi in kkROI:
    i = 0
    for task in tasks.keys(): 
        sns.lineplot(x=np.arange(-300,900,4),y=np.array(tasks[task][roi]).mean(0), color=colors[i], label=task)
        plt.fill_between(x=np.arange(-300,900,4), \
                          y1=(tasks[task][roi].mean(0) - sem(tasks[task][roi], 0)), \
                          y2=(tasks[task][roi].mean(0) + sem(tasks[task][roi], 0)),  \
                          color=colors[i], alpha=.1)
        i+=1
    plt.axvline(0, color='k');
    plt.axhline(0, color='k', alpha=0.3, linewidth = 0.5);
    plt.title(f"Flipped evoked response {roi}")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(f'/imaging/hauk/users/fm02/final_dTtT/evoked/avg/evoked_{roi}.png', format='png')    
    plt.show()