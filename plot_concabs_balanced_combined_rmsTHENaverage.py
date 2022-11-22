#!/usr/bin/env python
# coding: utf-8

### Author: federica.magnabosco@mrc-cbu.cam.ac.uk
### Fit decoding model LDvsSD individual ROIs and save accuracy

# Import some relevant packages.
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


kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']
kk2 = ['visual', 'hand', 'hear', 'abstract']

def rms(example):
    """Compute root mean square of each ROI.
    Input is a dataframe of length=n_vertices."""
    # first transform Series in np array of dimension n_vertics*timepoints, when the input is unstacked.
    example = np.vstack(np.array(example))
    # create np.array where to store info
    rms_example = np.zeros(example.shape[1])
    # loop over timepoints
    for i in np.arange(0,example.shape[1]):
        rms_example[i] = np.sqrt(np.mean(example[:,i]**2))
    
    return rms_example 


scores = []
for i in range(0, 18):
    with open(f"/imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SemCat/abs_balanced_scores_{i}.P", 'rb') as f:
        scores.append(pickle.load(f))

patterns = []
for i in range(0, 18):
    with open(f"/imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SemCat/abs_balanced_patterns_{i}.P", 'rb') as f:
        patterns.append(pickle.load(f))

# delete useless wrong keys
for score in scores:
    del([score['mlk'],
         score['frt'],
         score['odr']])
for pattern in patterns:
    del([pattern['mlk'],
         pattern['frt'],
         pattern['odr']])    
    
def reorg_scores(sbj_list):
    tasks = sbj_list[0].keys()
    reorg = dict.fromkeys(tasks)
    for task in reorg.keys():
        reorg[task] = []
    for participant in sbj_list:
        for task in reorg.keys():
            reorg[task].append(participant[task])
    
    final = dict.fromkeys(tasks)
    for task in reorg:
        final[task] = np.array(reorg[task])
        
    final['SD'] = np.stack([final['milk'], final['fruit'], final['odour']]).mean(axis=0)
    return final
    
def reorg_patterns(sbj_list):
    tasks = sbj_list[0].keys()
    reorg = dict.fromkeys(tasks)
    for task in reorg.keys():
        reorg[task] = dict.fromkeys(kkROI)
        for roi in reorg[task].keys():
            reorg[task][roi] = dict.fromkeys(sbj_list[0][task].keys())
            for semcat in reorg[task][roi].keys():
                reorg[task][roi][semcat] = []
                
            
    for participant in sbj_list:
        for task in reorg.keys():
            for roi in reorg[task].keys():
                for semcat in reorg[task][roi].keys():
                    reorg[task][roi][semcat].append(rms(participant[task][semcat].loc[roi]))
    for task in reorg.keys():
        for roi in reorg[task].keys():
            reorg[task][roi]['avg'] = []
    for i in range(18):
        for task in reorg.keys():
            for roi in reorg[task].keys():
                temp = np.stack([reorg[task][roi]['abstract'][i], 
                                 reorg[task][roi]['hear'][i],
                                 reorg[task][roi]['visual'][i],
                                 reorg[task][roi]['hand'][i]]).mean(axis=0)
                reorg[task][roi]['avg'].append(temp)
    
    reorg['SD'] = dict.fromkeys(kkROI)
    for roi in reorg['SD'].keys():
        reorg['SD'][roi] = dict.fromkeys(['avg'])
        reorg['SD'][roi]['avg'] = []
    for i in range(18):
        for roi in reorg['SD'].keys():
            temp = np.stack([reorg['milk'][roi]['avg'][i],
                             reorg['fruit'][roi]['avg'][i],
                             reorg['odour'][roi]['avg'][i]]).mean(axis=0)
            reorg['SD'][roi]['avg'].append(temp)
                
    return reorg


scores = reorg_scores(scores)
patterns = reorg_patterns(patterns)

# # create times array
times = np.arange(-300,900,4)

colors = sns.color_palette(['#FFBE0B',
                            '#FB5607',
                            '#FF006E',
                            '#8338EC',
                            '#3A86FF',
                            '#1D437F',
                            '#1D437F'
                            ])

# IBM colorblind palette
# '#648fff'
# '#785ef0'
# '#dc267f'
# '#fe6100'
# '#ffb000'
# '#000000'
# '#ffffff'

p_clust = {}
t_clust = {}
clusters = {}
p_values = {}
H0 = {} 
p_clust

for task in ['LD', 'SD']:
    p_clust[task] = pd.DataFrame(index=range(300))
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    score = np.array(scores[task]).reshape(18,1,300)
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
    t_clust[task], clusters[task], p_values[task], H0[task] = permutation_cluster_1samp_test(
        score-.5, n_jobs=1, threshold=threshold, adjacency=None,
        n_permutations='all')
    # Put the cluster data in a viewable format
    temp_p_clust = np.ones((1,300))
    for cl, p in zip(clusters[task], p_values[task]):
        temp_p_clust[cl] = p
    p_clust[task] = temp_p_clust.T
        

times = np.arange(-300,900,4)

for task in p_clust.keys():
    print(f'{task}: Decoding semantic category at timepoints: \
          {times[np.where(p_clust[task] < 0.05)[0]]}')

for task in ['LD', 'SD']:
    sns.lineplot(x=times, y=np.mean(np.array(scores[task]),0), color='k')
    plt.fill_between(x=times, \
                      y1=(np.mean(np.array(scores[task]),0)-sem(np.array(scores[task]),0)), \
                      y2=(np.mean(np.array(scores[task]),0)+sem(np.array(scores[task]),0)), \
                      color='k', alpha=.1)
    plt.axvline(0, color='k');
    mask = p_clust[task] < 0.05
    mask = mask.reshape(300)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                    label="Cluster based permutation p<.05",
                    color="green")
    # plt.legend();
    mask = stats.ttest_1samp(scores[task], .5)[1] < 0.05
    mask[0] = False
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                    label="uncorrected p<.05",
                    color="yellow")    
    #plt.title(f'{task} Semantic Category Decoding ROC AUC')
    plt.axhline(.5, color='k', linestyle='--', label='chance');
    plt.savefig(f'/imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SemCat/Figures/{task}_accuracy_balanced.png', format='png');
    # plt.legend();
    plt.show();
        

newp = dict.fromkeys(patterns.keys())
for task in patterns.keys():
    newp[task] = dict.fromkeys(kkROI)
    for roi in newp[task].keys():
        newp[task][roi] = []

for task in newp.keys():
    for i, sbj in enumerate(patterns[task]):
        for roi in kkROI:
            newp[task][roi].append(rms(np.array(sbj.loc[roi])))


        
for task in patterns.keys():        
    i = 0
    for roi in newp[task].keys():
        sns.lineplot(x=times,
                     y=np.array(patterns[task][roi]['avg']).mean(axis=0), 
                     color=colors[i]) # this takes mean over participants
        i += 1
    plt.axvline(0, color='k');
    #plt.title(f'{task} RMS patterns')
    plt.legend(newp[task].keys(), loc='upper left');
    plt.savefig(f'/imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SemCat/Figures/{task}_patterns_balanced.png', format='png')
    plt.show();