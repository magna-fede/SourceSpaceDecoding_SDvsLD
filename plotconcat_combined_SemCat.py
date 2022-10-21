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
kk2 = ['visual', 'hand', 'hear', 'neutral', 'emotional']

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
    with open(f"/imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SemCat/scores_concat_{i}.P" , 'rb') as f:
        scores.append(pickle.load(f))

patterns = []
for i in range(0, 18):
    with open(f"/imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SemCat/patterns_concat_{i}.P" , 'rb') as f:
        patterns.append(pickle.load(f))

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


reorg = dict.fromkeys(["ld", "sd"])

reorg["ld"] = []
reorg["sd"] = []

for sub_score in scores:
    reorg["ld"].append(sub_score["ld"])
    reorg["sd"].append(sub_score["sd"])
    
del(scores)
scores = dict.fromkeys(reorg.keys())
for task in reorg:
    scores[task] = pd.DataFrame(reorg[task], index=range(0,18))


reorg = dict.fromkeys(["ld", "sd"])

reorg["ld"] = []
reorg["sd"] = []

for sub_score in patterns:
    reorg["ld"].append(sub_score["ld"])
    reorg["sd"].append(sub_score["sd"])
    
patterns = dict.fromkeys(reorg.keys())
for task in reorg:
    patterns[task] = reorg[task]


p_clust = {}
t_clust = {}
clusters = {}
p_values = {}
H0 = {} 
p_clust

for task in ['ld', 'avg']:
    p_clust[task] = pd.DataFrame(index=range(300))
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    score = np.stack(scores[task]).reshape(18,1,300)
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

for task in ['ld', 'avg']:
    sns.lineplot(x=times, y=np.mean(np.stack(scores[task]),0), color='k')
    plt.fill_between(x=times, \
                      y1=(np.mean(np.stack(scores[task]),0)-sem(np.stack(scores[task]),0)), \
                      y2=(np.mean(np.stack(scores[task]),0)+sem(np.stack(scores[task]),0)), \
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
    mask = stats.ttest_1samp(np.stack(scores[task]), .5)[1] < 0.05
    mask[0] = False
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                    label="uncorrected p<.05",
                    color="yellow")    
    #plt.title(f'{task} Semantic Category Decoding ROC AUC')
    plt.axhline(.5, color='k', linestyle='--', label='chance');
    #plt.savefig(f'//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SemCat/Figures/{task}_accuracy.png', format='png');
    # plt.legend();
    plt.show();
        

patterns_roi = dict.fromkeys(kkROI)

for roi in patterns_roi.keys():
    patterns_roi[roi] = dict.fromkeys(['ld', 'sd'])
    for task in patterns_roi[roi].keys():
        patterns_roi[roi][task] = []

# calculate the ROOT-MEAN-SQUARE for each pattern in each task
# loop over participants    
for i in range(18):
    # loop over each roi
    for roi in patterns_roi.keys():
        # loop over each task
        for task in patterns_roi[roi].keys():
            cat = rms(np.array(patterns[task][i].loc[roi]))
            patterns_roi[roi][task].append(np.stack(cat))

# calculate the average of the RMS(pattern) across each task

# loop over participants    
 

i = 0
for roi in patterns_roi.keys():
    sns.lineplot(x=times, y=np.array(patterns_roi[roi]['sd']).mean(axis=0), color=colors[i]) # this takes mean over participants
    i += 1
plt.axvline(0, color='k');
#plt.title('average SD RMS patterns')
plt.legend(patterns_roi.keys(), loc='upper left');
#plt.savefig('//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SDvsSD/Figures/SDvsSD_patterns.png', format='png')
plt.show();

i = 0
for roi in patterns_roi.keys():
    sns.lineplot(x=times, y=np.array(patterns_roi[roi]['ld']).mean(axis=0), color=colors[i]) # this takes mean over participants
    i += 1
plt.axvline(0, color='k');
plt.title('LD RMS patterns')
plt.legend(patterns_roi.keys(), loc= 'upper left');
#plt.savefig('//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SDvsSD/Figures/SDvsSD_patterns.png', format='png')
plt.show();