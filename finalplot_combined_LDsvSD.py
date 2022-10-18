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

# 
kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

with open("//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/LDvsSD/scores.P" , 'rb') as f:
    scores = pickle.load(f)

with open("//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/LDvsSD/patterns.P" , 'rb') as f:
    patterns = pickle.load(f)

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

for task in scores.keys():
    scores[task] = np.array(scores[task])
        
# initialise average(scores) key
scores['avg'] = [ [] for _ in range(len(scores)) ]

# calcualte average performance for each participant, across tasks
for i in range(0, 18):
    scores['avg'].loc[i] = np.array([scores['mlk'][i],
                              scores['frt'][i],
                              scores['odr'][i]]).mean(axis=0)
        
# for roi in scores['avg'].keys():
#    scores['avg'][roi] = np.array(scores['avg'][roi])


i = 0
for task in scores.keys():
    # iter to select colours
    # average plot all ROIs in one plot
        # plot the average score across task (= scores['avg'], and across participants)    
    sns.lineplot(x=times, y=np.stack(scores[task]).mean(axis=0), color=colors[i], label=task)
    # plot the standard error of the mean
    plt.fill_between(x=times, \
                      y1=(np.stack(scores[task]).mean(axis=0)) - sem(np.stack(scores[task]),0), \
                      y2=(np.stack(scores[task]).mean(axis=0)) + sem(np.stack(scores[task]),0), \
                      color=colors[i], alpha=.1)
    i += 2
    # plot some line that are useful for inspection
plt.axvline(0, color='k');
plt.title('LD vs average(SD) Decoding ROC AUC')
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.legend();
#plt.savefig('//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/LDvsSD/Figures/average_LDvsSD_accuracy.png', format='png')
plt.show();

patterns_roi = dict.fromkeys(kkROI)

for roi in patterns_roi.keys():
    patterns_roi[roi] = dict.fromkeys(['frt', 'mlk', 'odr'])
    for task in patterns_roi[roi].keys():
        patterns_roi[roi][task] = []

# calculate the ROOT-MEAN-SQUARE for each pattern in each task
# loop over participants    
for i in range(18):
    # loop over each roi
    for roi in patterns_roi.keys():
        # loop over each task
        for task in patterns_roi[roi].keys():
            patterns_roi[roi][task].append(rms(np.array(patterns[task][i].loc[roi])))

for roi in patterns_roi.keys():
    patterns_roi[roi]['avg'] = []

# calculate the average of the RMS(pattern) across each task

# loop over participants    
for i in range(18):
    # loop over each roi
    for roi in patterns_roi.keys():
        patterns_roi[roi]['avg'].append(np.array([patterns_roi[roi]['mlk'][i],
                                 patterns_roi[roi]['frt'][i],
                                 patterns_roi[roi]['odr'][i]]).mean(axis=0))    

i = 0
for roi in patterns_roi.keys():
    sns.lineplot(x=times, y=np.array(patterns_roi[roi]['avg']).mean(axis=0), color=colors[i]) # this takes mean over participants
    i += 1
plt.axvline(0, color='k');
plt.title('LD vs average(SD) RMS patterns')
plt.legend(patterns_roi.keys());
plt.savefig('//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/LDvsSD/Figures/average_LDvsSD_patterns.png', format='png')
plt.show();
