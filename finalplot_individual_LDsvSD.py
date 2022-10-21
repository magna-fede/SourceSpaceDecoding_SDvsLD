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

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

sns.set_theme(context="notebook",
              style="white",
              font="sans-serif")

sns.set_style("ticks")

with open("/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/LDvsSD/scores.P" , 'rb') as f:
    scores = pickle.load(f)

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
    for roi in scores[task].keys():
        scores[task][roi] = np.array(scores[task][roi])
        
# initialise average(scores) key
scores['avg'] = [ [] for _ in range(len(scores)) ]

# calcualte average performance for each participant, across tasks
for i in range(0, 18):
    for roi in kkROI:
        scores['avg'][roi].append(np.array([scores['mlk'][roi][i],
                              scores['frt'][roi][i],
                              scores['odr'][roi][i]]).mean(axis=0))
        
for roi in scores['avg'].keys():
   scores['avg'][roi] = np.array(scores['avg'][roi])


for task in scores.keys():
    i = 0
    for roi in scores[task].keys():
    # iter to select colours
    # average plot all ROIs in one plot
        # plot the average score across task (= scores['avg'], and across participants)    
        sns.lineplot(x=times, y=np.stack(scores[task][roi]).mean(axis=0), color=colors[i], label=roi)
        # plot the standard error of the mean
        plt.fill_between(x=times, \
                          y1=(np.stack(scores[task][roi]).mean(axis=0)) - sem(np.stack(scores[task][roi]),0), \
                          y2=(np.stack(scores[task][roi]).mean(axis=0)) + sem(np.stack(scores[task][roi]),0), \
                          color=colors[i], alpha=.1)
        i += 1
    # plot some line that are useful for inspection
    plt.axvline(0, color='k');
    #plt.title(f'LD vs {task} Decoding ROC AUC')
    plt.axhline(.5, color='k', linestyle='--');
    plt.legend();
    #plt.savefig(f'//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/LDvsSD/Figures/{task}_LDvs_accuracy.png', format='png')
    plt.show();