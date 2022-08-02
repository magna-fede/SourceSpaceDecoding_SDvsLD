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

kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

with open("//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SDvsSD/scores.P" , 'rb') as f:
    scores = pickle.load(f)

# # create times array
times = np.arange(-300,900,4)

colors = sns.color_palette(['#FFBE0B',
                            '#FB5607',
                            '#FF006E',
                            '#8338EC',
                            '#3A86FF',
                            '#1D437F'
                            ])

for roi in scores.keys():
    scores[roi] = np.array(scores[roi])
        

# iter to select colours
i = 0

# average plot all ROIs in one plot
for roi in scores.keys():
    # plot the average score across task (= scores['avg'], and across participants)    
    sns.lineplot(x=times, y=np.array(scores[roi]).mean(axis=0), color=colors[i], label=roi)
# plot the standard error of the mean
    plt.fill_between(x=times, \
                  y1=(np.mean(np.array(scores[roi]),0) - sem(np.vstack(scores[roi]),0)), \
                  y2=(np.mean(np.array(scores[roi]),0) + sem(np.vstack(scores[roi]),0)), \
                  color=colors[i], alpha=.1)
    i+=1
# plot some line that are useful for inspection
plt.axvline(0, color='k');
plt.title('multiclass SD Decoding ROC AUC')
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.legend();
plt.savefig('//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SDvsSD/Figures/average_SDvsSD_accuracy.png', format='png')
plt.show();


