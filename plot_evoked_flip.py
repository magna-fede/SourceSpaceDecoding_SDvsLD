#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:44:03 2023

@author: fm02
"""

import numpy as np
import pandas as pd
import pickle
import os

import mne
from mne.minimum_norm import read_inverse_operator

import warnings
warnings.filterwarnings('ignore')

path = '/imaging/hauk/users/fm02/Decoding_SDLD'
os.chdir(path)

import seaborn as sns
import matplotlib.pyplot as plt

from Setareh.SN_semantic_ROIs import SN_semantic_ROIs

# list subjects directory
from Setareh.SN_semantic_ROIs import subjects

# subjects' MRI directories
from Setareh.SN_semantic_ROIs import subjects_mri

data_path = '/imaging/hauk/users/sr05/SemNet/SemNetData/'
main_path = '/imaging/hauk/users/rf02/Semnet/'

roi_sem = SN_semantic_ROIs()


with open("/imaging/hauk/users/fm02/first_output/evoked_responses/2706_evoked_flipped.P", 'rb') as f:
      evoked = pickle.load(f)

# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/evoked_responses/0722_evoked_RMSacrossROI.P", 'rb') as f:
#       evoked = pickle.load(f)      

colors = sns.color_palette(['#FFBE0B',
                            '#FB5607',
                            '#FF006E',
                            '#8338EC',
                            '#3A86FF',
                            '#1D437F'
                            ])

# for roi in kkROI: 
#     sns.set(style="ticks", rc={"lines.linewidth": 0.9,
#                                 'figure.figsize':(15,10)})
#     for vertex in avg_LD[roi][0]:
#         sns.lineplot(x=np.arange(-300,900,4), y=vertex)
#     plt.title(roi)
#     plt.show();      

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
            tasks[task][roi].append(np.array(evoked[task][i][roi]))

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
        tasks[task][roi].shape =(18, 300)        

for roi in kkROI:
    fig = plt.subplots()
    for task in tasks.keys(): 
        sns.lineplot(x=np.arange(-300,900,4),y=np.array(tasks[task][roi]).mean(0))
        plt.fill_between(x=np.arange(-300,900,4), \
                          y1=(tasks[task][roi].mean(0) - sem(tasks[task][roi], 0)), \
                          y2=(tasks[task][roi].mean(0) + sem(tasks[task][roi], 0)), \
                          alpha=.1)
    plt.axvline(0, color='k');
    plt.axhline(0, color='k', alpha=0.3, linewidth = 0.5);
    plt.title(f"RMS evoked response {roi}")
    plt.legend(tasks.keys())
    #plt.savefig(f'RMS evoked_{roi}.png', format='png')    
    plt.show()
    
for roi in kkROI:
    fig, ax = plt.subplots(figsize=(6,4))
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
    plt.title(f"{roi}")
    # handles, labels = plt.gca().get_legend_handles_labels()
    # labels = ['LD', 'SD 1 - "milk"', 'SD 2 - "fruit"', 'SD 3 - "odour"', 'SD']
    # by_label = dict(zip(labels, handles))
    # legend = plt.legend(by_label.values(), by_label.keys(), ncol=5, framealpha=1, frameon=True)
    leg = plt.legend()
    ax.get_legend().set_visible(False) 
    plt.tight_layout()
    # plt.savefig(f'/imaging/hauk/users/fm02/final_dTtT/evoked/flip_avg/flipped evoked_{roi}_nolegend.png', format='png')    
    plt.show()

# run only legend = ... line to plot the legend only