#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 12:49:49 2022

@author: fm02
"""
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']
kk2 = ['visual', 'hand', 'hear', 'neutral', 'emotional']

import sys


data_semcat = dict()
data_sdld = dict()

for i in range(0, 18):
    with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/scores_TimeGen_{i}_LG.P", "rb") as sbj_data:
        data_semcat[i] = pickle.load(sbj_data)

for i in range(0, 18):
    with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/LDvsSD/scores_TimeGen_{i}_LG.P", "rb") as sbj_data:
        data_sdld[i] = pickle.load(sbj_data)

times = np.arange(-300,900,4)

SD_mats = dict.fromkeys(kkROI)

value = input("Do you want semcat or sdld? \n")

if value=='semcat':
    data = data_semcat
elif value=='sdld':
    data = data_sdld

for roi in SD_mats.keys():
    SD_mats[roi] = list()

for i in range(0, 18):
    for roi in kkROI:
        SD_mats[roi].append(np.array([data[i]['mlk'][roi],
                          data[i]['frt'][roi],
                          data[i]['odr'][roi]]).mean(axis=0))

SD_mean = dict.fromkeys(kkROI)

vmax = dict()
vmax['semcat'] = 0.6
vmax['sdld'] = 0.9

for roi in kkROI:
    SD_mean[roi] = np.array(SD_mats[roi]).mean(axis=0)

for roi in kkROI:    
    fig, ax = plt.subplots(1)
    im = ax.matshow(SD_mean[roi], vmin=0.5, vmax=SD_mean[roi].max(), cmap='viridis', origin='lower',
                    extent=times[[0, -1, 0, -1]])
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(f'{value} - Generalization across time in {roi}')
    plt.colorbar(im, ax=ax)
    plt.show()


for i in range(0, 18):    
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10,3))
    for roi,ax in zip(['lATL', 'rATL', 'PTC'], axes.flat):
        im = ax.matshow(np.array(SD_mats[roi])[i], vmin=0.5, vmax=vmax[value], cmap='viridis', origin='lower',
                    extent=times[[0, -1, 0, -1]])
        ax.axhline(0., color='k')
        ax.axvline(0., color='k')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title(f'{roi}')
        
    axes[0].set_xlabel('Testing Time (s)')
    axes[0].set_ylabel('Training Time (s)')
    fig.suptitle(f'{value} - Participant {i}', fontsize=16)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()
