#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:37:35 2022

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


confmat = []
for sub in range(0, 18):
   with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/abs_balanced_confusion_matrix_{sub}.P",
              'rb') as f:
        confmat.append(pickle.load(f))

new_mat = dict.fromkeys(['sd', 'LD'])

for task in new_mat.keys():
    new_mat[task] = dict.fromkeys(kkROI)
    for roi in new_mat[task].keys():
        new_mat[task][roi] = []
        for i in range(0, 18):
            new_mat[task][roi].append(confmat[i][task][roi])
        new_mat[task][roi] = np.array(new_mat[task][roi]).mean(axis=0)
              
times = np.arange(-300,900,4)

clusters = dict.fromkeys(kkROI)
clusters['lATL'] = [((int(np.where(times==264)[0])), \
                      (int(np.where(times==500)[0])))] 
clusters['rATL'] = [((int(np.where(times==272)[0])), \
                          (int(np.where(times==372)[0])))]
clusters['PTC'] = [((int(np.where(times==236)[0])), \
                      (int(np.where(times==452)[0])))] 
clusters['IFG'] = [((int(np.where(times==272)[0])), \
                      (int(np.where(times==356)[0])))]

    
for roi in ['lATL', 'rATL', 'PTC', 'IFG']:
    i = 0
    for cluster in clusters[roi]:
        i += 1
        get_clust = np.stack(new_mat['sd'][roi][cluster[0]:cluster[1]]).mean(axis=0) # normalize over true values
            # for the true values check the proportions of each category

        ax=sns.heatmap(get_clust,
                    annot=True,
                    xticklabels=['visual',
                            'hand',
                            'hear',
                            'abstract'],
                    yticklabels=['visual',
                            'hand',
                            'hear',
                            'abstract'],
                    cmap="viridis",)
        ax.set(xlabel="Predicted", ylabel="True")
        ax.set_title(f"{roi}")
        plt.tight_layout()
        plt.savefig(f"//imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/Figures/sd_{roi}_confusionmatrix_balanced.png", format="png");
        plt.show()


clusters['lATL'] = [((int(np.where(times==288)[0])), \
                          (int(np.where(times==328)[0])))] 
for cluster in clusters['lATL']:
    i += 1
    get_clust = np.stack(new_mat['LD']['lATL'][cluster[0]:cluster[1]]).mean(axis=0) # normalize over true values
        # for the true values check the proportions of each category

    ax=sns.heatmap(get_clust,
                annot=True,
                xticklabels=['visual',
                        'hand',
                        'hear',
                        'abstract'],
                yticklabels=['visual',
                        'hand',
                        'hear',
                        'abstract'],
                cmap="viridis",)
    ax.set(xlabel="Predicted", ylabel="True")
    ax.set_title(f"LD lATL")
    plt.tight_layout()
    plt.savefig(f"//imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/Figures/ld_lATL_confusionmatrix_balanced.png", format="png");
    plt.show()