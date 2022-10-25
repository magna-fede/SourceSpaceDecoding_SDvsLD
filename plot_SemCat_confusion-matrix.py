#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:01:52 2022

@author: fm02
"""

# plot results of concat_indivisual_SemCat_confusion-matrix_slurm

import numpy as np
import pickle

import seaborn as sns
import matplotlib.pyplot as plt


kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']
kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']

confmat = []
for sub in range(0, 18):
   with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/confusion_matrix_{sub}.P",
              'rb') as f:
        confmat.append(pickle.load(f))

new_mat = dict.fromkeys(['sd', 'ld'])

for task in new_mat.keys():
    new_mat[task] = dict.fromkeys(kkROI)
    for roi in new_mat[task].keys():
        new_mat[task][roi] = []
        for i in range(0, 18):
            new_mat[task][roi].append(confmat[i][task][roi])
        new_mat[task][roi] = np.array(new_mat[task][roi]).mean(axis=0)
        
        
times = np.arange(-300,900,4)



clusters = dict.fromkeys(kkROI)
clusters['lATL'] = [((int(np.where(times==236)[0])), \
                      (int(np.where(times==504)[0]))), \
                    ((int(np.where(times==588)[0])), \
                      (int(np.where(times==652)[0])))]
clusters['rATL'] = [((int(np.where(times==264)[0])), \
                          (int(np.where(times==488)[0])))]
clusters['AG'] = [((int(np.where(times==312)[0])), \
                          (int(np.where(times==372)[0])))]
clusters['PTC'] = [((int(np.where(times==208)[0])), \
                      (int(np.where(times==460)[0]))), \
                    ((int(np.where(times==468)[0])), \
                      (int(np.where(times==556)[0])))]
clusters['IFG'] = [((int(np.where(times==260)[0])), \
                      (int(np.where(times==496)[0])))]
clusters['PVA'] = [((int(np.where(times==64)[0])), \
                      (int(np.where(times==128)[0]))), \
                    ((int(np.where(times==156)[0])), \
                      (int(np.where(times==240)[0]))), \
                    ((int(np.where(times==252)[0])), \
                      (int(np.where(times==344)[0])))]
    
for roi in kkROI:
    i = 0
    for cluster in clusters[roi]:
        i += 1
        get_clust = np.stack(new_mat['sd'][roi][cluster[0]:cluster[1]]).mean(axis=0) # normalize over true values
            # for the true values check the proportions of each category

        ax=sns.heatmap(get_clust,
                    annot=True,
                    xticklabels=kk2,
                    yticklabels=kk2,
                    cmap="viridis",)
        ax.set(xlabel="Predicted", ylabel="True")
        ax.set_title(f"{roi}, cluster {str(i)}")
        plt.show()
    
 

