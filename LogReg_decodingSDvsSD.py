# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:37:27 2021

@author: fm02
"""


#!/usr/bin/env python
# coding: utf-8


# Import some relevant packages.


import numpy as np
import pandas as pd
import pickle
import random


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef)

def rms(example):
    """Compute root mean square of each ROI.
    Input is a dataframe of length=n_vertices."""
    # first transform Series in np array of dimension n_vertics*timepoints
    example = np.vstack(np.array(example))
    # create np.array where to store info
    rms_example = np.zeros(example.shape[1])
    # loop over timepoints
    for i in np.arange(0,example.shape[1]):
        rms_example[i] = np.sqrt(np.mean(example[:,i]**2))
    
    return rms_example 

def trials_no_category(row):
    """Change number of trials when ignoring category.
    Adding 100 for each category so that each hundreds correspond to a category."""
    if row['category'] == 'visual':
        pass
    elif row['category'] == 'hand':
        row['trial'] = row['trial'] + 100
    elif row['category'] == 'hear':
        row['trial'] = row['trial'] + 200
    elif row['category'] == 'neutral':
        row['trial'] = row['trial'] + 300
    elif row['category'] =='emotional':
        row['trial'] = row['trial'] + 400
    
    return row
        
scores = {}
scores['mlkfrt'] = []
scores['frtodr'] = []
scores['odrmlk'] = []

patterns = {}
patterns['mlkfrt'] = []
patterns['frtodr'] = []
patterns['odrmlk'] = []

for sub in np.arange(0  ,18):
    print(f'Analysing participant number: {sub}')
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    with open(f'//cbsu/data/Imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
        output = pickle.load(f)
    
    kk = list(output.keys())
    
    # As we can observe, the words belong to different semantic categories (kk2).
    # In this project we will ignore it, and consider them just as different trials
    # belonging either to the LD or milk task. 
    
    kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
    kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']
    
    
    # The data we are working on, are not in a format useful for decoding, 
    # so we will reshape them.
    # 
    # In the starting dataset, information about each category was grouped together (see 'kk'),
    # while we want to group together all the information about a certain trial, at each timepoint.
    # We create dataframe so that we get information about trials, for each task and ROI.
    
    trials_ld = pd.DataFrame(columns=['ROI','category','trial','data'])
    trials_mlk = pd.DataFrame(columns=['ROI','category','trial','data'])
    trials_frt = pd.DataFrame(columns=['ROI','category','trial','data'])
    trials_odr = pd.DataFrame(columns=['ROI','category','trial','data'])

    
    # comments just on the first section, as it's doing the same job for each
    # task, category, and ROI
    for j,k in enumerate(kk):
        # check the key identity about the task
        if k[0:2] == 'LD':
            # check which category is this
            # (this checks if each category in kk2,
            # is present in k, the output[key] currently considered)
            mask_k = [k2 in k for k2 in kk2]
            # and save the category as a np string
            k2 = np.array(kk2)[mask_k][0]
            # check which ROI this is referring to
            mask_ROI = [k_ROI in k for k_ROI in kkROI]
            kROI = np.array(kkROI)[mask_ROI][0]
            # loop over trials
            for i in range(len(output[k])):
                    # save data (contained in output[k]) about that ROI
                    # for each trial (i) separately
                ls = [kROI, k2, i, output[k][i]]
                    # containing info about semantic_category, trial, and data
                row = pd.Series(ls, index=trials_ld.columns)
                    # and save in the relevant Dataframe, this case 
                    # Task = lexical decision, ROI = lATL
                trials_ld = trials_ld.append(row, ignore_index=True) 
        
        elif k[0:4] == 'milk':
            mask_k = [k2 in k for k2 in kk2]
            k2 = np.array(kk2)[mask_k][0]

            mask_ROI = [k_ROI in k for k_ROI in kkROI]
            kROI = np.array(kkROI)[mask_ROI][0]

            for i in range(len(output[k])):
                ls = [kROI, k2, i, output[k][i]]
                row = pd.Series(ls, index=trials_mlk.columns)
                trials_mlk = trials_mlk.append(row, ignore_index=True) 
            
        elif k[0:5] == 'fruit':
            mask_k = [k2 in k for k2 in kk2]
            k2 = np.array(kk2)[mask_k][0]

            mask_ROI = [k_ROI in k for k_ROI in kkROI]
            kROI = np.array(kkROI)[mask_ROI][0]

            for i in range(len(output[k])):
                ls = [kROI, k2, i, output[k][i]]
                row = pd.Series(ls, index=trials_frt.columns)
                trials_frt = trials_frt.append(row, ignore_index=True) 
        elif k[0:5] == 'odour':
            mask_k = [k2 in k for k2 in kk2]
            k2 = np.array(kk2)[mask_k][0]

            mask_ROI = [k_ROI in k for k_ROI in kkROI]
            kROI = np.array(kkROI)[mask_ROI][0]

            for i in range(len(output[k])):
                ls = [kROI, k2, i, output[k][i]]
                row = pd.Series(ls, index=trials_odr.columns)
                trials_odr = trials_odr.append(row, ignore_index=True) 
    # We now ignore the information about the categories and consider them just as different trials


    
    trials = {}

    trials['mlk'] = trials_mlk
    trials['frt'] = trials_frt
    trials['odr'] = trials_odr
    
    for tsk in trials.keys():
        trials[tsk] = trials[tsk].apply(trials_no_category,axis=1)

    trials_new = {}

    trials_new['mlk'] = []
    trials_new['frt'] = []
    trials_new['odr'] = []
    

    for tsk in trials_new.keys():
        for i in trials[tsk]['trial'].unique():
            trials_new[tsk].append(np.vstack(np.array(trials[tsk][trials[tsk]['trial']==i]['data'])))
        trials_new[tsk] = np.array(trials_new[tsk])
        
    trials_avg3 = dict.fromkeys(trials_new.keys())
    
    for k in trials_new.keys():
        
        while len(trials_new[k])%3 != 0:
            trials_new[k] = np.delete(trials_new[k], len(trials_new[k])-1, 0)
    # create random groups of trials
        new_tsk = np.split(trials_new[k],len(trials_new[k])/3)
        new_trials = []
    # calculate average for each timepoint of the 3 trials
        for nt in new_tsk:
            new_trials.append(np.mean(np.array(nt),0))
        # assign group it in the corresponding task
        
        trials_avg3[k] = np.array(new_trials)
    
    vertices = []
    
    for roi in trials_mlk[trials['mlk']['trial']==0]['data']:
        vertices.append(roi.shape[0])
    
    print([v for v in vertices])
    
    ROI_vertices = []
    
    for i in range(len(vertices)):
        ROI_vertices.extend([kkROI[i]]*vertices[i])
       
    
    # We create the X and y matrices that will be used for creating the model, by appendign milk and LD trials.
    # We also shuffle them.
    
    # prepare a series of classifier applied at each time sample
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearModel(LogisticRegression(C=1,
                                                       solver='liblinear'))) # asking LDA to store covariance
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    
    # just use subt instead of trials_semK if you want to have average of trials
    
   
    for task in scores.keys():
        X = np.concatenate([trials_avg3[task[0:3]],
                            trials_avg3[task[3:]]])
        
        y = np.array([task[0:3]]*len(trials_avg3[task[0:3]]) + \
                         [task[3:]]*len(trials_avg3[task[3:]]))
        
        X, y = shuffle(X, y, random_state=0)
        
        scores[task].append(cross_val_multiscore(time_decod,
                                                 X, y, cv=5).mean(axis=0))
        
        time_decod.fit(X, y)
        pattern = get_coef(time_decod, 'patterns_', inverse_transform=True)
        pattern = pd.DataFrame(pattern, index=ROI_vertices)
        patterns[task].append(pattern)
        
    # contrasting each semantic decision task vs lexical decision task
    # check when and where areas are sensitive to task difference on average
    

df_to_export = pd.DataFrame(patterns)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0111_LogReg_SDvsSD_patterns.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)
    
df_to_export = pd.DataFrame(scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0111_LogReg_SDvsSD_scores.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)
