#!/usr/bin/env python
# coding: utf-8

### Author: federica.magnabosco@mrc-cbu.cam.ac.uk
### Fit decoding model LDvsSD individual ROIs and save accuracy

# Import some relevant packages.

import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef)

def divide_ROI(trials):
    """Organise data so that there's info about each ROI for each trial.
    Ignore info about semantic category."""
    dic = dict.fromkeys(kk2)
    for i in dic.keys():
        dic[i] = []
    for category in kk2:
        for trial in trials['trial'][trials['category']==category].unique():
            dic[category].append(np.concatenate(np.array(trials['data'] \
                                                    [(trials['trial']==trial) & \
                                                     (trials['category']==category \
                                                      )])))
    all_trials = []
    for category in kk2:
        all_trials.append(np.stack(dic[category]))
    all_trials = np.concatenate(all_trials)
        # return an dict where keys are ROI
        # for each roi an arraycontaining the data
        # shape n_trial*n_vertices*n_timepoints
    return all_trials

kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

# initialise dictionaries and lists for storing scores
scores = []

patterns = []

# loop over participants
for sub in np.arange(0, 18):
    print(f"Analysing subject {sub}")
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    with open(f'//cbsu/data/imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
        output = pickle.load(f)
    
    kk = list(output.keys())
    
    # words belong to different semantic categories (kk2).
    # In this script we will ignore this,
    # and consider them just as different trials
    # belonging either to the LD or milk task. 
    
# First we will reorganise the data in pandas dataframe, instead of dict.
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

    # loop over all the 120 keys in output
    for j,k in enumerate(kk):
        # check which task
         # check the key identity about the task
        # if k[0:2] == 'LD':
        #     # check which semantic category
        #     # (this checks if each category in kk2,
        #     # is present in k, the output[key] currently considered)
        #     mask_k = [k2 in k for k2 in kk2]
        #     # and save the category as a string
        #     k2 = np.array(kk2)[mask_k][0]
        #     # check which ROI
        #     mask_ROI = [k_ROI in k for k_ROI in kkROI]
        #     kROI = np.array(kkROI)[mask_ROI][0]
        #     # loop over trials
        #     # this extracts data from the relevant key
        #     # and save data as a pandas dataframe (easier to access info)
        #     for i in range(len(output[k])):
        #         # save data (contained in output[k])
        #         # for each trial (i) separately
        #         ls = [kROI, k2, i, output[k][i]]
        #         # containing info about semantic_category, trial, and data
        #         row = pd.Series(ls, index=trials_ld.columns)
        #         # and append data to relevant dataframe
        #         trials_ld = trials_ld.append(row, ignore_index=True) 
        
        if k[0:4] == 'milk':
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
    
    # organise in a dict
    trials = {}
    trials['mlk'] = trials_mlk
    trials['frt'] = trials_frt
    trials['odr'] = trials_odr
    
    # dict for each task
    for tsk in trials.keys():
        trials[tsk] = divide_ROI(trials[tsk])
        
    ### now let's average 3 trials together
    # initialise dict
    trials_avg3 = dict.fromkeys(trials.keys())

    # loop over tasks   
    for k in trials.keys():
        # drop trials until we reach a multiple of 3
        # (this is so that we always average 3 trials together)
        while len(trials[k])%3 != 0:
            trials[k] = np.delete(trials[k], len(trials[k])-1, 0)
        # split data in groups of 3 trials
        new_tsk = np.vsplit(trials[k], len(trials[k])/3)
        new_trials = []
        # calculate average for each timepoint (axis=0) of the 3 trials
        for nt in new_tsk:
            new_trials.append(np.mean(np.array(nt),0))
        # assign group to the corresponding task in the dict
        # each is 3D array n_trial*n_vertices*n_timepoints
        trials_avg3[k] = np.array(new_trials)

    
    # retrive information about the vertices
    vertices = []

    # using mlk trial=0, vertices order doesn't change, 
    # so it doesn't matter which task and trial looking up    
    for roi in trials_mlk['data'][(trials_mlk['trial']==0) & \
                          (trials_mlk['category']=='visual')]:
        vertices.append(roi.shape[0])
    
    # print information about vertices
    print([v for v in vertices])
    
    ROI_vertices = []

    # create list with length=n_vertices containing ROI string for each vertex    
    for i in range(len(vertices)):
        ROI_vertices.extend([kkROI[i]]*vertices[i]) 

    # We create and run the model.
    # using example from MNE example https://mne.tools/stable/auto_examples/decoding/decoding_spatio_temporal_source.html

    # We expect the model to perform at chance before the presentation of the stimuli
    # (no ROI should be sensitive to task/semantics demands before the presentation of a word).
    
    # prepare a series of classifier applied at each time sample
    # this is the classifier
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearModel(LogisticRegression(C=1,
                                                       solver='lbfgs',
                                                       max_iter=1000)))
    # Search Light
    # "Fit, predict and score a series of models to each subset of the dataset along the last dimension"
    time_decod = SlidingEstimator(clf, scoring='roc_auc_ovr')
            
    # X input matrix, containing LD and task trials, it has dimension n_trial*n_vertices*n_timepoints
    X = np.concatenate([trials_avg3['mlk'],
                        trials_avg3['frt'],
                        trials_avg3['odr']])

    # Y category array. it has dimension n_trial
    y = np.array(['mlk']*len(trials_avg3['mlk']) + \
                 ['frt']*len(trials_avg3['frt']) + \
                 ['odr']*len(trials_avg3['odr']))

    # shuffle them, so random order     
    X, y = shuffle(X, y,
                   # random_state=0
                   )        
    # append the average of 5-fold cross validation to the scores dict for this task
    scores.append(cross_val_multiscore(time_decod,
                                             X, y, cv=5).mean(axis=0))

    time_decod.fit(X, y)

    # this already applies Haufe's trick
    # Retrieve patterns after inversing the z-score normalization step
    pattern = get_coef(time_decod, 'patterns_', inverse_transform=True)
    patterns.append(dict.fromkeys(['mlk', 'frt', 'odr']))
        # append ROI information
    for i, task in enumerate(['mlk', 'frt', 'odr']):
        patterns[-1][task] = pd.DataFrame(pattern[:,i,:], index=ROI_vertices)

# save the scores ...
df_to_export = pd.DataFrame(patterns)
with open("//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SDvsSD/patterns.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)
    
# save the scores ...
df_to_export = pd.DataFrame(scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SDvsSD/scores.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)

