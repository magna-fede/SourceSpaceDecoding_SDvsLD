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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

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
        
SDvsSD_scores = []
SDvsSD_coefficients = []
cat_scores = []

for sub in np.arange(0, 18):
    print(f'Analysing participant number: {sub}')
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    with open(f'//cbsu/data/Imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
        output = pickle.load(f)

    # with open(f'//imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
    #     output = pickle.load(f)
    
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

    trials_ld_ignore = trials_ld.apply(trials_no_category,axis=1)
    trials_mlk_ignore = trials_mlk.apply(trials_no_category,axis=1)
    trials_frt_ignore = trials_frt.apply(trials_no_category,axis=1)
    trials_odr_ignore = trials_odr.apply(trials_no_category,axis=1)
    
    mlks = []
    for i in trials_mlk_ignore['trial'].unique():
        mlks.append(np.concatenate(np.array(trials_mlk_ignore[trials_mlk_ignore['trial']==i]['data'])))
    
    lds = []
    for i in trials_ld_ignore['trial'].unique():
        lds.append(np.concatenate(np.array(trials_ld_ignore[trials_ld_ignore['trial']==i]['data'])))
        
    frts = []
    for i in trials_frt_ignore['trial'].unique():
        frts.append(np.concatenate(np.array(trials_frt_ignore[trials_frt_ignore['trial']==i]['data'])))
    
    odrs = []
    for i in trials_odr_ignore['trial'].unique():
        odrs.append(np.concatenate(np.array(trials_odr_ignore[trials_odr_ignore['trial']==i]['data'])))
    
    # now let's average 4 trials together
    for i,tsk in enumerate([lds,frts,mlks,odrs]):
        # make sure the number of trials is a multiple of 4, or eliminate excess
        while len(tsk)%4 != 0:
            tsk.pop()
        # create random groups of trials
        random.shuffle(tsk)
        new_tsk = list(chunks(tsk, 4))
        new_trials = []
        # calculate average for each timepoint of the 4 trials
        for nt in new_tsk:
            new_trials.append(np.mean(nt,0))
        # assign group it in the corresponding task
        if i==0:
            sub_lds = new_trials
        elif i==1:
            sub_frts = new_trials
        elif i==2:
            sub_mlks = new_trials
        elif i==3:
            sub_odrs = new_trials
    

    # Now let's convert back to np.array and see how many trials we have. 
 
    mlks = np.array(sub_mlks)
    lds = np.array(sub_lds)
    frts = np.array(sub_frts)
    odrs = np.array(sub_odrs)
    
    print(mlks.shape) 
    print(lds.shape) 
    print(frts.shape)
    print(odrs.shape)
    
    vertices = []
    
    for trial in trials_mlk_ignore['data'][trials_mlk_ignore['trial']==0]:
        vertices.append(trial.shape[0])
    
    print([v for v in vertices])
    
    ROI_vertices = []
    
    for i in range(len(vertices)):
        ROI_vertices.extend([kkROI[i]]*vertices[i])
    
    
    
    # We create the X and y matrices that will be used for creating the model, by appendign milk and LD trials.
    # We also shuffle them.
    
    
    # contrasting each semantic decision task vs lexical decision task
    # check when and where areas are sensitive to task difference on average
    
    X_mlkfrt = np.concatenate([mlks , frts])
    y_mlkfrt = np.array(['milk']*len(mlks) + ['fruit']*len(frts))
    
    X_frtodr = np.concatenate([frts , odrs])
    y_frtodr = np.array(['fruit']*len(frts) + ['odour']*len(odrs))
    
    X_odrmlk = np.concatenate([odrs , mlks])
    y_odrmlk = np.array(['odour']*len(odrs) + ['milk']*len(mlks))
    
    
    X_mlkfrt, y_mlkfrt = shuffle(X_mlkfrt, y_mlkfrt, random_state=0)
    X_frtodr, y_frtodr = shuffle(X_frtodr, y_frtodr, random_state=0)
    X_odrmlk, y_odrmlk = shuffle(X_odrmlk, y_odrmlk, random_state=0)
    
    # We create and run the model. We expect the model to perform at chance before the presentation of the stimuli (no ROI should be sensitive to task/semantics demands before the presentation of a word).
    
    # prepare a series of classifier applied at each time sample
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # select features for speed
                        LinearModel(LinearDiscriminantAnalysis(solver="svd")))
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    
    # Run cross-validated decoding analyses:
    scores_mlkfrt = cross_val_multiscore(time_decod, X_mlkfrt, y_mlkfrt, cv=5)
    scores_frtodr = cross_val_multiscore(time_decod, X_frtodr, y_frtodr, cv=5)
    scores_odrmlk = cross_val_multiscore(time_decod, X_odrmlk, y_odrmlk, cv=5)
    
    scores = pd.DataFrame(list(zip(scores_mlkfrt, scores_frtodr, scores_odrmlk)),
                          columns=['milkVSfruit',
                                   'fruitVSodour', 
                                   'odourVSmilk'])
    SDvsSD_scores.append(scores)
    
    # HEY!
    # thanks mne.
    # https://github.com/mne-tools/mne-python/blob/maint/0.23/mne/decoding/base.py#L291-L355
    # line 93
    # patterns does already apply Haufe's trick
    
    time_decod.fit(X_mlkfrt, y_mlkfrt)
    patterns_mlkfrt = get_coef(time_decod, 'patterns_', inverse_transform=True)
    
    time_decod.fit(X_frtodr, y_frtodr)
    patterns_frtodr = get_coef(time_decod, 'patterns_', inverse_transform=True)
    
    time_decod.fit(X_odrmlk, y_odrmlk)
    patterns_odrmlk = get_coef(time_decod, 'patterns_', inverse_transform=True)
    
    # this df has 4 columns:
        # one codes to which ROI the vertex belongs to
        # the other three refers to each task.

    df = pd.DataFrame(zip(ROI_vertices, 
                          patterns_mlkfrt, 
                          patterns_frtodr, 
                          patterns_odrmlk),
                      columns=['ROI',
                               'milkVSfruit',
                               'fruitVSodour', 
                               'odourVSmilk'])

    
    avg = []
    for i in range(len(df)):
        avg.append(np.mean([df['milkVSfruit'][i],
                            df['fruitVSodour'][i],
                            df['odourVSmilk'][i]],0))
    df['avg'] = avg

    
    SDvsSD_coefficients.append(df)
    
df_to_export = pd.Series(SDvsSD_scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/LDA/whichSD/1123_SDvsSD_scores.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)
    
df_to_export = pd.Series(SDvsSD_coefficients)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/LDA/whichSD/1123_SDvsSD_coefficients.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)


