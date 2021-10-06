# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:48:16 2021

@author: fm02
"""

### This script takes as input the data Setareh sent me
### which consists in source space activity for 18 subjects during
### 1 Lexical Decision Task and 3 Semantic Decision Tasks (milk, fruit, odour).
### vertices belong to six different ROIs.
### In this script calculates the average decoding performance across subjects
### of each ROI independently in detecting SD vs SD
### So, for each ROI (for each subject) a classifier is trained to distinguish
### each semantic task from the lexical task.
### Word semantic identities are not taken into account. 

# import relevant stuff

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef)

# initialise lists where we'll store output

list_avg_scores = []
list_mlk_scores = []
list_frt_scores = []
list_odr_scores = []

for sub in np.arange(0  ,18):
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
    
    trials_mlk = pd.DataFrame(columns=['ROI','category','trial','data'])
    trials_frt = pd.DataFrame(columns=['ROI','category','trial','data'])
    trials_odr = pd.DataFrame(columns=['ROI','category','trial','data'])

    
    # comments just on the first section, as it's doing the same job for each
    # task, category, and ROI
    for j,k in enumerate(kk):
        # check the key identity about the task
        # if k[0:2] == 'LD':
        #     # check which category is this
        #     # (this checks if each category in kk2,
        #     # is present in k, the output[key] currently considered)
        #     mask_k = [k2 in k for k2 in kk2]
        #     # and save the category as a np string
        #     k2 = np.array(kk2)[mask_k][0]
        #     # check which ROI this is referring to
        #     mask_ROI = [k_ROI in k for k_ROI in kkROI]
        #     kROI = np.array(kkROI)[mask_ROI][0]
        #     # loop over trials
        #     for i in range(len(output[k])):
        #             # save data (contained in output[k]) about that ROI
        #             # for each trial (i) separately
        #         ls = [kROI, k2, i, output[k][i]]
        #             # containing info about semantic_category, trial, and data
        #         row = pd.Series(ls, index=trials_ld.columns)
        #             # and save in the relevant Dataframe, this case 
        #             # Task = lexical decision, ROI = lATL
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
    
    # We now ignore the information about the categories and consider them just as different trials
    
    # as we will consider each ROI separately, create a dataframe for each task
    mlks = pd.DataFrame(columns=kkROI)
    frts = pd.DataFrame(columns=kkROI)
    odrs = pd.DataFrame(columns=kkROI)
    # lds = pd.DataFrame(columns=kkROI)
    
    # in this script, the above passage is redundant (as we don't need to merge
    # data from the same trial for each ROI - but it's convenient in other
    # scripts, so keeping it.
    # get data for each task for each ROI
    for ROI in kkROI:
        mlks[ROI] = trials_mlk['data'][trials_mlk['ROI']==ROI].reset_index(drop=True)
        frts[ROI] = trials_frt['data'][trials_frt['ROI']==ROI].reset_index(drop=True)
        odrs[ROI] = trials_odr['data'][trials_odr['ROI']==ROI].reset_index(drop=True)
        # lds[ROI] = trials_ld['data'][trials_ld['ROI']==ROI].reset_index(drop=True)
        
        
    # prepare a series of classifier applied at each time sample
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # keep all vertices
                        LinearModel(LogisticRegression(C=1, solver='liblinear'))) # always binary classification
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    
    # initialise dict for saving scores in participant's list
    scores_mlkfrt = dict.fromkeys(kkROI)
    scores_frtodr = dict.fromkeys(kkROI)
    scores_odrmlk = dict.fromkeys(kkROI)

    # for each roi, create and apply the classifier
    for roi in kkROI:
        # create X matrix for each SD vs LD
        X_mlkfrt = np.concatenate([np.stack(mlks[roi]),np.stack(frts[roi])])
        y_mlkfrt = np.array(['milk']*np.stack(mlks[roi]).shape[0] + 
                            ['fruit']*np.stack(frts[roi]).shape[0])
        
        X_frtodr = np.concatenate([np.stack(frts[roi]),np.stack(odrs[roi])])
        y_frtodr = np.array(['fruit']*np.stack(frts[roi]).shape[0] + 
                            ['odour']*np.stack(odrs[roi]).shape[0])
        
        X_odrmlk = np.concatenate([np.stack(odrs[roi]),np.stack(mlks[roi])])
        y_odrmlk = np.array(['odour']*np.stack(odrs[roi]).shape[0] + 
                            ['milk']*np.stack(mlks[roi]).shape[0])
        
        # randomise order (or otherwirse SD always before LD)
        # not sure if this is necessary, but it's proably worth to be sure
        X_mlkfrt, y_mlkfrt = shuffle(X_mlkfrt, y_mlkfrt, random_state=0)
        X_frtodr, y_frtodr = shuffle(X_frtodr, y_frtodr, random_state=0)
        X_odrmlk, y_odrmlk = shuffle(X_odrmlk, y_odrmlk, random_state=0)
        

        # Run cross-validated decoding analyses:
        # 5 cross-validations
        scores_mlkfrt[roi] = cross_val_multiscore(time_decod, X_mlkfrt,
                                                  y_mlkfrt, cv=5)
        scores_frtodr[roi] = cross_val_multiscore(time_decod, X_frtodr,
                                                  y_frtodr, cv=5)
        scores_odrmlk[roi] = cross_val_multiscore(time_decod, X_odrmlk,
                                                  y_odrmlk, cv=5)

    
    # avg_score will store average performance across the 3 classifiers for each participant
    avg_score = dict.fromkeys(kkROI)
    
    # calculate also the average the performance of each of the 3 models
    for roi in kkROI:
        # first calculcate mean performance for each CV and then for each task
        avg_score[roi] = np.mean([scores_mlkfrt[roi].mean(0),
                                  scores_frtodr[roi].mean(0),
                                  scores_odrmlk[roi].mean(0)],0)

    # append performance to list
    list_avg_scores.append(avg_score)
    list_mlk_scores.append(scores_mlkfrt)
    list_frt_scores.append(scores_frtodr)
    list_odr_scores.append(scores_odrmlk)
    
df_to_export = pd.DataFrame(list_avg_scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_ROIs_avg_scores.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)
df_to_export = pd.DataFrame(list_mlk_scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_ROIs_mlkfrt_scores.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)
df_to_export = pd.DataFrame(list_frt_scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_ROIs_frtodr_scores.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)
df_to_export = pd.DataFrame(list_odr_scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_ROIs_odrmlk_scores.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)
