
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
import random

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# initialise lists where we'll store output

scores = []

for sub in np.arange(0  ,18):
    print(sub)
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
    
    mlks_avg = pd.DataFrame(columns=kkROI)
    frts_avg = pd.DataFrame(columns=kkROI)
    odrs_avg = pd.DataFrame(columns=kkROI)
    
    # now let's average 4 trials together
    for ROI in kkROI:
        for i,tsk in enumerate([frts[ROI],mlks[ROI],odrs[ROI]]):
        # make sure the number of trials is a multiple of 4, or eliminate excess
            tsk = np.stack(np.array(tsk))
            while len(tsk)%4 != 0:
                tsk = np.delete(tsk,-1,axis=0)
            # create random groups of trials
            # note that np.random.shuffle operates on the first axis,
            # meaning that the order on axis 1,2 is untouched (I hope)
            np.random.shuffle(tsk)
            new_tsk = list(chunks(tsk, 4))
            new_trials = []
            # calculate average for each timepoint of the 4 trials
            for nt in new_tsk:
                new_trials.append(np.mean(nt,0))
            # assign group it in the corresponding task

            if i==0:
                frts_avg[ROI] = new_trials
            elif i==1:
                mlks_avg[ROI] = new_trials
            elif i==2:
                odrs_avg[ROI] = new_trials
        
            
    # prepare a series of classifier applied at each time sample
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # keep all vertices
                        LinearModel(LogisticRegression(C=1,  solver='lbfgs',max_iter=1000))) # always binary classification
    time_decod = SlidingEstimator(clf)
    
    # initialise dict for saving scores in participant's list
    score = dict.fromkeys(kkROI)


    # for each roi, create and apply the classifier
    for roi in kkROI:
        # create X matrix for each SD vs LD
        X = np.concatenate([np.stack(mlks_avg[roi]),
                            np.stack(frts_avg[roi]),
                            np.stack(odrs_avg[roi])])

        y = np.array(['milk']*np.stack(mlks_avg[roi]).shape[0] +
                     ['fruit']*np.stack(frts_avg[roi]).shape[0] +
                     ['odour']*np.stack(odrs_avg[roi]).shape[0])
        
        # randomise order (or otherwirse SD always before LD)
        # not sure if this is necessary, but it's proably worth to be sure
        X, y = shuffle(X, y, random_state=0)
        

        # Run cross-validated decoding analyses:
        # 5 cross-validations
        score[roi] = cross_val_multiscore(time_decod, X,
                                                  y, cv=5)


    # append performance to list
    scores.append(score)

    
# df_to_export = pd.DataFrame(list_avg_scores)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1015_SDvsSD_ROIs_avg_scores.P", 'wb') as outfile:
#     pickle.dump(df_to_export,outfile)
# df_to_export = pd.DataFrame(list_mlk_scores)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1015_SDvsSD_ROIs_mlkfrt_scores.P", 'wb') as outfile:
#     pickle.dump(df_to_export,outfile)
# df_to_export = pd.DataFrame(list_frt_scores)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1015_SDvsSD_ROIs_frtodr_scores.P", 'wb') as outfile:
#     pickle.dump(df_to_export,outfile)
# df_to_export = pd.DataFrame(list_odr_scores)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1015_SDvsSD_ROIs_odrmlk_scores.P", 'wb') as outfile:
#     pickle.dump(df_to_export,outfile)

df_to_export = pd.DataFrame(scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/eachROIseprately/whichSD/0426_SDvsSD_ROIseparately_scores.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/eachROIseprately/whichSD/0426_SDvsSD_ROIseparately_scores.P", 'rb') as f:
      df_scores = pickle.load(f)

import seaborn as sns
import matplotlib.pyplot as plt

times = np.arange(-300,900,4)
score = dict.fromkeys(kkROI)

for roi in kkROI:
    score[roi] = []
    for sub in range(18):
        score[roi].append(df_scores[roi][sub].mean(axis=0))

for roi in kkROI:
    sns.lineplot(x=times, y=np.array(score[roi]).mean(axis=0))
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('SD - individual ROI accuracy')
plt.axhline(1/3, color='k', linestyle='--', label='chance');
plt.legend(kkROI);
plt.show();
