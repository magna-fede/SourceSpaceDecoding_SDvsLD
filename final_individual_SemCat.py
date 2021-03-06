#!/usr/bin/env python
# coding: utf-8

# # Data for science Residency Project
# 
# In this project I will apply some of the notions lernt during the course to try to predict which brain regions and at which time point are sensitive to the different amount of semantic resources necessary for completing two different tasks. To do this, we will look at the source estimated activity of 6 Regions of Interest (ROIs) for one participant. The two tasks (lexical decision and semantic decision) are belived to vary in the amount of semantic resources necessary for completing the task. The activity is related to -300 ms to 900 ms post stimulus presentation.
# We will try to predict to which task each trial belongs to and, after that, we will try to understand which ROI carries is sensitive to different semantics demands, by looking at the average and the maximum coefficient in each ROI at each time point.

# Import some relevant packages.
# mne is a package used in the analysis of MEG and EEG brain data. We are importing some functions useful for decoding brain signal.
# 

import numpy as np
import pandas as pd
import pickle


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def divide_semK_ROI(trials):
    dic = {}
    for semK in kk2:
        dic[semK] = dict.fromkeys(kkROI)
        for i in dic[semK].keys():
            dic[semK][i] = []
        for trial in trials['trial'][trials['category']==semK].unique():
            for roi in kkROI:
                dic[semK][roi].append(np.concatenate \
                              (trials['data'] \
                               [(trials['category']==semK) \
                                & (trials['trial']==trial) \
                                & (trials['ROI']==roi)].values) )
        for roi in kkROI:
            dic[semK][roi] = np.array(dic[semK][roi])
        
    return dic

kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

# initialise dictionaries and lists for storing scores
scores = {}
scores['ld'] = dict.fromkeys(kkROI)
scores['mlk'] = dict.fromkeys(kkROI)
scores['frt'] = dict.fromkeys(kkROI)
scores['odr'] = dict.fromkeys(kkROI)

for task in scores.keys():
    for roi in scores[task].keys():
        scores[task][roi] = []
        
for sub in np.arange(0, 18):
    print(f"Analysing subject {sub}")
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    with open(f'//cbsu/data/imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
        output = pickle.load(f)

    # with open(f'/imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
    #     output = pickle.load(f)    
    
    
    # with open(f'C:/Users/User/OwnCloud/DSR/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
    #     output = pickle.load(f)    
    
    kk = list(output.keys())
    
    # As we can observe, the words belong to different semantic categories (kk2).
    # In this project we will ignore it, and consider them just as different trials
    # belonging either to the LD or milk task.
    
    
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
    trials['ld'] = trials_ld
    trials['mlk'] = trials_mlk
    trials['frt'] = trials_frt
    trials['odr'] = trials_odr
    
    trials_semK = {}
    
    for tsk in list(trials.keys()):
        trials_semK[tsk] = divide_semK_ROI(trials[tsk])
        
    # try not averaging because not enough trials otherwise
    
    ### now let's average 3 trials together
    # initialise dict
    trials_avg3 = dict.fromkeys(trials_semK.keys())
    
    for task in trials_avg3.keys():
        trials_avg3[task] = dict.fromkeys(kk2)
        for semK in trials_avg3[task].keys():
            trials_avg3[task][semK] = dict.fromkeys(kkROI)
            for roi in trials_avg3[task][semK].keys():
                trials_avg3[task][semK][roi] = []
            
    # loop over tasks   
    for task in trials_semK.keys():
        for semK in trials_semK[task].keys():
        # drop trials until we reach a multiple of 3
        # (this is so that we always average 3 trials together)
            for roi in trials_semK[task][semK].keys():
                while len(trials_semK[task][semK][roi])%3 != 0:
                    trials_semK[task][semK][roi] = np.delete(trials_semK[task][semK][roi], len(trials_semK[task][semK][roi])-1, 0)
                # split data in groups of 3 trials
                new_tsk = np.vsplit(trials_semK[task][semK][roi], len(trials_semK[task][semK][roi])/3)
                new_trials = []
                # calculate average for each timepoint (axis=0) of the 3 trials
                for nt in new_tsk:
                    new_trials.append(np.mean(np.array(nt),0))
                # assign group to the corresponding task in the dict
                # each is 3D array n_trial*n_vertices*n_timepoints
                trials_avg3[task][semK][roi] = np.array(new_trials)
                   
    # We create and run the model. We expect the model to perform at chance before the presentation of the stimuli (no ROI should be sensitive to task/semantics demands before the presentation of a word).
    
    # prepare a series of classifier applied at each time sample
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearModel(LogisticRegression(C=1,
                                                       solver='lbfgs',
                                                       max_iter=1000))) 
    time_decod = SlidingEstimator(clf, scoring='roc_auc_ovr')
    
 
    # just use subt instead of trials_semK if you want to have average of trials
    for task in trials_avg3.keys():
        for roi in kkROI:
            X = []
            y = []
            for semK in kk2:
                X.append(trials_avg3[task][semK][roi])
                y.extend([semK]*len(trials_avg3[task][semK][roi]))
            X = np.concatenate(X)
            y = np.array(y)

            X, y = shuffle(X, y, random_state=0)
            
            scores[task][roi].append(cross_val_multiscore(time_decod,
                                                     X, y, cv=5).mean(axis=0))
        
            
# save the scores ...
df_to_export = pd.DataFrame(scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/scores.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)
    
    

