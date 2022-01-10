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
import random
from itertools import combinations

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

scores = {}
scores['mlk'] = []
scores['frt'] = []
scores['odr'] = []

patterns = {}
patterns['mlk'] = []
patterns['frt'] = []
patterns['odr'] = []

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
    trials['ld'] = trials_ld
    trials['mlk'] = trials_mlk
    trials['frt'] = trials_frt
    trials['odr'] = trials_odr
    
    for tsk in trials.keys():
        trials[tsk] = trials[tsk].apply(trials_no_category,axis=1)
    
    trials_new = {}
    
    trials_new['ld'] = []
    trials_new['mlk'] = []
    trials_new['frt'] = []
    trials_new['odr'] = []
    

    for tsk in trials_new.keys():
        for i in trials[tsk]['trial'].unique():
            trials_new[tsk].append(np.vstack(np.array(trials[tsk][trials[tsk]['trial']==i]['data'])))
        trials_new[tsk] = np.array(trials_new[tsk])
    
    # now let's average 3 trials together

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
    
    
    # We create and run the model. We expect the model to perform at chance before the presentation of the stimuli (no ROI should be sensitive to task/semantics demands before the presentation of a word).
    
    # prepare a series of classifier applied at each time sample
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearModel(LogisticRegression(C=1,
                                                       solver='liblinear'))) # asking LDA to store covariance
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    
    # just use subt instead of trials_semK if you want to have average of trials
    
   
    for task in scores.keys():
        X = np.concatenate([trials_avg3['ld'],trials_avg3[task]])
        
        y = np.array(['ld']*len(trials_avg3['ld']) + \
                         [task]*len(trials_avg3[task]))
        
        X, y = shuffle(X, y, random_state=0)
        
        scores[task].append(cross_val_multiscore(time_decod,
                                                 X, y, cv=5).mean(axis=0))
        
        time_decod.fit(X, y)
        pattern = get_coef(time_decod, 'patterns_', inverse_transform=True)
        pattern = pd.DataFrame(pattern, index=ROI_vertices)
        patterns[task].append(pattern)
        
# np.array(patterns['frt'][0].loc['lATL']).mean(axis=0)

df_to_export = pd.DataFrame(patterns)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1130_LogReg_LDvsSD_patterns.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)
    
df_to_export = pd.DataFrame(scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1130_LogReg_LDvsSD_scores.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1130_LogReg_LDvsSD_patterns.P", 'rb') as f:
      patterns = pickle.load(f)
      
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1130_LogReg_LDvsSD_scores.P", 'rb') as f:
      scores = pickle.load(f)



import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

times = np.arange(-300,900,4)

scores['avg'] = [ [] for _ in range(len(scores)) ]

for i in range(len(scores['frt'])):
    scores['avg'][i] = np.array([scores['mlk'][i],
                                 scores['frt'][i],
                                 scores['odr'][i]]).mean(axis=0)
    
sns.lineplot(x=times, y=np.array(scores['avg']).mean(axis=0))
plt.fill_between(x=times, \
                 y1=(np.mean(np.array(scores['avg']),0)-sem(np.vstack(scores['avg']),0)), \
                 y2=(np.mean(np.array(scores['avg']),0)+sem(np.vstack(scores['avg']),0)), \
                 color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('LD vs average(SD) Semantic Category Decoding')
plt.axhline(.5, color='k', linestyle='--', label='chance');
# plt.legend();
plt.show();

for task in (['frt', 'mlk', 'odr']):
    sns.lineplot(x=times, y=np.array(scores[task]).mean(axis=0))
    # plt.fill_between(x=times, \
    #                  y1=(np.mean(np.array(scores[task]),0) - \
    #                      sem(np.array(scores[task]),0)), \
    #                  y2=(np.mean(np.array(scores[task]),0) + \
    #                      sem(np.array(scores[task]),0)), \
    #                  color='b', alpha=.1)

plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('LD vs SD Semantic Category Decoding')
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.legend(['frt', 'mlk', 'odr']);
plt.show();

patterns_roi = dict.fromkeys(kkROI)

for roi in patterns_roi.keys():
    patterns_roi[roi] = dict.fromkeys(['frt', 'mlk', 'odr'])
    for task in patterns_roi[roi].keys():
        patterns_roi[roi][task] = []
    
for i in range(18):
    for roi in patterns_roi.keys():
        for task in patterns_roi[roi].keys():
            patterns_roi[roi][task].append(rms(np.array(patterns[task][i].loc[roi])))

for roi in patterns_roi.keys():
    patterns_roi[roi]['avg'] = []
    
for i in range(18):
    for roi in patterns_roi.keys():
        patterns_roi[roi]['avg'].append(np.array([patterns_roi[roi]['mlk'][i],
                                 patterns_roi[roi]['frt'][i],
                                 patterns_roi[roi]['odr'][i]]).mean(axis=0))    
    
for roi in patterns_roi.keys():
    sns.lineplot(x=times, y=np.array(patterns_roi[roi]['avg']).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('LD vs average(SD) RMS patterns')

plt.legend(patterns_roi.keys());
plt.show();