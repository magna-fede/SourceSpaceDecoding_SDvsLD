#!/usr/bin/env python
# coding: utf-8

# # Author: Federica.Magnabosco@mrc-cbu.cam.ac.uk
# 
# In this project I will apply some of the notions lernt during the course to try to predict which brain regions and at which time point are sensitive to the different amount of semantic resources necessary for completing two different tasks. To do this, we will look at the source estimated activity of 6 Regions of Interest (ROIs) for one participant. The two tasks (lexical decision and semantic decision) are belived to vary in the amount of semantic resources necessary for completing the task. The activity is related to -300 ms to 900 ms post stimulus presentation.
# In this script, we will try to predict semantic word category.
# Import some relevant packages.

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

def divide_semK(trials):
    """Divide data based on which semantic category each trial belongs to."""
    dic = {}
    # loop over semantic category
    for semK in kk2:
        dic[semK] = []
        # loop over each trial (based on trial number)
        for trial in trials['trial'][trials['category']==semK].unique():
            dic[semK].append(np.concatenate \
                              (trials['data'] \
                               [(trials['category']==semK) \
                                & (trials['trial']==trial)].values)) 
        dic[semK] = np.array(dic[semK])
    return dic

# initialise dictionaries for storing scores and patterns
participant_scores = []
participant_patterns = []

# loop over participants
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
    
    
    kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
    kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']
    
    
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
    
    
    for j,k in enumerate(kk):
        # check the key identity about the task
        if k[0:2] == 'LD':
            # check which semantic category
            # (this checks if each category in kk2,
            # is present in k, the output[key] currently considered)
            mask_k = [k2 in k for k2 in kk2]
            # and save the category as a string
            k2 = np.array(kk2)[mask_k][0]
            # check which ROI
            mask_ROI = [k_ROI in k for k_ROI in kkROI]
            kROI = np.array(kkROI)[mask_ROI][0]
            # loop over trials
            # this extracts data from the relevant key
            # and save data as a pandas dataframe (easier to access info)
            for i in range(len(output[k])):
                # save data (contained in output[k])
                # for each trial (i) separately
                ls = [kROI, k2, i, output[k][i]]
                # containing info about semantic_category, trial, and data
                row = pd.Series(ls, index=trials_ld.columns)
                # and append data to relevant dataframe
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
 
    trials = {}
    trials['ld'] = trials_ld
    trials['mlk'] = trials_mlk
    trials['frt'] = trials_frt
    trials['odr'] = trials_odr
    
    trials_semK = {}

    # divide trials based on semantic category
    for tsk in list(trials.keys()):
        trials_semK[tsk] = divide_semK(trials[tsk])     
    
    # now let's average 3 trials together

    sub = {}
    sub['ld'] = dict.fromkeys(kk2)
    sub['mlk'] = dict.fromkeys(kk2)
    sub['frt'] = dict.fromkeys(kk2)
    sub['odr'] = dict.fromkeys(kk2)
    
    for dic in sub.keys():
        for semK in kk2:
            sub[dic][semK] = []
    
    for tsk in trials_semK.keys():
     
        # make sure the number of trials is a multiple of 3, or eliminate excess
        
        for k in trials_semK[tsk].keys():
            
            while len(trials_semK[tsk][k])%3 != 0:
                trials_semK[tsk][k] = np.delete(trials_semK[tsk][k],
                                                len(trials_semK[tsk][k])-1, 0)
        # create random groups of trials
            new_tsk = np.split(trials_semK[tsk][k],len(trials_semK[tsk][k])/3)
            new_trials = []
        # calculate average for each timepoint of the 3 trials
    # calculate average for each timepoint (axis=0) of the 3 trials
            for nt in new_tsk:
                new_trials.append(np.mean(np.array(nt),0))
    
            # assign group to the corresponding task in the dict       
            sub[tsk][k] = np.array(new_trials)
            
    trials_semK = sub

    # retrive information about the vertices    
    vertices = []
    
    # using 'frt', 'visual', trial=0, as vertices order doesn't change, 
    # so it doesn't matter which task, category and trial looking up    
    for roi in trials['frt'][(trials['frt']['category']=='visual') & \
                             (trials['frt']['trial']==0)]['data']:
        vertices.append(roi.shape[0])
    
    print([v for v in vertices])
    
    ROI_vertices = []
 
    # create list with length=n_vertices containing ROI string for each vertex       
    for i in range(len(vertices)):
        ROI_vertices.extend([kkROI[i]]*vertices[i])
    
    # We create and run the model.
    # this is taken from MNE example https://mne.tools/stable/auto_examples/decoding/decoding_spatio_temporal_source.html

    # We expect the model to perform at chance before the presentation of the stimuli
    # (no ROI should be sensitive to task/semantics demands before the presentation of a word).
    
    # prepare a series of classifier applied at each time sample
    # this is the classifier
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearModel(LogisticRegression(C=1,
                                                       solver='liblinear'))) # asking LDA to store covariance
    
    # Search Light
    # "Fit, predict and score a series of models to each subset of the dataset along the last dimension"
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    
    # create combination of all the possible comparisons among semantic categories    
    comb = []
    
    # len(kk2) = 5
    # len(groups) = 2
    # so len(comb) = 10
    for i in combinations(kk2,2):
        comb.append(i)
    
    scores = {}
    scores['ld'] = []
    scores['mlk'] = []
    scores['frt'] = []
    scores['odr'] = []
    
    patterns = {}
    patterns['ld'] = dict.fromkeys(kkROI)
    patterns['mlk'] = dict.fromkeys(kkROI)
    patterns['frt'] = dict.fromkeys(kkROI)
    patterns['odr'] = dict.fromkeys(kkROI)
    
    for tsk in patterns.keys():
        for roi in patterns[tsk].keys():
            patterns[tsk][roi] = []
    
    # loop over task
    for task in trials_semK.keys():
        # loop over each combination of semK classification
        for semKvsemK in comb:

            # X input matrix, containing LD and task trials, it has dimension n_trial*n_vertices*n_timepoints
            X = np.concatenate([trials_semK[task][semKvsemK[0]],
                                    trials_semK[task][semKvsemK[1]]])
            
            # Y category array. it has dimension n_trial            
            y = np.array([semKvsemK[0]]*len(trials_semK[task][semKvsemK[0]]) + \
                             [semKvsemK[1]]*len(trials_semK[task][semKvsemK[1]]))
            
            # shuffle them, not sure it is necessary     
            X, y = shuffle(X, y, random_state=0)
            
            # append the average of 5-fold cross validation to the scores dict for this task
            scores[task].append(cross_val_multiscore(time_decod,
                                                     X, y, cv=5).mean(axis=0))
            
            # now let's look at the weight backprojection,
            # MNE says "The fitting needs not be cross validated because the weights are based on
            # the training sets"        
            time_decod.fit(X, y)
            
            # this already applies Haufe's trick
            # Retrieve patterns after inversing the z-score normalization step            
            pattern = get_coef(time_decod, 'patterns_', inverse_transform=True)
            
            # append ROI information
            pattern = pd.DataFrame(pattern, index=ROI_vertices)
            
            # already append the ROOT-MEAN-SQUARE for patterns for that semK comparison
            # in each roi, for each task and participant
            for roi in kkROI:
                # this means that for each task, for each ROI, we have 10 patterns
                # (because 10 different combinations of semK)
                patterns[task][roi].append(rms(np.array(pattern.loc[roi])))
    
    participant_scores.append(scores)
    participant_patterns.append(patterns)
 
 # save the scores ...   
df_to_export = pd.DataFrame(participant_scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0113_LogReg_AVG_SemK_scores.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)

# ... and the patterns      
df_to_export = pd.DataFrame(participant_patterns)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0113_LogReg_AVG_SemK_patterns.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)

##########################################################

# this part will plot the results,
# you can either
    # comment previous part (leaving the imported packages)
    # or comment the loading part if you want to calculate again scores/patterns 
    
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0113_LogReg_AVG_SemK_scores.P", 'rb') as f:
    participant_scores = pickle.load(f)


with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0113_LogReg_AVG_SemK_patterns.P", 'rb') as f:
    participant_patterns = pickle.load(f)


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# create times array
times = np.arange(-300,900,4)

# initialise patterns where to store mean over semK
patterns_mean = {}
patterns_mean['ld'] = dict.fromkeys(kkROI)
patterns_mean['mlk'] = dict.fromkeys(kkROI)
patterns_mean['frt'] = dict.fromkeys(kkROI)
patterns_mean['odr'] = dict.fromkeys(kkROI)

# loop over task
for tsk in patterns_mean.keys():
    # and over rois
    for roi in patterns_mean[tsk].keys():
        patterns_mean[tsk][roi] = []

# loop over pariticpants
for sub in range(18):
    # loop over task
    for task in patterns_mean.keys():
        # loop over ROIs
        for roi in kkROI:
            # append the average of the RMS(patterns) over each semK comparison
            patterns_mean[task][roi].append(np.array(participant_patterns[sub][task][roi]).mean(axis=0))

# now calculate the average of RMS across tasks for each participant
patterns_mean['avg'] = dict.fromkeys(kkROI)
for roi in patterns_mean['avg'].keys():
    patterns_mean['avg'][roi] = []

# loop over participants    
for i in range(18):
    # loop over ROIs
    for roi in patterns_mean['avg'].keys():
        # append the average of RMS(pattern) of each SD task
        # (this is already the average across semK comparison)
        patterns_mean['avg'][roi].append(np.array([patterns_mean['mlk'][roi][i],
                                 patterns_mean['frt'][roi][i],
                                 patterns_mean['odr'][roi][i]]).mean(axis=0))    
    

# now plot!

# all ROIs in one plot, this is Semantic Decision average
for roi in patterns_mean['avg'].keys():
    sns.lineplot(x=times, y=np.array(patterns_mean['avg'][roi]).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('WordSemK SEMANTIC RMS patterns')
plt.legend(patterns_mean['avg'].keys());
plt.show();
plt.savefig(f'U:/Decoding_SDLD/Figures/LogReg_semK_avgSD_patterns.png', format='png')


for roi in patterns_mean['mlk'].keys():
    sns.lineplot(x=times, y=np.array(patterns_mean['mlk'][roi]).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('WordSemK MILK RMS patterns')
plt.legend(patterns_mean['mlk'].keys());
plt.show();
plt.savefig(f'U:/Decoding_SDLD/Figures/LogReg_semK_MILK_patterns.png', format='png')


for roi in patterns_mean['frt'].keys():
    sns.lineplot(x=times, y=np.array(patterns_mean['frt'][roi]).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('WordSemK FRUIT RMS patterns')
plt.legend(patterns_mean['frt'].keys());
plt.show();
plt.savefig(f'U:/Decoding_SDLD/Figures/LogReg_semK_FRUIT_patterns.png', format='png')

for roi in patterns_mean['odr'].keys():
    sns.lineplot(x=times, y=np.array(patterns_mean['odr'][roi]).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('WordSemK ODOUR RMS patterns')
plt.legend(patterns_mean['odr'].keys());
plt.show();
plt.savefig(f'U:/Decoding_SDLD/Figures/LogReg_semK_ODOUR_patterns.png', format='png')


for roi in patterns_mean['ld'].keys():
    sns.lineplot(x=times, y=np.array(patterns_mean['ld'][roi]).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('WordSemK LEXICAL DECISION RMS patterns')
plt.legend(patterns_mean['ld'].keys());
plt.show();
plt.savefig(f'U:/Decoding_SDLD/Figures/LogReg_semK_LD_patterns.png', format='png')

######## now working on accuracy (aka scores)

# get the mean over the 10 different combinations
scores_mean = dict.fromkeys(['ld', 
                             'mlk', 
                             'frt', 
                             'odr'])

for tsk in scores_mean.keys():
    scores_mean[tsk] = []

# loop over participants
for sub in range(18):
    # loop over tasks
    for task in scores_mean.keys():
        # append average performance over each semK combination
        scores_mean[task].append(np.array(participant_scores[sub][task]).mean(axis=0))

scores_mean['avg'] = []

for i in range(18):
    # append average performance across various SD tasks
    scores_mean['avg'].append(np.array([scores_mean['mlk'][i],
                                 scores_mean['frt'][i],
                                 scores_mean['odr'][i]]).mean(axis=0))    

# calculate when accuracy is significantly different from chance

from scipy.stats import ttest_1samp
from scipy import stats

from mne.stats import permutation_cluster_1samp_test

# this is taken from MNE's python example on how to perform 
# Non-parametric cluster-level paired t-test.
# https://mne.tools/stable/auto_tutorials/stats-sensor-space/10_background_stats.html#sphx-glr-auto-tutorials-stats-sensor-space-10-background-stats-py
   
_ , SDpvalues = ttest_1samp(scores_mean['avg'], popmean=.5, axis=0)

SDp_clust = np.zeros(300)
SD_mean = np.array(scores_mean['avg'])
# Reshape data to what is equivalent to (n_samples, n_space, n_time)
SD_mean.shape = (18, 1, 300)
# Compute threshold from t distribution (this is also the default)
threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
SDt_clust, SDclusters, SDp_values, H0 = permutation_cluster_1samp_test(
    SD_mean-.5, n_jobs=1, threshold=threshold, adjacency=None,
    n_permutations='all')
# Put the cluster data in a viewable format
p_clust = np.ones((1,300))
for cl, p in zip(SDclusters, SDp_values):
    p_clust[cl] = p
SDp_clust = p_clust.T
SDp_clust.shape = (300)

_ , LDpvalues = ttest_1samp(scores_mean['ld'], popmean=.5, axis=0)

LDp_clust = np.zeros(300)
LD_mean = np.array(scores_mean['ld'])
# Reshape data to what is equivalent to (n_samples, n_space, n_time)
LD_mean.shape = (18, 1, 300)
# Compute threshold from t distribution (this is also the default)
threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
LDt_clust, LDclusters, LDp_values, H0 = permutation_cluster_1samp_test(
    LD_mean-.5, n_jobs=1, threshold=threshold, adjacency=None,
    n_permutations='all')
# Put the cluster data in a viewable format
p_clust = np.ones((1,300))
for cl, p in zip(LDclusters, LDp_values):
    p_clust[cl] = p
LDp_clust = p_clust.T
LDp_clust.shape = (300)

# now plot accuracies, with areas where significantly different from chance

sns.lineplot(x=times, y=np.array(scores_mean['avg']).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
# check is at least one point where it's significant
if len(times[np.where(SDp_clust < 0.05)]) > 0:
    ##### CAREFUL! this is wrong way to do this, check in accuracy script
    for tp in times[np.where(SDp_clust < 0.05)]:
        plt.axvspan(tp,tp,
               label="Cluster based permutation p<.05",
               color="green", alpha=0.3)
plt.title(f'avg(SD) Accuracy Semantic Category Decoding')
plt.axhline(.5, color='k', linestyle='--', label='chance');
# plt.savefig(f'LD_{roi}.png', format='png')
# plt.legend();
plt.show();
    
sns.lineplot(x=times, y=np.array(scores_mean['ld']).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
if len(times[np.where(LDp_clust < 0.05)]) > 0:
    for tp in times[np.where(LDp_clust < 0.05)]:
        plt.axvspan(tp,tp,
               label="Cluster based permutation p<.05",
               color="green", alpha=0.3)
plt.title(f'LD Accuracy Semantic Category Decoding')
plt.axhline(.5, color='k', linestyle='--', label='chance');
# plt.savefig(f'LD_{roi}.png', format='png')
# plt.legend();
plt.show();
    

sns.lineplot(x=times, y=np.array(scores_mean['mlk']).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('WordSemK MILK accuracy')
plt.show();
sns.lineplot(x=times, y=np.array(scores_mean['frt']).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('WordSemK FRUIT accuracy')
plt.show();
sns.lineplot(x=times, y=np.array(scores_mean['odr']).mean(axis=0))
# plt.fill_between(x=times, \
#                  y1=(np.mean(np.array(scores['avg']),0)-sem(np.array(scores['avg']),0)), \
#                  y2=(np.mean(np.array(scores['avg']),0)+sem(np.array(scores['avg']),0)), \
#                  color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('WordSemK ODOUR accuracy')
plt.show();
