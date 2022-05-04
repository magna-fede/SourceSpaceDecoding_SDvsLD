# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:40:11 2021

@author: fm02
"""


# plot average activity for each vertex, for each participant
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

evoked_participants = []

for sub in np.arange(0  ,18):
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    with open(f'//cbsu/data/Imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
        output = pickle.load(f)
    
    print(f'Analysing participant {sub}')
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
    trials_ld = pd.DataFrame(columns=['ROI','category','trial','data'])
    
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
    
    # as we will consider each ROI separately, create a dataframe for each task
    all_trials = dict.fromkeys(['LD', 'MLK', 'FRT', 'ODR'])
    
    for key in all_trials.keys():
        all_trials[key] = pd.DataFrame(columns=kkROI)

    # in this script, the above passage is redundant (as we don't need to merge
    # data from the same trial for each ROI - but it's convenient in other
    # scripts, so keeping it.
    # get data for each task for each ROI
    for key in all_trials.keys():
        for ROI in kkROI:
            if key=='MLK':
                all_trials[key][ROI] = trials_mlk['data'][trials_mlk['ROI']==ROI].reset_index(drop=True)
            elif key=='FRT':
                all_trials[key][ROI] = trials_frt['data'][trials_frt['ROI']==ROI].reset_index(drop=True)
            elif key=='ODR':    
                all_trials[key][ROI] = trials_odr['data'][trials_odr['ROI']==ROI].reset_index(drop=True)
            elif key=='LD':
                all_trials[key][ROI] = trials_ld['data'][trials_ld['ROI']==ROI].reset_index(drop=True)
            
    avg = dict.fromkeys(['LD', 'MLK', 'FRT', 'ODR'])
    
    for key in avg.keys():
        avg[key] = dict.fromkeys(kkROI)
        for roi in kkROI:
            avg[key][roi] = []
    
    ########################################
    # HEY!                                 #
    # THINK ABOUT WHAT YOU WANT TO DO NOW! #
    ########################################
    
    #####
    # 1 #
    #####
    # this loops averages across trials (maintaining info from each vertex)
    # use when want to have mean activity at each vertex, across trials, for each timepoint
    # for task in avg.keys():
    #     for roi in kkROI:
    #     # transform in 3D-matrix
    #     # which has (n_vertices*timepoints*n_trials)
    #         avg_roi = np.dstack(all_trials[task][roi])
    #     # and average over trials (3rd dimension) for each vertex at each timepoint
    #         avg_roi = np.mean(avg_roi,2)
    #         avg[task][roi].append(avg_roi)
    
    
    #####
    # 2 #
    #####
    # this loops averages across vertices (maintaining info from each trial)
    # use when want to have mean activity in each roi, across vertices, for each timepoint

    for task in avg.keys():
        for roi in kkROI:
        # transform in 3D-matrix
        # which has (n_trials*n_vertices*timepoints*)
            avg_roi = np.stack(all_trials[task][roi])
        # and average over vertices (2nd dimension) for each trial at each timepoint
            avg_roi = np.mean(avg_roi, 1)
            avg[task][roi].append(avg_roi)

    evoked_participants.append(avg)
    
    
df_to_export = pd.DataFrame(evoked_participants)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/evoked_responses/0428_evoked.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

    
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/evoked_responses/0428_evoked.P", 'rb') as f:
      evoked = pickle.load(f)
   

# for roi in kkROI: 
#     sns.set(style="ticks", rc={"lines.linewidth": 0.9,
#                                 'figure.figsize':(15,10)})
#     for vertex in avg_LD[roi][0]:
#         sns.lineplot(x=np.arange(-300,900,4), y=vertex)
#     plt.title(roi)
#     plt.show();      
      
from scipy.stats import sem    


tasks = dict.fromkeys(['LD', 'MLK', 'FRT', 'ODR'])

kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

for task in tasks:
    tasks[task] = dict.fromkeys(kkROI)
    for roi in kkROI:
        tasks[task][roi] = []

for i in range(18):
    for task in tasks:
        for roi in kkROI:
            tasks[task][roi].append(np.array(evoked[task][i][roi][0]).mean(0))
 
from scipy.stats import ttest_1samp
from scipy import stats

from mne.stats import permutation_cluster_test
   
_ , LDpvalues = ttest_1samp(LD_mean, popmean=.5, axis=0)


X = dict.fromkeys(kkROI)

for roi in kkROI:
    X[roi] = []
    for task in tasks:
        X[roi].append(np.array(tasks[task][roi]))

F_clust, clusters, p_values, H0 = permutation_cluster_test(
   X['PVA'], n_jobs=1)


for task in tasks:
    X[]

# Reshape data to what is equivalent to (n_samples, n_space, n_time)
LD_mean.shape = (18, 1, 300)
# Compute threshold from t distribution (this is also the default)
threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
Lt_clust, Lclusters, Lp_values, H0 = permutation_cluster_1samp_test(
    LD_mean-.2, n_jobs=1, threshold=threshold, adjacency=None,
    n_permutations='all')
# Put the cluster data in a viewable format
Lp_clust = np.ones((1,300))
for cl, p in zip(Lclusters, Lp_values):
    Lp_clust[cl] = p
           
 
for roi in kkROI:
    for task in tasks.keys(): 
        sns.lineplot(x=np.arange(-300,900,4),y=np.array(tasks[task][roi]).mean(0))
        plt.fill_between(x=np.arange(-300,900,4), \
                          y1=(np.array(tasks[task][roi]).mean(0)-sem(np.array(tasks[task][roi]),0)), \
                          y2=(np.array(tasks[task][roi]).mean(0)+sem(np.array(tasks[task][roi]),0)), \
                          alpha=.2)
    plt.title(roi)
    plt.legend(tasks.keys())
    plt.show()
plt.show();