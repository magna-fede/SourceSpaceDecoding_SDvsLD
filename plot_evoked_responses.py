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

def rms(example):
    """Compute root mean square of each ROI.
    Input is a dataframe of length=n_vertices."""
    # first transform Series in np array of dimension n_vertics*timepoints
    
    return np.sqrt(np.mean(example**2))

# initialise lists where we'll store output

list_avg_scores = []
list_mlk_scores = []
list_frt_scores = []
list_odr_scores = []

evoked_participants = []

for sub in np.arange(0, 18):
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
            ###############
            ### AVERAGE ###
            ###############
        # # transform in 3D-matrix
        # # which has (n_trials*n_vertices*timepoints*)
        #     avg_roi = np.stack(all_trials[task][roi])
        # # and average over vertices (2nd dimension) for each trial at each timepoint
        #     avg_roi = np.mean(avg_roi, 1)
        #     avg[task][roi].append(avg_roi)
            ###########
            ### RMS ###
            ###########
            avg_roi = np.stack(all_trials[task][roi])
            avg_roi = np.apply_along_axis(rms, 1, np.stack(all_trials[task][roi]))
            # and average over vertices (2nd dimension) for each trial at each timepoint
            avg[task][roi].append(avg_roi)
            

    evoked_participants.append(avg)
    
    
# df_to_export = pd.DataFrame(evoked_participants)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/evoked_responses/0722_evoked_acrossROI.P", 'wb') as outfile:
#     pickle.dump(df_to_export,outfile)

df_to_export = pd.DataFrame(evoked_participants)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/evoked_responses/0722_evoked_RMSacrossROI.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

    
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/evoked_responses/0722_evoked_acrossROI.P", 'rb') as f:
      evoked = pickle.load(f)
      
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/evoked_responses/0722_evoked_RMSacrossROI.P", 'rb') as f:
#       evoked = pickle.load(f)      

colors = sns.color_palette(['#FFBE0B',
                            '#FB5607',
                            '#FF006E',
                            '#8338EC',
                            '#3A86FF',
                            '#1D437F'
                            ])

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

times = np.arange(-300,900,4)

for task in tasks:
    tasks[task] = dict.fromkeys(kkROI)
    for roi in kkROI:
        tasks[task][roi] = []

# for each participant, calculate average in each task, in each roi 
for i in range(18):
    for task in tasks:
        for roi in kkROI:
            #tasks[task][roi].append(np.apply_along_axis(rms, 0, evoked[task][i][roi][0]))
            tasks[task][roi].append(np.array(evoked[task][i][roi][0]).mean(0))

tasks['avg_SD'] = dict.fromkeys(kkROI)
for roi in kkROI:
    tasks['avg_SD'][roi] = []

for i in range(18):
    for roi in kkROI:
        # numpy array of size 3(tasks)*300(timepoints)
        all_sds = np.array([tasks['FRT'][roi][i], tasks['ODR'][roi][i], tasks['MLK'][roi][i]])
        # average across SD tasks, for each participant
        tasks['avg_SD'][roi].append(all_sds.mean(0))

for task in tasks.keys():
    for roi in kkROI:
        tasks[task][roi] = np.array(tasks[task][roi])
        tasks[task][roi].shape =(18, 1, 300) 

from scipy.stats import ttest_1samp
from scipy import stats
from scipy.stats import f_oneway

from mne.stats import permutation_cluster_test
from mne.stats import permutation_cluster_1samp_test
   
LDvsSD = dict.fromkeys(kkROI)

for roi in kkROI:
    t_obs, Lclusters, Lp_values, H0 = permutation_cluster_1samp_test(
        (tasks['LD'][roi] - tasks['avg_SD'][roi]),
        n_permutations=1000,
        n_jobs=1)
    Lp_clust = np.ones((1,300))
    for cl, p in zip(Lclusters, Lp_values):
        Lp_clust[cl] = p
    LDvsSD[roi] = Lp_clust.T

for task in tasks.keys():
    for roi in kkROI:
        tasks[task][roi].shape =(18, 300) 

for roi in kkROI:
    sns.lineplot(times, tasks['LD'][roi].mean(0), label='LD', color=colors[1])
    sns.lineplot(times, tasks['avg_SD'][roi].mean(0), label='average SD', color=colors[3])    
    plt.fill_between(x=times, \
                      y1=(tasks['LD'][roi].mean(0) - sem(tasks['LD'][roi],0)), \
                      y2=(tasks['LD'][roi].mean(0) + sem(tasks['LD'][roi],0)), \
                      color=colors[1], alpha=.1)
    plt.fill_between(x=times, \
                      y1=(tasks['avg_SD'][roi].mean(0) - sem(tasks['avg_SD'][roi],0)), \
                      y2=(tasks['avg_SD'][roi].mean(0) + sem(tasks['avg_SD'][roi],0)),                    
                      color=colors[3], alpha=.1)
    plt.axvline(0, color='k');
    plt.axhline(0, color='k', alpha=0.3, linewidth = 0.5);
    mask = (LDvsSD[roi] < 0.05).reshape(-1)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                   label="Cluster-based permutation p<.05",
                   color="green")
        
    # plt.legend();
    mask = stats.ttest_1samp(tasks['LD'][roi] - tasks['avg_SD'][roi], popmean=0)[1] < 0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                   label="uncorrected p<.05",
                   color="yellow")    
    plt.title(f'RMS LD vs SD - {roi}')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.savefig(f'RMS evoked_LDvsSD_{roi}.png', format='png')
    plt.show();


SDvsSD = dict.fromkeys(kkROI)

for roi in kkROI:
    F_obs, SDclusters, SDp_values, H0 = permutation_cluster_test(
        ([tasks['MLK'][roi], tasks['ODR'][roi], tasks['FRT'][roi]]),
        n_permutations=1000,
        n_jobs=1)
    SDp_clust = np.ones(300)
    for cl, p in zip(SDclusters, SDp_values):
        SDp_clust[cl] = p
    SDvsSD[roi] = SDp_clust.T

for task in tasks.keys():
    for roi in kkROI:
        tasks[task][roi].shape =(18, 300) 

for roi in kkROI:
    sns.lineplot(times, tasks['MLK'][roi].mean(0), label='SD milk', color=colors[0])
    sns.lineplot(times, tasks['ODR'][roi].mean(0), label='SD odour', color=colors[2])
    sns.lineplot(times, tasks['FRT'][roi].mean(0), label='SD fruit', color=colors[4])
    plt.fill_between(x=times, \
                      y1=(tasks['MLK'][roi].mean(0) - sem(tasks['MLK'][roi], 0)), \
                      y2=(tasks['MLK'][roi].mean(0) + sem(tasks['MLK'][roi], 0)), \
                      color=colors[0], alpha=.1)
    plt.fill_between(x=times, \
                      y1=(tasks['ODR'][roi].mean(0) - sem(tasks['ODR'][roi], 0)), \
                      y2=(tasks['ODR'][roi].mean(0) + sem(tasks['ODR'][roi], 0)), \
                      color=colors[2], alpha=.1)
    plt.fill_between(x=times, \
                      y1=(tasks['FRT'][roi].mean(0) - sem(tasks['FRT'][roi], 0)), \
                      y2=(tasks['FRT'][roi].mean(0) + sem(tasks['FRT'][roi], 0)), \
                      color=colors[4], alpha=.1)
    plt.axvline(0, color='k');
    plt.axhline(0, color='k', alpha=0.3, linewidth = 0.5);
    mask = (SDvsSD[roi] < 0.05).reshape(-1)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                   label="Cluster based permutation p<.05",
                   color="green")
        
    # plt.legend();
    mask = f_oneway(tasks['MLK'][roi], tasks['ODR'][roi], tasks['FRT'][roi], axis =0)[1] <0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                   label="uncorrected p<.05",
                   color="yellow")    
    plt.title(f'RMS SD vs SD - {roi}')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    #plt.savefig(f'RMS evoked_SDvsSD_{roi}.png', format='png')
    # plt.legend();
    plt.show();
           

sns.set_palette(colors)
 
for roi in kkROI:
    for task in tasks.keys(): 
        sns.lineplot(x=np.arange(-300,900,4),y=np.array(tasks[task][roi]).mean(0))
        plt.fill_between(x=np.arange(-300,900,4), \
                          y1=(tasks[task][roi].mean(0) - sem(tasks[task][roi], 0)), \
                          y2=(tasks[task][roi].mean(0) + sem(tasks[task][roi], 0)), \
                          alpha=.1)
    plt.axvline(0, color='k');
    plt.axhline(0, color='k', alpha=0.3, linewidth = 0.5);
    plt.title(f"RMS evoked response {roi}")
    plt.legend(tasks.keys())
    #plt.savefig(f'RMS evoked_{roi}.png', format='png')    
    plt.show()