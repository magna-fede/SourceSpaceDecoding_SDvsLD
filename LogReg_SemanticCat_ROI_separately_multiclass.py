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


participant_scores = []

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
    
    trials_semK = {}
    
    for tsk in list(trials.keys()):
        trials_semK[tsk] = divide_semK_ROI(trials[tsk])
        
    # try not averaging because not enough trials otherwise
    
    # now let's average 3 trials together
    sub_lds = {}
    sub_frts = {}
    sub_mlks = {}
    sub_odrs = {}
    
    for dic in [sub_lds, sub_frts, sub_mlks, sub_odrs]:
        for semK in kk2:
            dic[semK] = dict.fromkeys(kkROI)
            for i in dic[semK].keys():
                dic[semK][i] = []
            
    
    for i, tsk in enumerate(trials_semK.values()):
     
        # make sure the number of trials is a multiple of 3, or eliminate excess
        for k in tsk.keys():
            for roi in tsk[k].keys():
                        
                while len(tsk[k][roi])%3 != 0:
                    tsk[k][roi] = np.delete(tsk[k][roi], len(tsk[k][roi])-1, 0)
        # create random groups of trials
                new_tsk = np.split(tsk[k][roi],len(tsk[k][roi])/3)
                new_trials = []
            # calculate average for each timepoint of the 3 trials
                for nt in new_tsk:
                    new_trials.append(np.mean(nt,0))
                # assign group it in the corresponding task
                
                if i==0:
                    sub_lds[k][roi] = new_trials
                elif i==1:
                    sub_mlks[k][roi] = new_trials
                elif i==2:
                    sub_frts[k][roi] = new_trials
                elif i==3:
                    sub_odrs[k][roi] = new_trials
            
    subt = {}
    subt['ld'] = sub_lds
    subt['mlk'] = sub_mlks
    subt['frt'] = sub_frts
    subt['odr'] = sub_odrs
    
    trials_semK = subt
    
    # We create and run the model. We expect the model to perform at chance before the presentation of the stimuli (no ROI should be sensitive to task/semantics demands before the presentation of a word).
    
    # prepare a series of classifier applied at each time sample
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearModel(LogisticRegression(C=1,
                                                       solver='lbfgs',
                                                       max_iter=1000))) # asking LDA to store covariance
    time_decod = SlidingEstimator(clf)
    
    scores = {}
    scores['ld'] = dict.fromkeys(kkROI)
    scores['mlk'] = dict.fromkeys(kkROI)
    scores['frt'] = dict.fromkeys(kkROI)
    scores['odr'] = dict.fromkeys(kkROI)
    
    for tsk in scores.keys():
        for roi in scores[tsk].keys():
            scores[tsk][roi] = []
    
    # just use subt instead of trials_semK if you want to have average of trials
    for task in trials_semK.keys():
        for roi in kkROI:
            a = pd.DataFrame(columns=[roi,'semK'])
            for semK in subt['frt']:
                a = pd.concat([a,
                               pd.DataFrame(zip(pd.DataFrame.from_dict(subt[task][semK])[roi],
                                                [semK]*len(pd.DataFrame.from_dict(subt[task][semK])[roi])),
                                            columns=[roi,'semK'])],
                              ignore_index=True)
                
            X = np.stack(a[roi])
            
            y = a['semK']
            X, y = shuffle(X, y, random_state=0)
            
            scores[task][roi] = cross_val_multiscore(time_decod,
                                                     X, y, cv=5)
        
            
    participant_scores.append(scores)
    
    



# df_to_export = pd.DataFrame(SDLD_scores)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1104_LDA_SDLD_scores.P",
#           'wb') as outfile:
#     pickle.dump(df_to_export,outfile)
    
df_to_export = pd.DataFrame(participant_scores)
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0315_LogReg_ROI_SemK_scores_multiclass.P",
          'wb') as outfile:
    pickle.dump(df_to_export,outfile)

    
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0315_LogReg_ROI_SemK_scores_multiclass.P", 'rb') as f:
      df_to_export = pickle.load(f)

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem


LD_mean = dict.fromkeys(kkROI)

for k in LD_mean.keys():
    LD_mean[k] = []

for participant in df_to_export['ld']:
    for roi in kkROI:
        avg_task = np.array(participant[roi]).mean(axis=0)
        LD_mean[roi].append(avg_task)

for k in LD_mean.keys():
    LD_mean[k] = np.array(LD_mean[k])

MLK_mean = dict.fromkeys(kkROI)

for k in LD_mean.keys():
    MLK_mean[k] = []

for participant in df_to_export['mlk']:
    for roi in kkROI:
        avg_task = np.array(participant[roi]).mean(axis=0)
        MLK_mean[roi].append(avg_task)

for k in MLK_mean.keys():
    MLK_mean[k] = np.array(MLK_mean[k])


FRT_mean = dict.fromkeys(kkROI)

for k in FRT_mean.keys():
    FRT_mean[k] = []

for participant in df_to_export['frt']:
    for roi in kkROI:
        avg_task = np.array(participant[roi]).mean(axis=0)
        FRT_mean[roi].append(avg_task)

for k in FRT_mean.keys():
    FRT_mean[k] = np.array(FRT_mean[k])
    
ODR_mean = dict.fromkeys(kkROI)

for k in ODR_mean.keys():
    ODR_mean[k] = []

for participant in df_to_export['odr']:
    for roi in kkROI:
        avg_task = np.array(participant[roi]).mean(axis=0)
        ODR_mean[roi].append(avg_task)

for k in ODR_mean.keys():
    ODR_mean[k] = np.array(ODR_mean[k])

from scipy import stats

from mne.stats import permutation_cluster_1samp_test


Lp_clust = pd.DataFrame(index=range(300),columns=kkROI)
for roi in kkROI:
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    LD_mean[roi].shape = (18, 1, 300)
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
    Lt_clust, Lclusters, Lp_values, H0 = permutation_cluster_1samp_test(
        LD_mean[roi]-.2, n_jobs=1, threshold=threshold, adjacency=None,
        n_permutations='all')
    # Put the cluster data in a viewable format
    p_clust = np.ones((1,300))
    for cl, p in zip(Lclusters, Lp_values):
        p_clust[cl] = p
    Lp_clust[roi] = p_clust.T
    
    
Mp_clust = pd.DataFrame(index=range(300),columns=kkROI)
for roi in kkROI:
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    MLK_mean[roi].shape = (18, 1, 300)
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
    Mt_clust, Mclusters, Mp_values, H0 = permutation_cluster_1samp_test(
        MLK_mean[roi]-.2, n_jobs=1, threshold=threshold, adjacency=None,
        n_permutations='all')
    # Put the cluster data in a viewable format
    p_clust = np.ones((1,300))
    for cl, p in zip(Mclusters, Mp_values):
        p_clust[cl] = p
    Mp_clust[roi] = p_clust.T

Fp_clust = pd.DataFrame(index=range(300),columns=kkROI)
for roi in kkROI:
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    FRT_mean[roi].shape = (18, 1, 300)
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
    Ft_clust, Fclusters, Fp_values, H0 = permutation_cluster_1samp_test(
        FRT_mean[roi]-.2, n_jobs=1, threshold=threshold, adjacency=None,
        n_permutations='all')
    # Put the cluster data in a viewable format
    p_clust = np.ones((1,300))
    for cl, p in zip(Fclusters, Fp_values):
        p_clust[cl] = p
    Fp_clust[roi] = p_clust.T
    
Op_clust = pd.DataFrame(index=range(300),columns=kkROI)
for roi in kkROI:
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    ODR_mean[roi].shape = (18, 1, 300)
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
    Ot_clust, Oclusters, Op_values, H0 = permutation_cluster_1samp_test(
        ODR_mean[roi]-.2, n_jobs=1, threshold=threshold, adjacency=None,
        n_permutations='all')
    # Put the cluster data in a viewable format
    p_clust = np.ones((1,300))
    for cl, p in zip(Oclusters, Op_values):
        p_clust[cl] = p
    Op_clust[roi] = p_clust.T
    

times = np.arange(-300,900,4)

print(f'MILK TASK : Decoding semantic category at timepoints: \
      {times[np.where(Mp_clust < 0.05)[1]]}')
print(f'FRUIT TASK : Decoding semantic category at timepoints: \
      {times[np.where(Fp_clust < 0.05)[1]]}')
print(f'ODOUR TASK : Decoding semantic category at timepoints: \
      {times[np.where(Op_clust < 0.05)[1]]}')
print(f'LEXICAL DECISION: Decoding semantic category at timepoints: \
      {times[np.where(Lp_clust < 0.05)[1]]}')

for roi in kkROI:
    LD_mean[roi] = np.array(LD_mean[roi]).reshape((18,300))
    MLK_mean[roi] = np.array(MLK_mean[roi]).reshape((18,300))
    FRT_mean[roi] = np.array(FRT_mean[roi]).reshape((18,300))
    ODR_mean[roi] = np.array(ODR_mean[roi]).reshape((18,300))

for roi in kkROI:
    sns.lineplot(x=times, y=np.mean(LD_mean[roi],0))
    plt.fill_between(x=times, \
                      y1=(np.mean(LD_mean[roi],0)-sem(LD_mean[roi],0)), \
                      y2=(np.mean(LD_mean[roi],0)+sem(LD_mean[roi],0)), \
                      color='b', alpha=.1)
    plt.axvline(0, color='k');
    plt.axvline(50, color='k', linewidth=1, alpha=0.3);
    plt.axvline(100, color='k',linewidth=1, alpha=0.3);
    plt.axvline(150, color='k', linewidth=1, alpha=0.3);
    plt.axvline(200, color='k', linewidth=1, alpha=0.3);
    mask = Lp_clust[roi] < 0.05
    mask = mask.values.reshape(300)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                   label="Cluster based permutation p<.05",
                   color="green")
    # plt.legend();
    mask = stats.ttest_1samp(LD_mean[roi], .2)[1] < 0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                   label="Cluster based permutation p<.05",
                   color="yellow")    
    plt.title(f'LD - {roi} Semantic Category Decoding')
    plt.axhline(.2, color='k', linestyle='--', label='chance');
    plt.savefig(f'LD_{roi}.png', format='png')
    # plt.legend();
    plt.show();

for roi in kkROI:
    sns.lineplot(x=times, y=np.mean(MLK_mean[roi],0))
    plt.fill_between(x=times, \
                      y1=(np.mean(MLK_mean[roi],0)-sem(MLK_mean[roi],0)), \
                      y2=(np.mean(MLK_mean[roi],0)+sem(MLK_mean[roi],0)), \
                      color='b', alpha=.1)
    plt.axvline(0, color='k');
    plt.axvline(50, color='k', linewidth=1, alpha=0.3);
    plt.axvline(100, color='k',linewidth=1, alpha=0.3);
    plt.axvline(150, color='k', linewidth=1, alpha=0.3);
    plt.axvline(200, color='k', linewidth=1, alpha=0.3);
    plt.axhline(.2, color='k', linestyle='--', label='chance');
    mask = Mp_clust[roi] < 0.05
    mask = mask.values.reshape(300)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                   label="Cluster based permutation p<.05",
                   color="green")
    # plt.legend();
    mask = stats.ttest_1samp(MLK_mean[roi], .2)[1] < 0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                   label="Cluster based permutation p<.05",
                   color="yellow")    
    plt.title(f'MILK - {roi} Semantic Category Decoding');
    plt.savefig(f'MILK_{roi}.png', format='png');
    # plt.legend();
    plt.show();

for roi in kkROI:
    sns.lineplot(x=times, y=np.mean(FRT_mean[roi],0))
    plt.fill_between(x=times, \
                      y1=(np.mean(FRT_mean[roi],0)-sem(FRT_mean[roi],0)), \
                      y2=(np.mean(FRT_mean[roi],0)+sem(FRT_mean[roi],0)), \
                      color='b', alpha=.1)
    plt.axvline(0, color='k');
    plt.axvline(50, color='k', linewidth=1, alpha=0.3);
    plt.axvline(100, color='k',linewidth=1, alpha=0.3);
    plt.axvline(150, color='k', linewidth=1, alpha=0.3);
    plt.axvline(200, color='k', linewidth=1, alpha=0.3);
    plt.axhline(.2, color='k', linestyle='--', label='chance');
    mask = Fp_clust[roi] < 0.05
    mask = mask.values.reshape(300)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                   label="Cluster based permutation p<.05",
                   color="green")

    # plt.legend();
    mask = stats.ttest_1samp(FRT_mean[roi], .2)[1] < 0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                   label="Cluster based permutation p<.05",
                   color="yellow")    
    plt.title(f'FRUIT - {roi} Semantic Category Decoding');
    plt.savefig(f'FRUIT_{roi}.png', format='png');
    # plt.legend();
    plt.show();

for roi in kkROI:    
    sns.lineplot(x=times, y=np.mean(ODR_mean[roi],0))
    plt.fill_between(x=times, \
                      y1=(np.mean(ODR_mean[roi],0)-sem(ODR_mean[roi],0)), \
                      y2=(np.mean(ODR_mean[roi],0)+sem(ODR_mean[roi],0)), \
                      color='b', alpha=.1)
    plt.axvline(0, color='k');
    plt.axvline(50, color='k', linewidth=1, alpha=0.3);
    plt.axvline(100, color='k',linewidth=1, alpha=0.3);
    plt.axvline(150, color='k', linewidth=1, alpha=0.3);
    plt.axvline(200, color='k', linewidth=1, alpha=0.3);
    plt.axhline(.2, color='k', linestyle='--', label='chance');
    mask = Op_clust[roi] < 0.05
    mask = mask.values.reshape(300)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                   label="Cluster based permutation p<.05",
                   color="green")
    # plt.legend();
    mask = stats.ttest_1samp(ODR_mean[roi], .2)[1] < 0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                   label="Cluster based permutation p<.05",
                   color="yellow")    

    plt.title(f'ODOUR {roi} Semantic Category Decoding'); 
    plt.savefig(f'ODOUR_{roi}.png', format='png');
    plt.show();

SD_mean = dict.fromkeys(kkROI)
for roi in kkROI:  
    SD_mean[roi] = np.mean(np.array([MLK_mean[roi], 
                                  FRT_mean[roi], 
                                  ODR_mean[roi] ]),
                      axis=0)
    
    SD_mean[roi].shape = (18, 1, 300)


SDp_clust = pd.DataFrame(index=range(300),columns=kkROI)
for roi in kkROI:
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    SD_mean[roi].shape = (18, 1, 300)
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - 0.05, 18 - 1)
    SDt_clust, SDclusters, SDp_values, H0 = permutation_cluster_1samp_test(
        SD_mean[roi]-.2, n_jobs=1, threshold=threshold, adjacency=None,
        n_permutations='all')
    # Put the cluster data in a viewable format
    p_clust = np.ones((1,300))
    for cl, p in zip(SDclusters, SDp_values):
        p_clust[cl] = p
    SDp_clust[roi] = p_clust.T
    
for roi in kkROI:
    SD_mean[roi] = np.array(SD_mean[roi]).reshape((18,300))
    
    sns.lineplot(x=times, y=np.mean(SD_mean[roi], 0))
    plt.fill_between(x=times, \
                      y1=(np.mean(SD_mean[roi],0)-sem(SD_mean[roi],0)), \
                      y2=(np.mean(SD_mean[roi],0)+sem(SD_mean[roi],0)), \
                      color='b', alpha=.1)
    plt.axvline(0, color='k');
    plt.axvline(50, color='k', linewidth=1, alpha=0.3);
    plt.axvline(100, color='k',linewidth=1, alpha=0.3);
    plt.axvline(150, color='k', linewidth=1, alpha=0.3);
    plt.axvline(200, color='k', linewidth=1, alpha=0.3);
    plt.axhline(.2, color='k', linestyle='--', label='chance');
    # if len(times[np.where(SDp_clust[roi] < 0.05)]) > 0:
    #     for tp in times[np.where(SDp_clust[roi] < 0.05)]:
    #         plt.axvspan(tp,tp,
    #                 label="Cluster based permutation p<.05",
    #                 color="green", alpha=0.3)
    mask = SDp_clust[roi] < 0.05
    mask = mask.values.reshape(300)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                   label="Cluster based permutation p<.05",
                   color="green")
    plt.title('SD_avg Semantic Category Decoding'); 
    # plt.legend();
    mask = stats.ttest_1samp(SD_mean[roi], .2)[1] < 0.05
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.2,
                   label="Cluster based permutation p<.05",
                   color="yellow")
    plt.title('SD_avg Semantic Category Decoding')
    #plt.legend();
    plt.savefig(f'SD_avg_{roi}.png', format='png')
    plt.show();
    
times[np.where(stats.ttest_1samp(SD_mean[roi], .2)[1] < 0.05)]
