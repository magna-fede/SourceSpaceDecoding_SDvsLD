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


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# mne is a library for analysis of MEG/EEG brain data

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
    rms_example = np.sqrt(np.mean(example**2))
    
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
        
SDLD_scores = []
SDLD_coefficients = []
LDvsSD_ranks = []

for sub in np.arange(0, 18):
    print(f"Analysing participant: {sub}")
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    # with open(f'//imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
    #     output = pickle.load(f)
    
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
    # The above shapes refletc (n_trials * n_vertices * timpoints), as we can see the number of trials is similar, and we have the same number of vertices and timepoints (aka our brain signal) for each trial.
    
    # We now want to group each vertex with the ROI they belong to (as we lost this information during the manipulation of the data).
    
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
    
    X_mlk = np.concatenate([mlks , lds])
    y_mlk = np.array(['milk']*len(mlks) + ['LD']*len(lds))
    
    X_frt = np.concatenate([frts , lds])
    y_frt = np.array(['fruit']*len(frts) + ['LD']*len(lds))
    
    X_odr = np.concatenate([odrs , lds])
    y_odr = np.array(['odour']*len(odrs) + ['LD']*len(lds))
    
    
    X_mlk, y_mlk = shuffle(X_mlk, y_mlk, random_state=0)
    X_frt, y_frt = shuffle(X_frt, y_frt, random_state=0)
    X_odr, y_odr = shuffle(X_odr, y_odr, random_state=0)
    
    # We create and run the model. We expect the model to perform at chance before the presentation of the stimuli (no ROI should be sensitive to task/semantics demands before the presentation of a word).
    
    # prepare a series of classifier applied at each time sample
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # select features for speed
                        LinearModel(LogisticRegression(C=1, solver='liblinear')))
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    
    # # Run cross-validated decoding analyses:
    # scores_mlk = cross_val_multiscore(time_decod, X_mlk, y_mlk, cv=5)
    # scores_frt = cross_val_multiscore(time_decod, X_frt, y_frt, cv=5)
    # scores_odr = cross_val_multiscore(time_decod, X_odr, y_odr, cv=5)
    
    # scores = pd.DataFrame(list(zip(scores_mlk, scores_frt, scores_odr)),
    #                       columns=['milk','fruit','odour'])
    # SDLD_scores.append(scores)
    
    # # HEY!
    # # thanks mne.
    # # https://github.com/mne-tools/mne-python/blob/maint/0.23/mne/decoding/base.py#L291-L355
    # # line 93
    # # patterns does already apply Haufe's trick
    time_decod.fit(X_mlk, y_mlk)
    patterns_mlk = get_coef(time_decod, 'patterns_', inverse_transform=True)
    
    time_decod.fit(X_frt, y_frt)
    patterns_frt = get_coef(time_decod, 'patterns_', inverse_transform=True)
    
    time_decod.fit(X_odr, y_odr)
    patterns_odr = get_coef(time_decod, 'patterns_', inverse_transform=True)
    
    # # this df has 4 columns:
    #     # one codes to which ROI the vertex belongs to
    #     # the other three refers to each task.

    # df = pd.DataFrame(zip(ROI_vertices, patterns_mlk, patterns_frt, patterns_odr),columns=['ROI','milk', 'fruit', 'odour'])

    
    # avg = []
    # for i in range(len(df)):
    #     avg.append(np.mean([df['milk'][i],df['fruit'][i],df['odour'][i]],0))
    # df['avg'] = avg

    
    # SDLD_coefficients.append(df)
    
    
    mlk_selected = pd.DataFrame(index=kkROI,
                                columns=range(300))
    frt_selected = pd.DataFrame(index=kkROI,
                                columns=range(300))
    odr_selected = pd.DataFrame(index=kkROI,
                                columns=range(300))
    

    # in this loop the rank is calculated based on number of coefficient for each ROI 
    # (e.g., sum of coefs positions)
    
    mlk_ranks = pd.DataFrame(index=kkROI,
                                columns=range(300))
    
    for i in range(300):
        coefscores_milk_ps = pd.DataFrame(zip(ROI_vertices,
                          SelectKBest(k='all').fit(X_mlk[:,:,i],
                                                    y_mlk).pvalues_))
        df = pd.DataFrame(zip(ROI_vertices, patterns_mlk[:,i]))

        df_less05 = df.loc[coefscores_milk_ps[1]<.05]
        
        df_avg = df_less05.groupby(by=0).apply(rms)
        
        if len(df_avg)==0:
            mlk_selected[i] = np.nan*6
        else:
            mlk_selected[i] = df_avg        
        
        coefscores_fruit_ps = pd.DataFrame(zip(ROI_vertices,
                          SelectKBest(k='all').fit(X_frt[:,:,i],
                                                    y_frt).pvalues_))
        df = pd.DataFrame(zip(ROI_vertices, patterns_frt[:,i]))

        df_less05 = df.loc[coefscores_fruit_ps[1]<.05]
        
        df_avg = df_less05.groupby(by=0).apply(rms)
        
        if len(df_avg)==0:
            frt_selected[i] = np.nan*6
        else:
            frt_selected[i] = df_avg


        coefscores_odour_ps = pd.DataFrame(zip(ROI_vertices,
                          SelectKBest(k='all').fit(X_odr[:,:,i],
                                                    y_odr).pvalues_))
        df = pd.DataFrame(zip(ROI_vertices, patterns_odr[:,i]))

        df_less05 = df.loc[coefscores_odour_ps[1]<.05]
        
        df_avg = df_less05.groupby(by=0).apply(rms)
        
        if len(df_avg)==0:
            odr_selected[i] = np.nan*6
        else:
            odr_selected[i] = df_avg        
    
    arrays = [
        ["milk", "milk", "milk", "milk", "milk", "milk",
         "fruit", "fruit", "fruit", "fruit", "fruit", "fruit",
         "odour", "odour", "odour", "odour", "odour", "odour"],
        kkROI*3]
    
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["task", "ROI"])
    
    s = pd.DataFrame(np.vstack([mlk_selected,frt_selected,odr_selected]),index=index,columns=range(300))

    avg_coef = s.groupby(by=s.index.names[1]).mean()

    def process_index(k):
        return tuple(["avg",k])

    avg_coef_index = pd.MultiIndex.from_tuples([process_index(k)
                                          for k,v in avg_coef.iterrows()])
    new_avg_coef = avg_coef.set_index(avg_coef_index)
    
    s = s.append(new_avg_coef)
    
    LDvsSD_ranks.append(s)

    # in this loop the rank is calculated based on position 
    # (e.g., sum of coefs positions)
    # for i in range(300):
    #     coefscores_mf = pd.DataFrame(zip(ROI_vertices,
    #                               SelectKBest(k='all').fit(X_mlk[:,:,i],
    #                                                         y_mlk).scores_))

    #     coefscores_mf = coefscores_mf.sort_values(by=1, ascending=False).reset_index()
    #     coefscores_mf = coefscores_mf.tail(len(coefscores_mf)//10)
    
    #     coef_rank = pd.Series(index=kkROI)
    
    #     for roi in kkROI:
    #         coef_rank[roi] = coefscores_mf[coefscores_mf[0]==roi].index.values.sum()
    #     coef_rank = coef_rank.sort_values(ascending=False)
    #     coef_rank_app = pd.Series(np.arange(1,7), index=coef_rank.index.values )
        
    #     mlk_ranks[i] = coef_rank_app
    
    
    #     coefscores_fo = pd.DataFrame(zip(ROI_vertices,
    #                               SelectKBest(k='all').fit(X_frt[:,:,i],
    #                                                         y_frt).scores_))
    #     coefscores_fo = coefscores_fo.sort_values(by=1, ascending=True).reset_index()
    #     # get the best 10%
    #     # note, this is different from selectPercentile, because we are fitting
    #     # the data considering all time points, and not just the 10% best features
    #     coefscores_fo = coefscores_fo.tail(len(coefscores_fo)//10)
    
    #     coef_rank = pd.Series(index=kkROI)
    
    #     for roi in kkROI:
    #         coef_rank[roi] = coefscores_fo[coefscores_fo[0]==roi].index.values.sum()
    #     coef_rank = coef_rank.sort_values(ascending=False)
    #     coef_rank_app = pd.Series(np.arange(1,7), index=coef_rank.index.values )
        
    #     frt_ranks[i] = coef_rank_app
    
    #     coefscores_om = pd.DataFrame(zip(ROI_vertices,
    #                               SelectKBest(k='all').fit(X_odr[:,:,i],
    #                                                         y_odr).scores_))
    #     coefscores_om = coefscores_om.sort_values(by=1, ascending=True).reset_index()
    #     # get the best 10%
    #     # note, this is different from selectPercentile, because we are fitting
    #     # the data considering all time points, and not just the 10% best features
    #     coefscores_om = coefscores_om.tail(len(coefscores_om)//10)
    
    #     coef_rank = pd.Series(index=kkROI)
    
    #     for roi in kkROI:
    #         coef_rank[roi] = coefscores_om[coefscores_om[0]==roi].index.values.sum()
    #     coef_rank = coef_rank.sort_values(ascending=False)
    #     coef_rank_app = pd.Series(np.arange(1,7), index=coef_rank.index.values )
        
    #     odr_ranks[i] = coef_rank_app
    
    # all_ranks = pd.concat([mlk_ranks,frt_ranks,odr_ranks])
    # avg_ranks = all_ranks.groupby(by=all_ranks.index).mean()  
    
    # LDvsSD_ranks.append(avg_ranks)


import seaborn as sns

import matplotlib.pyplot as plt

big_ranks = pd.concat(LDvsSD_ranks)
avg_big_ranks = big_ranks.groupby(by=big_ranks.index).mean()

ax = sns.heatmap(avg_big_ranks, cmap="YlGnBu", xticklabels=False)
plt.axvline(75, color='k');
plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(150, color='k',linewidth=1, alpha=0.3);
plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(225, color='k', linewidth=1, alpha=0.3);
plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])

plt.title("LDvsaverage rank (low is better)");

plt.show()

for i,df in enumerate(LDvsSD_ranks):
    ax = sns.heatmap(df, cmap="YlGnBu")
    plt.axvline(75, color='k');
    plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
    plt.axvline(150, color='k',linewidth=1, alpha=0.3);
    plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
    plt.axvline(225, color='k', linewidth=1, alpha=0.3);
    plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])

    plt.title(f"ranks participant {i}")
    plt.show()

# df_to_export = pd.DataFrame(SDLD_scores)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0923_SDLD_scores.P",
#           'wb') as outfile:
#     pickle.dump(df_to_export,outfile)
    
# df_to_export = pd.DataFrame(SDLD_coefficients)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0923_SDLD_coefficients.P",
#           'wb') as outfile:
#     pickle.dump(df_to_export,outfile)
    
participants = {}
for i,df in enumerate(LDvsSD_ranks):
    participants[i] = df

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1109_LDvsSD_avgless05.P", 'wb') as outfile:
    pickle.dump(participants,outfile)   

 
