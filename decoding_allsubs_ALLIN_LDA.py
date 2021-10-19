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

# def get_patterns(model,X,y,filters_):
#     """Copied from mne patterns calculation from filters (aka coefficients)"""
#     inv_Y = 1.
#     X = X - X.mean(0, keepdims=True)
#     if y.ndim == 2 and y.shape[1] != 1:
#         y = y - y.mean(0, keepdims=True)
#         inv_Y = np.linalg.pinv(np.cov(y.T))
#     patterns = np.cov(X.T).dot(model.filters_.T.dot(inv_Y)).T
# patterns_ = np.cov(X.T).dot(self.filters_.T.dot(inv_Y)).T
#     return patterns
    
SDLD_scores = []
SDLD_coefficients = []
cat_scores = []

for sub in np.arange(0  ,18):
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    # with open(f'//cbsu/data/imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
    #     output = pickle.load(f)

    with open(f'/imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
        output = pickle.load(f)    
    
    
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
    
    for roi in trials_mlk_ignore[trials_mlk_ignore['trial']==0]['data']:
        vertices.append(roi.shape[0])
    
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
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearDiscriminantAnalysis(solver="svd",
                                                   store_covariance=True))
    time_decod = SlidingEstimator(clf, scoring='roc_auc')
    
    # Run cross-validated decoding analyses:
    scores_mlk = cross_val_multiscore(time_decod, X_mlk, y_mlk, cv=5)
    scores_frt = cross_val_multiscore(time_decod, X_frt, y_frt, cv=5)
    scores_odr = cross_val_multiscore(time_decod, X_odr, y_odr, cv=5)
    
    scores = pd.DataFrame(list(zip(scores_mlk, scores_frt, scores_odr)),
                          columns=['milk','fruit','odour'])
    SDLD_scores.append(scores)
     
    # HEY!
    # thanks mne.
    # https://github.com/mne-tools/mne-python/blob/maint/0.23/mne/decoding/base.py#L291-L355
    # line 93
    # patterns does already apply Haufe's trick
    time_decod.fit(X_mlk, y_mlk)
    coef_mlk = get_coef(time_decod, 'coef_', inverse_transform=True)
    
    # the covariance matrix as a parameter by LDA
    # not 100% sure this is actually getting that parameter
    cov_mlk = get_coef(time_decod, 'covariance_', inverse_transform=True)
    patterns_mlk = []
    for i in range(0,300):
        patterns.append(cov_mlk[:,:,i].dot(coef_mlk[:,i]).T)
    patterns_mlk = np.array(patterns).reshape(583,300)
    
    time_decod.fit(X_frt, y_frt)
    patterns_frt = get_coef(time_decod, 'patterns_', inverse_transform=True)
    
    time_decod.fit(X_odr, y_odr)
    patterns_odr = get_coef(time_decod, 'patterns_', inverse_transform=True)
    
    # this df has 4 columns:
        # one codes to which ROI the vertex belongs to
        # the other three refers to each task.

    df = pd.DataFrame(zip(ROI_vertices, patterns_mlk, patterns_frt, patterns_odr),columns=['ROI','milk', 'fruit', 'odour'])

    
    avg = []
    for i in range(len(df)):
        avg.append(np.mean([df['milk'][i],df['fruit'][i],df['odour'][i]],0))
    df['avg'] = avg

    
    SDLD_coefficients.append(df)
    
    # ## Change topic
    ########## NOT WORKING RIGHT NOW!
    # get back to this in the future
    
    # Let's know change topic and try to decode the word semantic category from the task. Arguably, if there is a difference in the information available when doing lexical decision task vs. when doing a semantic decision task, the accuracy should differ in the two tasks.
    
    
    # lds = []
    # y_lds = []
    # for cat in kk2:
    #     scat = trials_ld[trials_ld['category']==cat]
    #     ld_list = []
    #     for i in range(scat['trial'].max()):
    #         new = scat [(scat ['category']==cat) & (scat ['trial']==i)]
    #         ld_list.append(np.concatenate(new['data'].values))
    #     while len(ld_list)%4 != 0:
    #             ld_list.pop()
    #     new_ldList = list(chunks(ld_list, 4))
    #     new_lds = []
    #     for nt in new_ldList:
    #         new_lds.append(np.mean(nt,0)) 
    #     lds.extend(new_lds)    
    #     y_lds.extend([cat]*len(new_lds))    

    # mlks = []
    # y_mlks = []
    # for cat in kk2:
    #     scat = trials_mlk[trials_mlk['category']==cat]
    #     mlk_list = []
    #     for i in range(scat['trial'].max()):
    #         new = scat [(scat ['category']==cat) & (scat ['trial']==i)]
    #         mlk_list.append(np.concatenate(new['data'].values))
    #     while len(mlk_list)%4 != 0:
    #             mlk_list.pop()
    #     new_mlkList = list(chunks(mlk_list, 4))
    #     new_mlks = []
    #     for nt in new_mlkList:
    #         new_mlks.append(np.mean(nt,0)) 
    #     mlks.extend(new_mlks)    
    #     y_mlks.extend([cat]*len(new_mlks))    
    
    # frts = []
    # y_frts = []
    # for cat in kk2:
    #     scat = trials_frt[trials_frt['category']==cat]
    #     frt_list = []
    #     for i in range(scat['trial'].max()):
    #         new = scat [(scat ['category']==cat) & (scat ['trial']==i)]
    #         frt_list.append(np.concatenate(new['data'].values))
    #     while len(frt_list)%4 != 0:
    #             frt_list.pop()
    #     new_frtList = list(chunks(frt_list, 4))
    #     new_frts = []
    #     for nt in new_frtList:
    #         new_frts.append(np.mean(nt,0)) 
    #     frts.extend(new_frts)    
    #     y_frts.extend([cat]*len(new_frts))    
        
    # odrs = []
    # y_odrs = []
    # for cat in kk2:
    #     scat = trials_odr[trials_odr['category']==cat]
    #     odr_list = []
    #     for i in range(scat['trial'].max()):
    #         new = scat [(scat ['category']==cat) & (scat ['trial']==i)]
    #         odr_list.append(np.concatenate(new['data'].values))
    #     while len(odr_list)%4 != 0:
    #             odr_list.pop()
    #     new_odrList = list(chunks(odr_list, 4))
    #     new_odrs = []
    #     for nt in new_odrList:
    #         new_odrs.append(np.mean(nt,0)) 
    #     odrs.extend(new_odrs)    
    #     y_odrs.extend([cat]*len(new_odrs))    
    
 
    
    # X_lds = np.array(lds)
    
    # X_mlks = np.array(mlks)
    
    # X_frts = np.array(frts)
    
    # X_odrs = np.array(odrs)
    
    # X_allSD = np.array([lds + mlks + odrs]).reshape(len,583,300)
    # y_allSD = np.array([y_lds + y_mlks + y_odrs]).reshape(178)
    
    # X_lds, y_lds = shuffle(X_lds, y_lds)
    
    # X_mlks, y_mlks = shuffle(X_mlks, y_mlks)
    
    # X_frts, y_frts = shuffle(X_frts, y_frts)
    
    # X_odrs, y_odrs = shuffle(X_odrs, y_odrs)
    
    # X_allSD, y_allSD = shuffle(X_allSD, y_allSD)
    
    # prepare a series of classifier applied at each time sample
    # change the model because now it's not binary anymore
    # we need muticlass classifier
    # also need to change the scoring. for now let's do accuracy
    
    # clf = make_pipeline(StandardScaler(),  # z-score normalization
    #                     SelectKBest(f_classif, k='all'),  # select features for speed
    #                     LinearDiscriminantAnalysis(solver="svd"))
    # time_decod = SlidingEstimator(clf, scoring='accuracy')
    
    # scores_lds = cross_val_multiscore(time_decod, X_lds, y_lds, cv=5)
    # scores_mlks = cross_val_multiscore(time_decod, X_mlks, y_mlks, cv=5)
    # scores_frts = cross_val_multiscore(time_decod, X_frts, y_frts, cv=5)
    # scores_odrs = cross_val_multiscore(time_decod, X_odrs, y_odrs, cv=5)
    
    # scores_allSD = cross_val_multiscore(time_decod, X_allSD, y_allSD, cv=5)
    
    # df_cat = pd.DataFrame(list(zip(scores_lds,
    #                                scores_mlks,
    #                                scores_frts,
    #                                scores_odrs)),
    #                       columns =['scores_lds','scores_mlks',
    #                                'scores_frts','scores_odrs'])
    # cat_scores.append(df_cat)
    # Plot average decoding scores of 5 splits for each model
    # fig, ax = plt.subplots(1);
    # ax.plot(np.arange(-300,900,4), scores_lds.mean(0), label='average score');
    # ax.axhline(.2, color='k', linestyle='--', label='chance');
    # ax.axvline(0, color='k');
    # ax.set_title('Lexical Decision - Accuracy in predicting word category')
    # plt.legend();
    
    # fig, ax = plt.subplots(1);
    # ax.plot(np.arange(-300,900,4), scores_mlks.mean(0), label='average score');
    # ax.axhline(.2, color='k', linestyle='--', label='chance');
    # ax.axvline(0, color='k');
    # ax.set_title('Semantic Decision (milk) - Accuracy in predicting word category')
    # plt.legend();
    
    # fig, ax = plt.subplots(1);
    # ax.plot(np.arange(-300,900,4), scores_frts.mean(0), label='average score');
    # ax.axhline(.2, color='k', linestyle='--', label='chance');
    # ax.axvline(0, color='k');
    # ax.set_title('Semantic Decision (fruit) - Accuracy in predicting word category')
    # plt.legend();
    
    # fig, ax = plt.subplots(1);
    # ax.plot(np.arange(-300,900,4), scores_odrs.mean(0), label='average score');
    # ax.axhline(.2, color='k', linestyle='--', label='chance');
    # ax.axvline(0, color='k');
    # ax.set_title('Semantic Decision (odours) - Accuracy in predicting word category')
    # plt.legend();
    
    # fig, ax = plt.subplots(1);
    # ax.plot(np.arange(-300,900,4), scores_allSD.mean(0), label='average score');
    # ax.axhline(.2, color='k', linestyle='--', label='chance');
    # ax.axvline(0, color='k');
    # ax.set_title('All Semantic Decisions - Accuracy in predicting word category')
    # plt.legend();




# df_to_export = pd.DataFrame(SDLD_scores)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0923_SDLD_scores.P",
#           'wb') as outfile:
#     pickle.dump(df_to_export,outfile)
    
# df_to_export = pd.DataFrame(SDLD_coefficients)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0923_SDLD_coefficients.P",
#           'wb') as outfile:
#     pickle.dump(df_to_export,outfile)
    
    

 
