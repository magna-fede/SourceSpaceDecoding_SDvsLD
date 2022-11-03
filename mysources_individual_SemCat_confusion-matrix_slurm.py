#!/usr/bin/env python
# coding: utf-8


import sys

import numpy as np
import pandas as pd
import pickle


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator)

kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

# initialise dictionaries and lists for storing scores
scores = {}
scores['ld'] = dict.fromkeys(kkROI)
scores['sd'] = dict.fromkeys(kkROI)

class mydata:
    """Organise the data so everything is easily accessible"""
    def __init__(self, task, data, metadata, vertices):
        self.task = task
        self.data = np.array(data)
        self.metadata = metadata.reset_index(drop=True)
        self.vertices = np.array(vertices)

    def get_roi_tc(self, roi):
        return self.data[:,self.vertices==roi,:]
    
    def get_semcat_tc(self, semcat):
        return self.data[self.metadata['cat']==semcat]        
    
    def get_roi_semcat_tc(self, roi, semcat):
        temp = self.data[self.metadata['cat']==semcat]
        return temp[:,self.vertices==roi,:]

def divideNaverage_per_LEN(sources):
    """input is my_source[task]"""
    counts_letters = sources.metadata['LEN'].value_counts()
    divided = dict.fromkeys(counts_letters.index.values)
    
    for n_letters in divided.keys():
        divided[n_letters] = sources.data[sources.metadata['LEN']==n_letters]

        while divided[n_letters].shape[0]%3 != 0:
            divided[n_letters] = divided[n_letters][:-1, :]
    # split data in groups of 3 trials
        new_tsk = np.vsplit(divided[n_letters], len( divided[n_letters])/3)
        new_trials = []
    # calculate average for each timepoint (axis=0) of the 3 trials
        for nt in new_tsk:
            new_trials.append(np.mean(np.array(nt),0))
    # assign group to the corresponding task in the dict
    # each is 3D array n_trial*n_vertices*n_timepoints
        divided[n_letters] = np.array(new_trials)    
    return divided

def divide_per_ROI_ldsd(sources, trials):
    tc_per_roi = dict.fromkeys(['ld', 'sd'])
    y = dict.fromkeys(['ld', 'sd'])
    for task in tc_per_roi.keys():
        tc_per_roi[task] = dict.fromkeys(kkROI)
        y[task] = []
        for roi in tc_per_roi[task]:
            tc_per_roi[task][roi] = []
    for task in trials.keys():
        if task=='LD':
            for roi in kkROI:
                temp = []
                for n_len in trials[task].keys():
                    mask = sources[task].vertices==roi
                    temp.append(trials[task][n_len][:,np.where(mask),:].reshape(len(trials[task][n_len]),
                                                                    len(np.where(mask)[0]),
                                                                    trials[task][n_len].shape[-1]))
                    # create y just from one ROI (y doesn't change from ROIs)
                    if roi == 'lATL':
                        y['ld'].extend([n_len]*len(trials[task][n_len]))
                tc_per_roi['ld'][roi] = np.vstack(temp)
                    
        else:            
            for roi in kkROI:
                temp = []
                for n_len in trials[task].keys():
                    mask = sources[task].vertices==roi
                    temp.append(trials[task][n_len][:,np.where(mask),:].reshape(len(trials[task][n_len]),
                                                                len(np.where(mask)[0]),
                                                                trials[task][n_len].shape[-1]))
                    if roi == 'lATL':
                        y['sd'].extend([n_len]*len(trials[task][n_len]))
                tc_per_roi['sd'][roi].append(np.vstack(temp))
    
    for roi in tc_per_roi['sd'].keys():
        tc_per_roi['sd'][roi] = np.vstack(tc_per_roi['sd'][roi])
    for task in y.keys():
        y[task] = np.array(y[task])
        
    return tc_per_roi, y

def individual_confusion_matrix(sub):

    print(f"Analysing subject {sub}")
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    # with open(f'//cbsu/data/imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
    #     output = pickle.load(f)
    
    with open(f"/imaging/hauk/users/fm02/Decoding_SDLD/re-epoched_data/mysourcespace_{sub}.P", "rb") as f:
       my_source = pickle.load(f)
       
    trials = dict.fromkeys(my_source.keys())
    for task in trials.keys():
        trials[task] = divideNaverage_per_LEN(my_source[task])
    
    X, y = divide_per_ROI_ldsd(my_source, trials)
    # try not averaging because not enough trials otherwise
    
    ### now let's average 3 trials together
    # initialise dict

    # We create and run the model. We expect the model to perform at chance before the presentation of the stimuli (no ROI should be sensitive to task/semantics demands before the presentation of a word).
        
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearModel(LogisticRegression(C=1,
                                                       solver='lbfgs',
                                                       max_iter=1000))) 
    time_decod = SlidingEstimator(clf, scoring='roc_auc_ovr')
    
 
    # just use subt instead of trials_semK if you want to have average of trials
    
    conf_mat = dict.fromkeys(['sd', 'ld'])
    for task in conf_mat.keys():
        conf_mat[task] = dict.fromkeys(kkROI)
        for roi in conf_mat[task].keys():
            conf_mat[task][roi] = []
 
    # just use subt instead of trials_semK if you want to have average of trials

    for task in ['sd', 'ld']:
        for roi in kkROI:
            print(f"Fitting for {task, roi}")
            X_now = X[task][roi]
            y_now = y[task]
        
            X_now, y_now = shuffle(X_now, y_now, random_state=0)
                    
            predicted = []
            
            for i in range(0, 300):
                predicted.append(cross_val_predict(clf, X_now[:,:,i], y_now, cv=5))

            for i in range(0, 300):
                conf_mat[task][roi].append(confusion_matrix(y_now, predicted[i],
                                                            labels=np.unique(y_now),
                                                            normalize='true'))
                
    with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/length_confusion_matrix_{sub}.P",
              'wb') as outfile:
        pickle.dump(conf_mat,outfile)
    print("Done.")

    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 18) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    individual_confusion_matrix(ss)    

