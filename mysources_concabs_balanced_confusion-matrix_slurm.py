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
scores['LD'] = dict.fromkeys(kkROI)
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


def trials2keep(data_class, stim2use):
    trials_to_drop = []
    for i,trial in enumerate(data_class.metadata['word']):
        if trial not in stim2use['word'].values:
            trials_to_drop.append(i)
    new_data = np.delete(data_class.data, trials_to_drop, axis=0)
    new_metadata = data_class.metadata.drop(index=trials_to_drop, axis=0)    
    new_trials = mydata(data_class.task,
                new_data,
                new_metadata,
                data_class.vertices)
    return new_trials
    

def individual_confusion_matrix(sub):

    print(f"Analysing subject {sub}")
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    # with open(f'//cbsu/data/imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
    #     output = pickle.load(f)
    
    with open(f"/imaging/hauk/users/fm02/Decoding_SDLD/re-epoched_data/mysourcespace_{sub}.P", "rb") as f:
       my_source = pickle.load(f)
    
    stim2use = pd.read_csv('/home/fm02/Decoding_SDLD/Stimuli/metadata_for_abstract.txt',
                           sep='\t')
    
    for task in my_source.keys():
        my_source[task] = trials2keep(my_source[task], stim2use)
    
    trials = dict.fromkeys(my_source.keys())
    
    for task in trials.keys():
        trials[task] = dict.fromkeys(kkROI)
        for roi in kkROI:
            trials[task][roi] = dict.fromkeys(kk2)
            for semcat in trials[task][roi]:
                trials[task][roi][semcat] = my_source[task].get_roi_semcat_tc(roi, semcat)

    trials_avg3 = dict.fromkeys(trials.keys())
    
    for task in trials_avg3.keys():
        trials_avg3[task] = dict.fromkeys(kk2)
        for semK in trials_avg3[task].keys():
            trials_avg3[task][semK] = dict.fromkeys(kkROI)
            for roi in trials_avg3[task][semK].keys():
                trials_avg3[task][semK][roi] = []
    

    for task in trials.keys():
        for roi in trials[task].keys():
        # drop trials until we reach a multiple of 3
        # (this is so that we always average 3 trials together)
            for semcat in trials[task][roi].keys():
                while len(trials[task][roi][semcat])%3 != 0:
                    trials[task][roi][semcat] = np.delete(trials[task][roi][semcat], 
                                                          len(trials[task][roi][semcat])-1, 0)
                # split data in groups of 3 trials
                new_tsk = np.vsplit(trials[task][roi][semcat], len(trials[task][roi][semcat])/3)
                new_trials = []
                # calculate average for each timepoint (axis=0) of the 3 trials
                for nt in new_tsk:
                    new_trials.append(np.mean(np.array(nt),0))
                # assign group to the corresponding task in the dict
                # each is 3D array n_trial*n_vertices*n_timepoints
                
                trials_avg3[task][semcat][roi] = np.array(new_trials)
                
    trials_avg3['sd'] = dict.fromkeys(kk2)
    for semK in trials_avg3['sd']:
        trials_avg3['sd'][semK] = dict.fromkeys(kkROI)
        for roi in trials_avg3['sd'][semK].keys():
            trials_avg3['sd'][semK][roi] = np.concatenate([trials_avg3['milk'][semK][roi],
                                                           trials_avg3['fruit'][semK][roi],
                                                           trials_avg3['odour'][semK][roi]])
    del(trials_avg3['milk'])
    del(trials_avg3['fruit'])       
    del(trials_avg3['odour'])
    
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearModel(LogisticRegression(C=1,
                                                       solver='liblinear',
                                                       max_iter=1000))) 
    time_decod = SlidingEstimator(clf, scoring='roc_auc_ovr')
    
 
    # just use subt instead of trials_semK if you want to have average of trials
    
    conf_mat = dict.fromkeys(['sd', 'LD'])
    for task in conf_mat.keys():
        conf_mat[task] = dict.fromkeys(kkROI)
        for roi in conf_mat[task].keys():
            conf_mat[task][roi] = []
            
 
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
            
            for i,trial in enumerate(y):
                if trial in ['neutral', 'emotional']:
                    y[i] = 'abstract'
                # elif trial in ['hand', 'hear', 'visual']:
                #     y[i] = 'concrete'
                    
            X, y = shuffle(X, y, random_state=0)
            
            scores[task][roi] = cross_val_multiscore(time_decod,
                                                     X, y, cv=5).mean(axis=0)
            
            predicted = []
    
            for i in range(0, 300):
                predicted.append(cross_val_predict(clf, X[:,:,i], y, cv=5))

            for i in range(0, 300):
                conf_mat[task][roi].append(confusion_matrix(y, predicted[i],
                                                            labels=['visual',
                                                                    'hand',
                                                                    'hear',
                                                                    'abstract'],
                                                            normalize='true'))
                
    with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/abs_balanced_scores_{sub}.P",
                  'wb') as outfile:
            pickle.dump(scores,outfile)            
    with open(f"/imaging/hauk/users/fm02/final_dTtT/individual_ROIs/SemCat/abs_balanced_confusion_matrix_{sub}.P",
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

