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

from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator,
                          get_coef)

kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

# initialise dictionaries and lists for storing scores
scores = {}

patterns = {}

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
    

def combined_scoresNpatterns(sub):

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
        trials[task] = dict.fromkeys(kk2)
        for semcat in trials[task]:
            trials[task][semcat] = my_source[task].get_semcat_tc(semcat)

    trials_avg3 = dict.fromkeys(trials.keys())
    
    for task in trials_avg3.keys():
        trials_avg3[task] = dict.fromkeys(kk2)
        for semK in trials_avg3[task].keys():
            trials_avg3[task][semK] = []
    

    for task in trials.keys():
        # drop trials until we reach a multiple of 3
        # (this is so that we always average 3 trials together)
        for semcat in trials[task].keys():
            while len(trials[task][semcat])%3 != 0:
                trials[task][semcat] = np.delete(trials[task][semcat], 
                                                      len(trials[task][semcat])-1, 0)
            # split data in groups of 3 trials
            new_tsk = np.vsplit(trials[task][semcat], len(trials[task][semcat])/3)
            new_trials = []
            # calculate average for each timepoint (axis=0) of the 3 trials
            for nt in new_tsk:
                new_trials.append(np.mean(np.array(nt),0))
            # assign group to the corresponding task in the dict
            # each is 3D array n_trial*n_vertices*n_timepoints
            
            trials_avg3[task][semcat] = np.array(new_trials)
    
    vertices = my_source['LD'].vertices
    
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                        SelectKBest(f_classif, k='all'),  # it's not the whole brain so I think we are fine using them all
                        LinearModel(LogisticRegression(C=1,
                                                       solver='liblinear',
                                                       max_iter=1000))) 
    time_decod = SlidingEstimator(clf, scoring='roc_auc_ovr')
    

    # just use subt instead of trials_semK if you want to have average of trials
    for task in trials_avg3.keys():

            X = []
            y = []
            for semK in kk2:
                X.append(trials_avg3[task][semK])
                y.extend([semK]*len(trials_avg3[task][semK]))
            X = np.concatenate(X)
            y = np.array(y)
            
            for i,trial in enumerate(y):
                if trial in ['neutral', 'emotional']:
                    y[i] = 'abstract'
                # elif trial in ['hand', 'hear', 'visual']:
                #     y[i] = 'concrete'
                    
            X, y = shuffle(X, y, random_state=0)
            
            scores[task] = cross_val_multiscore(time_decod,
                                                X, y, cv=5).mean(axis=0)
            
            time_decod.fit(X, y)

            # this already applies Haufe's trick
            # Retrieve patterns after inversing the z-score normalization step
            pattern = get_coef(time_decod, 'patterns_', inverse_transform=True)
            patterns[task] = dict.fromkeys(np.unique(y))
               
            for i, semK in enumerate(patterns[task].keys()):
                patterns[task][semK] = pd.DataFrame(pattern[:,i,:], index=vertices)             
            
    with open(f"/imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SemCat/abs_balanced_scores_{sub}.P",
                  'wb') as outfile:
            pickle.dump(scores,outfile)            
    with open(f"/imaging/hauk/users/fm02/final_dTtT/combined_ROIs/SemCat/abs_balanced_patterns_{sub}.P",
              'wb') as outfile:
        pickle.dump(patterns,outfile)
    print("Done.")

    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 18) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    combined_scoresNpatterns(ss)    

