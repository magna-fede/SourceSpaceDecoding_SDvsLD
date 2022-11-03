#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Federica.Magnabosco@mrc-cbu.cam.ac.uk
based on semnet/decoding by setare10
"""
import sys

import os
import mne
import time
import pickle
import numpy as np
import sn_config as C
from joblib import Parallel, delayed
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

# path to raw data
data_path = C.data_path

# subjects' directories
subjects = C.subjects
roi = SN_semantic_ROIs()
labels = C.rois_labels
lambda2 = C.lambda2_epoch
mri_sub = C.subjects_mri
categories = ['visual', 'hand', 'hear', 'neutral', 'emotional']
f_down_sampling = 250
re_epoched = '/imaging/hauk/users/fm02/Decoding_SDLD/re-epoched_data/'

    
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



#for i in np.arange(0  ,18):
def my_sources(i):
    print('***************************',i,'***************************')
    meg = subjects[i]
    sub_to = mri_sub[i][1:15]
    inv_fname_epoch_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_epoch_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'

    # morph labels from fsaverage to each subject
    morphed_labels = mne.morph_labels(roi, subject_to=sub_to,
                                      subject_from='fsaverage',
                                      subjects_dir=data_path)
    epochs = dict.fromkeys(['fruit', 'milk', 'odour', 'LD'])

    inv_op_SD = read_inverse_operator(inv_fname_epoch_SD)
    inv_op_LD = read_inverse_operator(inv_fname_epoch_LD)
    
    stc = dict.fromkeys(['fruit', 'milk', 'odour', 'LD'])
    for task in epochs.keys():
        # read,crop, resample epochs, and source estimate epochs
        epoch_fname = re_epoched+f"/{i}_block_{task}_epochs-andmeta-epo.fif"

        epochs[task] = mne.read_epochs(epoch_fname, preload=True)
        
        epochs[task] = epochs[task].resample(f_down_sampling)
        
        if task=='LD':
            stc[task] = apply_inverse_epochs(epochs[task], inv_op_LD, lambda2, method='MNE',
                                         pick_ori="normal", return_generator=False)
        elif task in ['milk', 'fruit', 'odour']:
            stc[task] = apply_inverse_epochs(epochs[task], inv_op_SD, lambda2, method='MNE',
                                         pick_ori="normal", return_generator=False)
        else:
            print('This should not happen')
    
    vertices = []
    for roi_idx in np.arange(0, 6):
        n_vertices, n_timepoints = stc[task][0].in_label(
            morphed_labels[roi_idx]).data.shape
        vertices.extend([labels[roi_idx]]*n_vertices)
    data = dict.fromkeys(['fruit', 'milk', 'odour', 'LD'])
    
    for task in stc.keys():   

        morphed_labels[roi_idx].subject = sub_to
        trials = []
        # creates output array of size (trials x vertices x timepoints)
        for n_trial, stc_trial in enumerate(stc[task]):
            x = []
            for roi_idx in np.arange(0, 6):
                x.extend(stc_trial.in_label(
                morphed_labels[roi_idx]).data)
            trials.append(np.array(x))
        
        data_block = np.stack(trials)
        data[task] = mydata(task, data_block, epochs[task].metadata, vertices)
        
    with open(f"/imaging/hauk/users/fm02/Decoding_SDLD/re-epoched_data/mysourcespace_{i}.P",
              'wb') as outfile:
        pickle.dump(data,outfile)

    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 18) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    my_sources(ss)    

