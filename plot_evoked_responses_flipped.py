# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:40:11 2021

@author: fm02
"""


# plot average activity for each vertex, for each participant
import numpy as np
import pandas as pd
import pickle
import os

import mne
from mne.minimum_norm import read_inverse_operator

import warnings
warnings.filterwarnings('ignore')

path = '/imaging/hauk/users/fm02/Decoding_SDLD'
os.chdir(path)

from Setareh.SN_semantic_ROIs import SN_semantic_ROIs

# list subjects directory
from Setareh.SN_semantic_ROIs import subjects

# subjects' MRI directories
from Setareh.SN_semantic_ROIs import subjects_mri

data_path = '/imaging/hauk/users/sr05/SemNet/SemNetData/'
main_path = '/imaging/hauk/users/rf02/Semnet/'

roi_sem = SN_semantic_ROIs()
    
list_avg_scores = []
list_mlk_scores = []
list_frt_scores = []
list_odr_scores = []

evoked_participants = []
flipped_participants =[]

for sub in np.arange(0, 18):
    # import the dataset containing 120 categories (6 ROIs * 4 tasks *5 categories)
    # each key contains an array with size (number of trials * number of vertices * time points)
    # with open(f'//cbsu/data/Imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
    #     output = pickle.load(f)
 
    with open(f'/imaging/hauk/users/fm02/dataSDLD/activities_sub_{sub}.json', 'rb') as f:
        output = pickle.load(f)
    
    print(f'Analysing participant {sub}')
    kk = list(output.keys())
    
    # As we can observe, the words belong to different semantic categories (kk2).
    # In this project we will ignore it, and consider them just as different trials
    # belonging either to the LD or milk task. 
    
    kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
    kkROI = ['lATL', 'rATL', 'PTC', 'IFG', 'AG', 'PVA']
    
    
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
    for task in avg.keys():
        for roi in kkROI:
        # transform in 3D-matrix
        # which has (n_vertices*timepoints*n_trials)
            avg_roi = np.dstack(all_trials[task][roi])
        # and average over trials (3rd dimension) for each vertex at each timepoint
            avg_roi = np.mean(avg_roi,2)
            avg[task][roi] = avg_roi
    
    #####
    # 2 #
    #####
    # this loops averages across vertices (maintaining info from each trial)
    # use when want to have mean activity in each roi, across vertices, for each timepoint

    # for task in avg.keys():
    #     for roi in kkROI:
    #         ###############
    #         ### AVERAGE ###
    #         ###############
    #     # # transform in 3D-matrix
    #     # # which has (n_trials*n_vertices*timepoints*)
    #     #     avg_roi = np.stack(all_trials[task][roi])
    #     # # and average over vertices (2nd dimension) for each trial at each timepoint
    #     #     avg_roi = np.mean(avg_roi, 1)
    #     #     avg[task][roi].append(avg_roi)
    #         ###########
    #         ### RMS ###
    #         ###########
    #         avg_roi = np.stack(all_trials[task][roi])
    #         avg_roi = np.apply_along_axis(rms, 1, np.stack(all_trials[task][roi]))
    #         # and average over vertices (2nd dimension) for each trial at each timepoint
    #         avg[task][roi].append(avg_roi)
    # evoked_participants.append(avg)
    
    #####
    # 3 #
    #####
    
    morphed_labels = mne.morph_labels(roi_sem, subjects_mri[sub][1:15],
                                      subject_from='fsaverage',
                                      subjects_dir=data_path)

    fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
    
    inverse_operator_LD = read_inverse_operator('/imaging/hauk/users/sr05/SemNet/SemNetData/' + \
                                             subjects[sub] + \
                                             'InvOp_LD_EMEG-inv.fif')

    inverse_operator_SD = read_inverse_operator('/imaging/hauk/users/sr05/SemNet/SemNetData/' + \
                                             subjects[sub] + \
                                             'InvOp_SD_EMEG-inv.fif')        

    flip_LD = {}
    flip_SD = {}

    for i, area in enumerate(kkROI):
        
        flip_LD[area] = mne.label_sign_flip(morphed_labels[i], inverse_operator_LD['src'])        
        flip_SD[area] = mne.label_sign_flip(morphed_labels[i], inverse_operator_SD['src'])  

    label_mean_flip = {}
    label_mean_flip['LD'] = {}
    label_mean_flip['MLK'] = {}
    label_mean_flip['FRT'] = {}
    label_mean_flip['ODR'] = {}
    
    for i,area in enumerate(kkROI):
        label_mean_flip['LD'][area] = np.mean(flip_LD[area][:, np.newaxis] * avg['LD'][area], axis=0)
        for task in ['MLK', 'FRT', 'ODR']:
            label_mean_flip[task][area] = np.mean(flip_SD[area][:, np.newaxis] * avg[task][area], axis=0)
    
    evoked_participants.append(avg)
    flipped_participants.append(label_mean_flip)
    
# df_to_export = pd.DataFrame(evoked_participants)
# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/evoked_responses/0722_evoked_acrossROI.P", 'wb') as outfile:
#     pickle.dump(df_to_export,outfile)

df_to_export = pd.DataFrame(evoked_participants)
with open("/imaging/hauk/users/fm02/first_output/evoked_responses/0804_evoked_AVGacrossROI.P", 'wb') as outfile:
    pickle.dump(df_to_export,outfile)

df_to_export2 = pd.DataFrame(flipped_participants)
with open("/imaging/hauk/users/fm02/first_output/evoked_responses/0804_evoked_flipped.P", 'wb') as outfile:
    pickle.dump(df_to_export2,outfile)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
