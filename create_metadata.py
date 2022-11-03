# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:26:07 2022

@author: fm02
epoching is adapted from setareh10/semnet-project/sn_epoching.py
"""

import sys

import numpy as np
import pandas as pd
import os
import mne
from importlib import reload

path = "/home/fm02/Decoding_SDLD/SourceSpaceDecoding_SDvsLD"
os.chdir(path)
import sn_config as C
reload(C)

# visual = trignum=1
# auditory = trignum=2
# hand = trignum=3
# neutral = trignum=4
# emotional = trignum=5

def epochs_metadata(sub):
    
    path = "/imaging/hauk/users/rf02/Setareh/wordlist_ling"
    os.chdir(path)
    
    hand = pd.read_csv("handall.txt", sep="\t")
    hear = pd.read_csv("hearall.txt", sep="\t")
    visual = pd.read_csv("visall.txt", sep="\t")
    neutral = pd.read_csv("word_nabs.txt", sep="\t")
    emotional = pd.read_csv("word_eabs.txt", sep="\t")
    pseudo = pd.read_csv('pseudo_word.txt', sep='\t')
    
    path = "/home/fm02/Decoding_SDLD/Stimuli"
    os.chdir(path)
    
    words = pd.read_csv("wordlist_final_SQ.txt", sep='\t')
    pseudo_words = pd.read_csv('wordlist_final_LD_pw.txt', sep="\t")
    
    visual = visual[visual['word'].isin(words['word'].values)]
    hear = hear[hear['word'].isin(words['word'].values)]
    hand = hand[hand['word'].isin(words['word'].values)]
    neutral = neutral[neutral['word'].isin(words['word'].values)]
    emotional = emotional[emotional['word'].isin(words['word'].values)]
    
    pseudo = pseudo[pseudo['word'].isin(pseudo_words['word'].values)]
    pseudo = pseudo.rename(columns={'UN2_F': 'Bigram Frequency',
                                'UN3_F': 'Trigram Frequency',
                                'LEN': 'Number of letters',
                                'FREQ': 'Frequency',
                                'Orth': 'Orthographic Neighbourhood Size'})
    pseudo['cat'] = 'pseudowords'
    
    visual['cat'] = 'visual'
    hear['cat'] = 'hear'
    hand['cat'] = 'hand'
    neutral['cat'] = 'neutral'
    emotional['cat'] = 'emotional'
    
    description = pd.concat([visual, hear, hand, neutral, emotional], ignore_index=True)
    
    metadata = pd.merge(description, words[['word', 'trignum', 'trigindiv']], on='word')
    
    print("participant: ", sub)
    meg = C.subjects[sub]

    print(f"That's the data: {meg}")    

    for task in ["fruit", "odour", "milk", "LD"]:
        event_id = {'visual': 1, 
                    'hear': 2, 
                    'hand': 3, 
                    'neutral': 4,
                    'emotional': 5}
            
        filename = C.data_path + meg + \
            f"block_{task}_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif"
        print(f"Creating epoch from; {filename}")
        raw = mne.io.Raw(filename, preload=True)
        picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=False, 
                               stim=False)
        events = mne.find_events(raw, stim_channel="STI101",
                                 min_duration=0.002, shortest_event=1)
        # Considering the device(!) delay
        events[:, 0] += int(np.round(raw.info["sfreq"] * C.stim_delay))
        
#        remove = list()
        # Finding events with false responses
        for e in range(events.shape[0] - 2):
            if task == "LD":
                if events[e, 2] in C.category_code and events[e+2, 2] != 16384:                    
#                    remove.append(tuple([events[e, 2],events[e+1, 2]]))
                    events[e, 2] = 7777
                elif events[e, 2] in np.array([6, 7, 9]) \
                        and events[e+2, 2] != 4096:                    
#                    remove.append(tuple([events[e, 2],events[e+1, 2]]))
                    events[e, 2] = 8888
            else:
                if events[e, 2] in C.category_code and events[e+2, 2] > 100:
 #                   remove.append(tuple([events[e, 2],events[e+1, 2]]))
                    events[e, 2] = 7777
                elif events[e, 2] == 8 and events[e+2, 2] < 100:
  #                  remove.append(tuple([events[e, 2],events[e+1, 2]]))
                    events[e, 2] = 8888
        # Extracting epochs from a raw instance
        epochs = mne.Epochs(raw, events, event_id, C.tmin, C.tmax, picks=picks,
                            proj=True, baseline=(C.tmin, 0), reject=C.reject)
        
        # for trial in remove:
        #     idx_toremove = meta_trials[((meta_trials['trignum']==trial[0]) & \
        #                                 (meta_trials['trigindiv']==trial[0]))].index
        #     meta_trials = meta_trials.drop(idx_toremove, axis=0)
        
        meta_trials = pd.DataFrame(columns=metadata.columns)
        
        empty_trial = pd.Series(data='na', index=metadata.columns)
        # i = list()
        # for e in range(len(events)-1):
        #     if events[e, 2] in C.category_code:
        #         i.append(tuple([events[e,2], events[e+1,2]]))
        
        # word = []                
        # for couple in i:
        #     word.append(metadata['word'][(metadata['trignum']==couple[0]) & \
        #                           (metadata['trigindiv']==couple[1])].item())
        for e in range(len(events)-1):
                this_trial = metadata[(metadata['trignum']==events[e, 2]) & \
                                      (metadata['trigindiv']==events[e+1, 2])]
                meta_trials = pd.concat([meta_trials, this_trial], axis=0, ignore_index=True)
        meta_trials
        epochs.metadata = meta_trials       
        epochs = epochs.drop_bad()
        
        
        epochs.metadata.to_csv(f"/imaging/hauk/users/fm02/Decoding_SDLD/Stimuli/data_{sub}_{task}.csv",
                            index=False)        
                
        # checking for the existence of desired directory to save the data
        output = f"/imaging/hauk/users/fm02/Decoding_SDLD/re-epoched_data/{sub}_block_{task}_epochs-andmeta-epo.fif"
        # saving epochs
        epochs.save(output, overwrite=True)


# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 18) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    epochs_metadata(ss)
    