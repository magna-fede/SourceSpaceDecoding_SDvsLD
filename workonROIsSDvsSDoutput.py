
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:54:00 2021

@author: magna.fede
"""
# import relevant stuff

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load scores result from previous script

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_ROIs_avg_scores.P", 'rb') as f:
     avg_scores = pickle.load(f)

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_ROIs_mlkfrt_scores.P", 'rb') as f:
     mlk_scores = pickle.load(f)
     
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_ROIs_frtodr_scores.P", 'rb') as f:
     frt_scores = pickle.load(f)
     
with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_ROIs_odrmlk_scores.P", 'rb') as f:
     odr_scores = pickle.load(f)
     
kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

# intilaise participant's average for each classifier
participants_mlk = pd.DataFrame(index=np.arange(0,18),columns=kkROI)
participants_frt = pd.DataFrame(index=np.arange(0,18),columns=kkROI)
participants_odr = pd.DataFrame(index=np.arange(0,18),columns=kkROI)

# get average scores for each ROI for each participant
# remember for each participant we have 5 cv, want average

# loop over ROIs
for roi in kkROI:
    # index is the same for all the tasks
    for ind in mlk_scores.index:
        participants_mlk[roi][ind] = np.mean(mlk_scores[roi][ind],0)
        participants_frt[roi][ind] = np.mean(frt_scores[roi][ind],0)
        participants_odr[roi][ind] = np.mean(odr_scores[roi][ind],0)


avg_all = pd.DataFrame(index=range(300),columns=kkROI)

for roi in kkROI:  
    sns.lineplot(x=np.arange(-300,900,4), y=np.mean(participants_mlk[roi],0))
    # please note that np.mean(np.vstack(participants_mlk[roi]),0) 
    # is equal to np.mean(participants_mlk[roi],0)
    plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(participants_mlk[roi]),0)-np.std(np.array(participants_mlk[roi]),0)), \
                 y2=(np.mean(np.array(participants_mlk[roi]),0)+np.std(np.array(participants_mlk[roi]),0)), \
                 color='b', alpha=.1)
    avg_all[roi] = np.mean(participants_mlk[roi],0)
    plt.axvline(0, color='k');
    plt.axhline(.5, color='k', linestyle='--', label='chance');
    plt.title(roi)
    plt.show();
fig, ax = plt.subplots(1);
ax.plot(np.arange(-300,900,4),avg_all);
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.title('Decoding MILKvsLD accuracy of each ROI separately')
plt.legend(kkROI);
plt.show();

avg_all = pd.DataFrame(index=range(300),columns=kkROI)

for roi in kkROI:  
    sns.lineplot(x=np.arange(-300,900,4), y=np.mean(participants_frt[roi],0))
    # please note that np.mean(np.vstack(participants_frt[roi]),0) 
    # is equal to np.mean(participants_frt[roi],0)
    plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(participants_frt[roi]),0)-np.std(np.array(participants_frt[roi]),0)), \
                 y2=(np.mean(np.array(participants_frt[roi]),0)+np.std(np.array(participants_frt[roi]),0)), \
                 color='b', alpha=.1)
    avg_all[roi] = np.mean(participants_frt[roi],0)
    plt.axvline(0, color='k');
    plt.axhline(.5, color='k', linestyle='--', label='chance');
    plt.title(roi)
    plt.show();
fig, ax = plt.subplots(1);
ax.plot(np.arange(-300,900,4),avg_all);
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.title('Decoding FRUITvsLD accuracy of each ROI separately')
plt.legend(kkROI);
plt.show();

avg_all = pd.DataFrame(index=range(300),columns=kkROI)

for roi in kkROI:  
    sns.lineplot(x=np.arange(-300,900,4), y=np.mean(participants_odr[roi],0))
    # please note that np.mean(np.vstack(participants_odr[roi]),0) 
    # is equal to np.mean(participants_odr[roi],0)
    plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(participants_odr[roi]),0)-np.std(np.array(participants_odr[roi]),0)), \
                 y2=(np.mean(np.array(participants_odr[roi]),0)+np.std(np.array(participants_odr[roi]),0)), \
                 color='b', alpha=.1)
    avg_all[roi] = np.mean(participants_odr[roi],0)
    plt.axvline(0, color='k');
    plt.axhline(.5, color='k', linestyle='--', label='chance');
    plt.title(roi)
    plt.show();
fig, ax = plt.subplots(1);
ax.plot(np.arange(-300,900,4),avg_all);
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.title('Decoding ODOURvsLD accuracy of each ROI separately')
plt.legend(kkROI);
plt.show();

avg_all = pd.DataFrame(index=range(300),columns=kkROI)

for roi in kkROI:  
    sns.lineplot(x=np.arange(-300,900,4), y=np.mean(avg_scores[roi],0))
    # please note that np.mean(np.vstack(participants_mlk[roi]),0) 
    # is equal to np.mean(participants_mlk[roi],0)
    plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(avg_scores[roi]),0)-np.std(np.array(avg_scores[roi]),0)), \
                 y2=(np.mean(np.array(avg_scores[roi]),0)+np.std(np.array(avg_scores[roi]),0)), \
                 color='b', alpha=.1)
    avg_all[roi] = np.mean(avg_scores[roi],0)
    plt.axvline(0, color='k');
    plt.axhline(.5, color='k', linestyle='--', label='chance');
    plt.title(roi)
    plt.show();
fig, ax = plt.subplots(1);
ax.plot(np.arange(-300,900,4),avg_all);
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.title('Decoding SDvsSD accuracy of each ROI separately')
plt.legend(kkROI);
plt.show();