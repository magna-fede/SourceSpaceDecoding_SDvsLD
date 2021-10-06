# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:34:13 2021

@author: fm02
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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


with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_avg_scores.P", 'rb') as f:
     SDSD_scores = pickle.load(f)

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1005_SDvsSD_coefficients.P", 'rb') as f:
     SDSD_coefficients = pickle.load(f)


kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']
    
SDSD_scores = pd.Series(SDSD_scores[0])
SDSD_coefficients = pd.Series(SDSD_coefficients[0])

SDSD_avg_score = []

for i in SDSD_scores:
    SDSD_avg_score.append(np.mean([i['milkVSfruit'].mean(0),
                                   i['fruitVSodour'].mean(0),
                                   i['odourVSmilk'].mean(0)],0))

SDSD_milkVSfruit_score = []
for i in SDSD_scores:
    SDSD_milkVSfruit_score.append(i['milkVSfruit'].mean(0))
SDSD_fruitVSodour_score = []
for i in SDSD_scores:
    SDSD_fruitVSodour_score.append(i['fruitVSodour'].mean(0))
SDSD_odourVSmilk_score = []
for i in SDSD_scores:
    SDSD_odourVSmilk_score.append(i['odourVSmilk'].mean(0))

sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(SDSD_avg_score),0));
plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(SDSD_avg_score),0)-np.std(np.array(SDSD_avg_score),0)), \
                 y2=(np.mean(np.array(SDSD_avg_score),0)+np.std(np.array(SDSD_avg_score),0)), \
                 color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.title('Average decoding accuracy SDvsSD considering all vertices')
plt.show()

sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(SDSD_milkVSfruit_score),0),
             color='tab:blue' ,label='milkVSfruit');
plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(SDSD_milkVSfruit_score),0)-np.std(np.array(SDSD_milkVSfruit_score),0)), \
                 y2=(np.mean(np.array(SDSD_milkVSfruit_score),0)+np.std(np.array(SDSD_milkVSfruit_score),0)), \
                 color='tab:blue', alpha=.1)
sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(SDSD_fruitVSodour_score),0),
             color='tab:green', label='fruitVSodour');
plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(SDSD_fruitVSodour_score),0)-np.std(np.array(SDSD_fruitVSodour_score),0)), \
                 y2=(np.mean(np.array(SDSD_fruitVSodour_score),0)+np.std(np.array(SDSD_fruitVSodour_score),0)), \
                 color='tab:green', alpha=.1)
sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(SDSD_odourVSmilk_score),0),
             color= 'tab:orange', label='odourVSmilk');
plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(SDSD_odourVSmilk_score),0)-np.std(np.array(SDSD_odourVSmilk_score),0)), \
                 y2=(np.mean(np.array(SDSD_odourVSmilk_score),0)+np.std(np.array(SDSD_odourVSmilk_score),0)), \
                 color='tab:orange', alpha=.1)
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('Classifier ability to decode type of task SDvsSD')
plt.legend();


######################################################################
# Plot  ROIs coefficients
######################################################################
avg_all = pd.DataFrame(index=range(300),columns=kkROI)

participants = pd.DataFrame(index=np.arange(0,18),columns=kkROI)
# calculate root-mean-squared(coefficients) for each ROI for each participant
for roi in kkROI:
   for i,df in enumerate(SDSD_coefficients):
       participants[roi].iloc[i] = rms(df['avg'][df['ROI']==roi])


# and plot the average of rms(coefficients) with sd
for roi in kkROI:
    plt.subplots(1)
    sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(participants[roi]),0))
    plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(participants[roi]),0)-np.std(np.array(participants[roi]),0)), \
                 y2=(np.mean(np.array(participants[roi]),0)+np.std(np.array(participants[roi]),0)), \
                 color='b', alpha=.1)
    plt.title(roi);
    avg_all[roi] = np.mean(np.array(participants[roi]),0)
fig, ax = plt.subplots(1);
ax.plot(np.arange(-300,900,4),avg_all)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.legend(kkROI);
