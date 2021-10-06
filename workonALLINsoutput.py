# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:15:57 2021

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


with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0923_SDLD_scores.P", 'rb') as f:
     SDLD_scores = pickle.load(f)

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/0923_SDLD_coefficients.P", 'rb') as f:
     SDLD_coefficients = pickle.load(f)


kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']
    
SDLD_scores = pd.Series(SDLD_scores[0])
SDLD_coefficients = pd.Series(SDLD_coefficients[0])

SDLD_avg_score = []

for i in SDLD_scores:
    SDLD_avg_score.append(np.mean([i['milk'].mean(0),
                                   i['fruit'].mean(0),
                                   i['odour'].mean(0)],0))

SDLD_milk_score = []
for i in SDLD_scores:
    SDLD_milk_score.append(i['milk'].mean(0))
SDLD_fruit_score = []
for i in SDLD_scores:
    SDLD_fruit_score.append(i['fruit'].mean(0))
SDLD_odour_score = []
for i in SDLD_scores:
    SDLD_odour_score.append(i['odour'].mean(0))

sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(SDLD_avg_score),0));
plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(SDLD_avg_score),0)-np.std(np.array(SDLD_avg_score),0)), \
                 y2=(np.mean(np.array(SDLD_avg_score),0)+np.std(np.array(SDLD_avg_score),0)), \
                 color='b', alpha=.1)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.title('Decoding AVG(SD)vsLD accuracy considering all vertices')
plt.show()

sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(SDLD_milk_score),0),
             color='tab:blue' ,label='milk');
plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(SDLD_milk_score),0)-np.std(np.array(SDLD_milk_score),0)), \
                 y2=(np.mean(np.array(SDLD_milk_score),0)+np.std(np.array(SDLD_milk_score),0)), \
                 color='tab:blue', alpha=.1)
sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(SDLD_fruit_score),0),
             color='tab:green', label='fruit');
plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(SDLD_fruit_score),0)-np.std(np.array(SDLD_fruit_score),0)), \
                 y2=(np.mean(np.array(SDLD_fruit_score),0)+np.std(np.array(SDLD_fruit_score),0)), \
                 color='tab:green', alpha=.1)
sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(SDLD_odour_score),0),
             color= 'tab:orange', label='odour');
plt.fill_between(x=np.arange(-300,900,4), \
                 y1=(np.mean(np.array(SDLD_odour_score),0)-np.std(np.array(SDLD_odour_score),0)), \
                 y2=(np.mean(np.array(SDLD_odour_score),0)+np.std(np.array(SDLD_odour_score),0)), \
                 color='tab:orange', alpha=.1)
plt.axhline(.5, color='k', linestyle='--', label='chance');
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.title('Classifier ability to decode type of task SDvsLD')
plt.legend();


######################################################################
# Plot  ROIs coefficients
######################################################################
avg_all = pd.DataFrame(index=range(300),columns=kkROI)

participants = pd.DataFrame(index=np.arange(0,18),columns=kkROI)
# calculate root-mean-squared(coefficients) for each ROI for each participant
for roi in kkROI:
   for i,df in enumerate(SDLD_coefficients):
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


# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
#
# semantic category detection is not working right now!   
#
# ######################################################################
# # Plot average decoding scores of 5 splits for each model
# ######################################################################
# cat_lds_score = []
# for i in cat_scores:
#     cat_lds_score.append(i['scores_lds'].mean(0))

# sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(cat_lds_score),0));
# plt.fill_between(x=np.arange(-300,900,4), \
#                  y1=(np.mean(np.array(cat_lds_score),0)-np.std(np.array(cat_lds_score),0)), \
#                  y2=(np.mean(np.array(cat_lds_score),0)+np.std(np.array(cat_lds_score),0)), \
#                  color='b', alpha=.1)
# plt.axhline(.2, color='k', linestyle='--', label='chance');
# plt.set_title('Lexical Decision')

# cat_odrs_score = []
# for i in cat_scores:
#     cat_odrs_score.append(i['scores_odrs'].mean(0))

# plt.subplot()
# sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(cat_odrs_score),0))
# plt.fill_between(x=np.arange(-300,900,4), \
#                  y1=(np.mean(np.array(cat_odrs_score),0)-np.std(np.array(cat_odrs_score),0)), \
#                  y2=(np.mean(np.array(cat_odrs_score),0)+np.std(np.array(cat_odrs_score),0)), \
#                  color='b', alpha=.1)
# plt.axhline(.2, color='k', linestyle='--', label='chance')
# plt.show();

# cat_frts_score = []
# for i in cat_scores:
#     cat_frts_score.append(i['scores_frts'].mean(0))

# plt.subplot()
# sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(cat_frts_score),0))
# plt.fill_between(x=np.arange(-300,900,4), \
#                  y1=(np.mean(np.array(cat_frts_score),0)-np.std(np.array(cat_frts_score),0)), \
#                  y2=(np.mean(np.array(cat_frts_score),0)+np.std(np.array(cat_frts_score),0)), \
#                  color='b', alpha=.1)
# plt.axhline(.2, color='k', linestyle='--', label='chance')
# plt.show();

# cat_mlks_score = []
# for i in cat_scores:
#     cat_mlks_score.append(i['scores_mlks'].mean(0))

# plt.subplot()
# sns.lineplot(x=np.arange(-300,900,4), y=np.mean(np.array(cat_mlks_score),0))
# plt.fill_between(x=np.arange(-300,900,4), \
#                  y1=(np.mean(np.array(cat_mlks_score),0)-np.std(np.array(cat_mlks_score),0)), \
#                  y2=(np.mean(np.array(cat_mlks_score),0)+np.std(np.array(cat_mlks_score),0)), \
#                  color='b', alpha=.1)
# plt.axhline(.2, color='k', linestyle='--', label='chance')
# plt.show();

# with open("SDLD_coefficients.P", 'rb') as f:
#     SDLD_coefficients = pickle.load(f)
# with open("SDLD_scores.P", 'rb') as f:
#      SDLD_scores = pickle.load(f)
    
# with open("categories_scores.P", 'rb') as f:
#      cat_scores = pickle.load(f)
     