
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


with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/LDA/LDvsSD/1123_LDA_SDLD_scores.P", 'rb') as f:
      SDLD_scores = pickle.load(f)

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/LDA/LDvsSD/1123_LDA_SDLD_coefficients.P", 'rb') as f:
      SDLD_coefficients = pickle.load(f)

# with open("//imaging/hauk/users/fm02/first_output/0923_SDLD_scores.P", 'rb') as f:
#      SDLD_scores = pickle.load(f)

# with open("//imaging/hauk/users/fm02/first_output/0923_SDLD_coefficients.P", 'rb') as f:
#      SDLD_coefficients = pickle.load(f)


kk2 = ['visual', 'hand', 'hear', 'neutral','emotional']
kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']
    
SDLD_scores = pd.Series(SDLD_scores)
SDLD_coefficients = pd.Series(SDLD_coefficients)

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


# # if just one participant, then use
# participants = pd.DataFrame(index=np.arange(0,1),columns=kkROI)

participants = pd.DataFrame(index=np.arange(0,18),columns=kkROI)
# calculate root-mean-squared(coefficients) for each ROI for each participant
for roi in kkROI:
   for i,df in enumerate(SDLD_coefficients):
       participants[roi].iloc[i] = rms(df['avg'][df['ROI']==roi])

# avg_rolling = pd.DataFrame(index=range(300),columns=kkROI)

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
    # avg_rolling[roi] = np.mean(np.array(pd.DataFrame(participants[roi][0]).rolling(2, win_type='gaussian').sum(std=3)),0)
fig, ax = plt.subplots(1);
ax.plot(np.arange(-300,900,4),avg_all)
plt.axvline(0, color='k');
plt.axvline(50, color='k', linewidth=1, alpha=0.3);
plt.axvline(100, color='k',linewidth=1, alpha=0.3);
plt.axvline(150, color='k', linewidth=1, alpha=0.3);
plt.axvline(200, color='k', linewidth=1, alpha=0.3);
plt.legend(kkROI);

sns.heatmap(avg_all.T, cmap="YlGnBu", xticklabels=False)
plt.axvline(75, color='k');
plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(150, color='k',linewidth=1, alpha=0.3);
plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(225, color='k', linewidth=1, alpha=0.3);
plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])

plt.title(f"Patterns in each ROI");
ax.figsize=(15,15)
plt.show()