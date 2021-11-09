# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:40:28 2021
@author: fm02
"""


import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1101_SDvsSD_ranks.P", 'rb') as f:
     SDvsSD = pickle.load(f)

kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

SDvsSD_ranks = list(SDvsSD.values())
     
big_ranks_SD = pd.concat(SDvsSD_ranks)

avg_big_ranks_SD = big_ranks_SD.groupby(by=big_ranks_SD.index).mean()

ax = sns.heatmap(avg_big_ranks_SD, cmap="YlGnBu", xticklabels=False)
plt.axvline(75, color='k');
plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(150, color='k',linewidth=1, alpha=0.3);
plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(225, color='k', linewidth=1, alpha=0.3);
plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])

plt.title("SDvsSD average rank (low is better)");

plt.show()

# for i,df in enumerate(SDvsSD_ranks):
#     ax = sns.heatmap(df, cmap="YlGnBu")
#     plt.axvline(75, color='k');
#     plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
#     plt.axvline(150, color='k',linewidth=1, alpha=0.3);
#     plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
#     plt.axvline(225, color='k', linewidth=1, alpha=0.3);
#     plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
    
#     plt.title(f"ranks participant {i}")
#     plt.show()

for roi in kkROI:
    ax = avg_big_ranks_SD.loc[roi].plot()
plt.legend()
ax.set_ylim(ax.get_ylim()[::-1])
plt.axvline(75, color='k');
plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(150, color='k',linewidth=1, alpha=0.3);
plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(225, color='k', linewidth=1, alpha=0.3);
plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
plt.show()

# ------------------------------------------------------------

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1101_LDvsSD_ranks.P", 'rb') as f:
     LDvsSD = pickle.load(f)

kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

LDvsSD_ranks = list(LDvsSD.values())
     
big_ranks_LD = pd.concat(LDvsSD_ranks)

avg_big_ranks_LD = big_ranks_LD.groupby(by=big_ranks_LD.index).mean()

ax = sns.heatmap(avg_big_ranks_LD, cmap="YlGnBu", xticklabels=False)
plt.axvline(75, color='k');
plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(150, color='k',linewidth=1, alpha=0.3);
plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(225, color='k', linewidth=1, alpha=0.3);
plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])

plt.title("LDvsSD average rank (low is better)");

plt.show()

# for i,df in enumerate(LDvsSD_ranks):
#     ax = sns.heatmap(df, cmap="YlGnBu")
#     plt.axvline(75, color='k');
#     plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
#     plt.axvline(150, color='k',linewidth=1, alpha=0.3);
#     plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
#     plt.axvline(225, color='k', linewidth=1, alpha=0.3);
#     plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
    
#     plt.title(f"LDvsSD ranks participant {i}")
#     plt.show()

for roi in kkROI:
    ax = avg_big_ranks_LD.loc[roi].plot()
plt.legend()
ax.set_ylim(ax.get_ylim()[::-1])
plt.axvline(75, color='k');
plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(150, color='k',linewidth=1, alpha=0.3);
plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(225, color='k', linewidth=1, alpha=0.3);
plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
plt.title("LDvsSD average ranks")
plt.show()

for roi in kkROI:
    ax = avg_big_ranks_LD.loc[roi].plot()
    ax = avg_big_ranks_SD.loc[roi].plot()
    plt.legend(['LD','SD'])
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.axvline(75, color='k');
    plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
    plt.axvline(150, color='k',linewidth=1, alpha=0.3);
    plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
    plt.axvline(225, color='k', linewidth=1, alpha=0.3);
    plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
    plt.title(f"LDvsSD average ranks in {roi}")
    plt.show()
    
difference_ranks = pd.DataFrame(index=kkROI,
                                columns=range(300))
 
from scipy.stats import ttest_rel
   
for roi in kkROI:
    for i in range(300):
        _ , difference_ranks[i].loc[roi] = ttest_rel(big_ranks_LD.loc[roi][i],
                          big_ranks_SD.loc[roi][i])
        
for roi in kkROI:
    ax = difference_ranks[difference_ranks<.05].loc[roi].plot()
plt.legend()

plt.axvline(75, color='k');
plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(150, color='k',linewidth=1, alpha=0.3);
plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(225, color='k', linewidth=1, alpha=0.3);
plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
plt.title("LDvsSD average ranks")
plt.show()    