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


# with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1101_SDvsSD_ranks.P", 'rb') as f:
#      SDvsSD = pickle.load(f)

# kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

# SDvsSD_ranks = list(SDvsSD.values())
     
# big_ranks_SD = pd.concat(SDvsSD_ranks)

# avg_big_ranks_SD = big_ranks_SD.groupby(by=big_ranks_SD.index).mean()

# ax = sns.heatmap(avg_big_ranks_SD, cmap="YlGnBu", xticklabels=False)
# plt.axvline(75, color='k');
# plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
# plt.axvline(150, color='k',linewidth=1, alpha=0.3);
# plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
# plt.axvline(225, color='k', linewidth=1, alpha=0.3);
# plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])

# plt.title("SDvsSD average rank (low is better)");

# plt.show()

# # for i,df in enumerate(SDvsSD_ranks):
# #     ax = sns.heatmap(df, cmap="YlGnBu")
# #     plt.axvline(75, color='k');
# #     plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
# #     plt.axvline(150, color='k',linewidth=1, alpha=0.3);
# #     plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
# #     plt.axvline(225, color='k', linewidth=1, alpha=0.3);
# #     plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
    
# #     plt.title(f"ranks participant {i}")
# #     plt.show()

# for roi in kkROI:
#     ax = avg_big_ranks_SD.loc[roi].plot()
# plt.legend()
# ax.set_ylim(ax.get_ylim()[::-1])
# plt.axvline(75, color='k');
# plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
# plt.axvline(150, color='k',linewidth=1, alpha=0.3);
# plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
# plt.axvline(225, color='k', linewidth=1, alpha=0.3);
# plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
# plt.show()

# ------------------------------------------------------------

with open("//cbsu/data/Imaging/hauk/users/fm02/first_output/1109_LDvsSD_avgless05.P", 'rb') as f:
     LDvsSD = pickle.load(f)

kkROI = ['lATL', 'rATL', 'AG', 'PTC', 'IFG', 'PVA']

LDvsSD_ranks = list(LDvsSD.values())

lis = list(range(18))
times = len(LDvsSD_ranks[0])

big_ranks_LD = pd.concat(LDvsSD_ranks)
big_ranks_LD['participant'] = sum(([x]*times for x in lis),[])
big_ranks_LD = big_ranks_LD.set_index(['participant', big_ranks_LD.index])    

for task in ['milk','fruit','odour','avg']:
    avg_significantcoef = big_ranks_LD.loc[:,task,:].groupby(by='ROI').mean()
    
    ax = sns.heatmap(avg_significantcoef, cmap="YlGnBu", xticklabels=False)
    plt.axvline(75, color='k');
    plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
    plt.axvline(150, color='k',linewidth=1, alpha=0.3);
    plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
    plt.axvline(225, color='k', linewidth=1, alpha=0.3);
    plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
    
    plt.title(f"{task} : ROI Average coefficients (SelectKBest p<.05)");
    ax.figsize=(15,15)
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

avg_significantcoef = big_ranks_LD.loc[:,'avg',:].groupby(by='ROI').mean()

for roi in kkROI:
    avg_significantcoef.loc[roi].plot()    
plt.legend()
ax.set_ylim(ax.get_ylim()[::-1])
plt.axvline(75, color='k');
plt.axvline(112.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(150, color='k',linewidth=1, alpha=0.3);
plt.axvline(187.5, color='k', linewidth=1, alpha=0.3);
plt.axvline(225, color='k', linewidth=1, alpha=0.3);
plt.xticks([0, 75, 112.5, 150, 187.5, 225, 275], ['-300','0', '150', '300', '450', '600', '800'])
plt.title("LDvsSD ROIs Average coefficients (SelectKBest p<.05)")
plt.show()
