# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:25:21 2016

@author: edin
"""
from __future__ import division

import logging
import time

import pandas as pd
import numpy as np
import prepare_data as prep
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib as mpl

    
def get_trace(user, geometry='grid', delta_l=100, delta_t=15, only_POI=False):
    filename = '/home/edin/paper/limits_of_predictability/traces/'
    if only_POI:
        filename += 'only_POI/'
    filename += geometry +'/' + str(delta_l) +'/' + user+ '_' + str(delta_t) + '.csv'                
    
    trace = pd.read_csv(filename, index_col=0, parse_dates = True, header=None)
    return trace

def get_binary_df(trace, n=4):
    loc_prob = trace['loc'].value_counts(normalize=True,sort=True)
    result = pd.DataFrame()
    for j in range(n):
        a = trace.pivot(index='date', columns='hour', values='loc_'+str(j))
        result = pd.concat([result, a], axis=1, ignore_index=True)
    a = trace.pivot(index='date', columns='hour', values='loc_n')
    result = pd.concat([result, a], axis=1, ignore_index=True)
    a = trace.pivot(index='date', columns='hour', values='loc_error')
    result = pd.concat([result, a], axis=1, ignore_index=True)
    result[result!=0] = 1
    return result[1:-1]

def get_full_df(df_binary):
    col = df_binary.columns
    loc_0 = df_binary[col[0:24]]
    for i in range(1, int(df_binary.shape[1]/24)):
        loc_n = df_binary[col[i*24:(i+1)*24]]
        loc_n = loc_n*(i+1)
        loc_n.columns = col[0:24]
        loc_0 = loc_0.add(loc_n)
    loc_0 -= 1
    return loc_0

def get_full_PCA(component):
    b = component[0:24]
    for i in range(1, int(len(component)/24)):
        b += component[24*i:24*(i+1)]*i
    return b

def get_loc_matrix(trace, n=4):
    loc_matrix = trace.pivot(index='date', columns='hour', values='all_locs')
    return loc_matrix[1:-1]


def prepare_trace(trace, n=4):
    trace['idx'] = trace.index    
    trace['hour'] = trace['idx'].apply(lambda x: x.hour)
    trace['date'] = trace['idx'].apply(lambda x: x.isocalendar())
    trace['loc'] = trace[1]
    loc_prob = trace['loc'].value_counts(normalize=True,sort=True)
    top_locations = loc_prob[0:n]
    if -1 in top_locations.index:
        top_locations = loc_prob[0:n+1]
        top_locations = top_locations[top_locations.index != -1]
    trace['loc_error'] = trace['loc'].apply(lambda x: n+2 if x==-1 else 0)
    trace['loc_n'] = trace['loc'].apply(lambda x: n+1 if x not in top_locations and x!=-1 else 0)
    trace['all_locs'] = trace['loc_error'] + trace['loc_n']
    for i, location in enumerate(top_locations.index):
        trace['loc_'+str(i)] = trace['loc'].apply(lambda x: i+1 if x == location else 0)
        trace['all_locs'] = trace['all_locs'] + trace['loc_'+str(i)]
    return trace

def get_binary_matrix(loc_matrix):
    loc_prob = trace['loc_n'].value_counts(normalize=True,sort=True)
    result = pd.DataFrame()
    for j in range(len(loc_prob.unique())):
        a = trace.pivot(index='date', columns='hour', values='loc_'+str(j))
        result = pd.concat([result, a], axis=1, ignore_index=True)
    return result

user_list = prep.get_userlist()

# n denotes the number of locations I want to incorporate in the analysis. 
# n = 4 takes the top 4 most frequent locations and encodes them as 1,2,3,4
# the rest of the locations are encoded as n+1 and all missing data is encoded as n+2
n = 4

for i, user in enumerate(user_list[0:1]):
    user = user_list[11]
    trace = get_trace(user, delta_l=400, delta_t=60) 
    # trace = trace[500:2500]
    trace = prepare_trace(trace)
    loc_matrix = get_loc_matrix(trace[1000:3000], n=n)

    cmap1 = mpl.colors.ListedColormap(sns.color_palette("husl", 5))
    
    yticks = loc_matrix.index
    
    sns.heatmap(loc_matrix, vmin=1, vmax=n+2, cmap =cmap1,annot=False, linewidths=0.1)
    plt.yticks(rotation=0) 
    plt.show()
#     

#     trace = trace[trace[1] != -1]
#     trace = trace.iloc[:1500] 
#     trace['idx'] = trace.index    
#     trace['hour'] = trace['idx'].apply(lambda x: x.hour)
#     trace['date'] = trace['idx'].apply(lambda x: x.isocalendar())
#     trace['loc'] = trace[1]
#     loc_prob = trace['loc'].value_counts(normalize=True,sort=True)
#     trace['loc_pop'] = trace['loc'].apply(lambda x: loc_prob.at[x])
#     trace['loc_n'] = trace['loc_pop'].apply(lambda x: np.where(loc_prob.unique() == x)[0][0] if x !=-1 else len(loc_prob.unique())+2)
    
#     for j in range(len(loc_prob.unique())):
#         trace['loc_'+str(j)] = trace['loc_n'].apply(lambda x: 1 if x==j else 0)
#     cmap1 = mpl.colors.ListedColormap(sns.color_palette("husl", len(loc_prob.unique())))
#     loc_matrix = trace.pivot(index='date', columns='hour', values='loc_n')
    
#     yticks = loc_matrix.index
    
#     sns.heatmap(loc_matrix, vmin=max(trace['loc_n'].unique()), vmax=min(trace['loc_n'].unique()), cmap =cmap1,annot=False, linewidths=0.1)
#     plt.yticks(rotation=0) 
#     plt.show()
#     # # plt.savefig(str(i)+'test.png')

binary_df = get_binary_df(trace)
binary_df = binary_df[binary_df[binary_df.columns[24*(n+1):]].sum(axis=1) < 12]

print "For this user we have %i days worht of data" % (binary_df.shape[0])

means = binary_df - binary_df.mean()

pca = PCA(n_components=5)
pca.fit(means.dropna())

# first_component = pca.components_[0]
# first_full_vec = get_full_PCA(first_component)

# second_component = pca.components_[1]
# second_full_vec = get_full_PCA(second_component)

# plt.plot(pca.components_[0][0:24], color='red')
# plt.plot(pca.components_[0][24:48], color='blue')
# plt.plot(pca.components_[0][48:72], color='green')


# plt.plot(pca.components_[1][0:24], color='red', alpha=0.5)
# plt.plot(pca.components_[1][24:48], color='blue', alpha=0.5)
# plt.plot(pca.components_[1][48:72], color='green', alpha=0.5)
# plt.show()



import pylab as pl
pl.pcolor(pca.components_[0:3], cmap='RdBu')
pl.colorbar()
pl.show()