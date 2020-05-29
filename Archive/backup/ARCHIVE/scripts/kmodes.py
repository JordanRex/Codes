# kmodes code
# to be applied for a dataset with categorical features to be clustered on

#%reset -f

import os
import pandas as pd 
import numpy as np 
import math
from kmodes.kmodes import KModes
from collections import Counter
#os.listdir()

# function to prepare data for kmodes and implement a tuned model (tuned for distance)
def prepare(path):
    data = pd.read_csv(path)
    #print(data.columns, '\n')
    data['psg'] = np.where(pd.isnull(data['psg']),data['band'],data['psg'])
    data1 = data[['psg',u'career_potential_sign','group_risk',u'career_velocity_sign', u'leadership_sign', u'performance_sign', u'rewards_sign','department']]
    
    l={}
    
    for i in range(3,20):
        # init = ['Huang', 'Cao']
        kmodes_huang = KModes(n_clusters=i, init='Huang', verbose=0, n_init=10)
        kmodes_huang.fit(data1)
        
        # Print cluster centroids of the trained model.
        print('k-modes centroids:')
        print(kmodes_huang.cluster_centroids_)
        # Print training statistics
        print('Final training cost: {}'.format(kmodes_huang.cost_))
        print('Training iterations: {}'.format(kmodes_huang.n_iter_))
        
        key_1 = str('clusters_' + str(i))
        key_2 = str('training_cost_' + str(i))
        key_3 = str('training_iterations_' + str(i))
        key_4 = str('labels_' + str(i))
        key_5 = str('df_' + str(i))
        key_6 = str('Data_with_labels')

        value_1 = pd.DataFrame(kmodes_huang.cluster_centroids_, columns=data1.columns)
        value_2 = kmodes_huang.cost_
        value_3 = kmodes_huang.n_iter_
        value_4 = kmodes_huang.labels_
        
        l[key_1] = value_1
        l[key_2] = value_2
        l[key_3] = value_3
        l[key_4] = value_4
        
        labels_key = str('labels_' + str(i))
        df = pd.concat([data1.reset_index(drop=True), pd.DataFrame({labels_key: kmodes_huang.labels_})], axis=1)
        if i == 3 : df_complete = df
        if i > 3 : df_complete = pd.concat([df_complete.reset_index(drop=True), pd.DataFrame({labels_key: kmodes_huang.labels_})], axis=1)
        df1 = pd.DataFrame({'count' : df.groupby( [ labels_key, "group_risk"] ).size()}).reset_index()
        df2 = pd.DataFrame(df1.groupby(labels_key).agg({'count': ['max', 'sum', 'min']}))
        df2.columns=['_'.join(col) for col in df2.columns.values]
        df2.reset_index(inplace=True)
        
        df3 = (df1
           .sort_values([labels_key,'count'], ascending=False)
           .drop_duplicates([labels_key])
           .reset_index(drop=True)
           .rename(columns={'group_risk': 'top_risk_group', 'count': 'count_max'}))
        
        df_full = pd.merge(df3, df2, how='left')
        df_full['ratio'] = df_full['count_max']/df_full['count_sum']
        l[key_5] = df_full
    l[key_6] = df_complete    
    return l

l = prepare(path='xxx.csv')
display(l['Data_with_labels'])