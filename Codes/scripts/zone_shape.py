# zone shape module

##################################################################################################

# create zone level dictionary
pred_withzone = predictions_df.merge(preddf[['global_id', 'zone']], how='left', on=['global_id'])
zone_dict = pred_withzone.zone.unique()
#create a data frame dictionary to store your data frames
zone_dict = {x: pd.DataFrame for x in zone_dict}
zone_dict_ids = {x: pd.DataFrame for x in zone_dict}
for key in zone_dict.keys():
    zone_dict[key] = pred_withzone[:][pred_withzone.zone == key]
    zone_dict[key] = zone_dict[key][['1A', '3B', '3A', '4B', '4A']]
    zone_dict_ids[key] = pred_withzone[:][pred_withzone.zone == key]    
    zone_dict_ids[key] = zone_dict_ids[key][['global_id']]

##################################################################################################
# shape function
def opr_shape(zone_dict, zone_dict_ids, global_distribution_df):
    global_finaldf = {}
    for key in zone_dict.keys():
        probs_df = {}
        for i in range(zone_dict[key].shape[1]):
            probs_df[i] = pd.DataFrame({str('prob_'+str(i)): zone_dict[key].iloc[:,i]})
            probs_df[i]['true_index'] = probs_df[i].index
            probs_df[i].sort_values(by=[str('prob_'+str(i))], inplace=True, kind='mergesort', ascending=False)
            probs_df[i].reset_index(inplace=True, drop=True)
            probs_df[i][str(str(i)+'_new_index')] = probs_df[i].index
            probs_df[i][str(str(i)+'_index_perc')] = probs_df[i][str(str(i)+'_new_index')].rank(pct=True)
            probs_df[i][str(str(i)+'_flag')] = np.where(probs_df[i][str(str(i)+'_index_perc')] < 
                                                        global_distribution_df.iloc[i, 1], 1, 0)

        merge = functools.partial(pd.merge, left_index=False, right_index=False, how='inner', on='true_index')
        zone_finaldf = functools.reduce(merge, probs_df.values())

        zone_finaldf['class'] = np.where(zone_finaldf['4_flag']==1, 
                                         4, 
                                         np.where(zone_finaldf['3_flag']==1, 
                                                  3, 
                                                  np.where(zone_finaldf['0_flag']==1, 
                                                           0, 
                                                           np.where(zone_finaldf['2_flag']==1, 2, 1))))
        global_finaldf[key] = zone_finaldf[['true_index', 'class']]
        global_finaldf[key].set_index('true_index', inplace=True)
        global_finaldf[key].sort_index(inplace=True)
        global_finaldf[key].reset_index(drop=True, inplace=True)

    GLOBAL_aftershape_dict = {}
    for key in zone_dict_ids.keys():
        GLOBAL_aftershape_dict[key] = pd.concat([zone_dict_ids[key].reset_index(drop=True), global_finaldf[key]], axis=1)
    GLOBAL_aftershape_df = pd.concat(GLOBAL_aftershape_dict.values(), ignore_index=True)
    return GLOBAL_aftershape_df
    
#GLOBAL_aftershape_df = opr_shape(zone_dict, zone_dict_ids, global_distribution_df)

# df = df.merge(GLOBAL_aftershape_df, on=['global_id'], how='left')
# df['rules_prediction'] = df['rules_prediction'].map(dep_dict_without1B)
# df['predictions'] = np.where(df['rules_prediction'].isna, df['class'], df['rules_prediction'])
# df['predictions'] = df['predictions'].astype(int)
# df.drop(columns=['rules_prediction', 'class'], inplace=True)
