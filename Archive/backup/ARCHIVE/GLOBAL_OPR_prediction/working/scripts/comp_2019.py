# competency 2019
## add to the comp_full dataframe

# with adlsfsc.open(path + '/2019/Data/Raw_Data/navigate/competency/rating_files/comp2019_fc.xlsx') as f:
#     comp2019 = helpers.xlsx_read(f)

# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_2019.pickle', 'wb') as f:
#     pickle.dump(comp2019, f)
#     f.close()

# comp_2019 = open('E:/ADLS/pickles/competency_2019.pickle', 'wb')
# pickle.dump(comp2019, comp_2019)
# comp_2019.close()

# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_2019.pickle', 'rb') as f:
#     comp2019 = pickle.load(f)
#     f.close()

comp_2019 = open('E:/ADLS/pickles/competency_2019.pickle', 'rb')
comp2019 = pickle.load(comp_2019)
comp_2019.close()

comp2019 = comp2019[['employee_global_id', 'competency_group', 'competency', 'manager_rating_numeric_value', 'year']]
comp2019.dropna(inplace=True)
comp2019['competency_group_type_l1'] = comp2019['competency'].str.split(' - ').str[0]
comp2019 = helpers.process_columns(comp2019, cols=['competency_group', 'competency', 
                                                     'competency_group_type_l1'])
comp2019.reset_index(inplace=True, drop=True)

# creating the grouped dfs and merging the new features (from manager_rating)
comp2019 = cust_funcs.group_agg_feats(comp2019, group_cols=['employee_global_id', 'competency_group', 'year'], 
                                       agg_col='manager_rating_numeric_value',
                                       new_cols=['employee_global_id', 'competency_group', 'year', 
                                                 'mr_pers_compgroup_year_comp_score_sum', 
                                                 'mr_pers_compgroup_year_comp_score_mean'])
comp2019 = cust_funcs.group_agg_feats(comp2019, group_cols=['employee_global_id', 'competency_group', 
                                                              'competency_group_type_l1', 'year'], 
                                       agg_col='manager_rating_numeric_value', 
                                       new_cols=['employee_global_id', 'competency_group', 'competency_group_type_l1', 
                                                 'year', 'mr_pers_compgroupl1_year_comp_score_sum', 
                                                 'mr_pers_compgroupl1_year_comp_score_mean'])

comp_temp_grpl1 = comp2019.copy()
comp_temp_grpl2 = comp2019.copy()
comp_temp_grpl1 = comp_temp_grpl1[['employee_global_id', 'year', 'competency_group', 
                       'mr_pers_compgroup_year_comp_score_mean']]
comp_temp_grpl2 = comp_temp_grpl2[['employee_global_id', 'year', 'competency_group', 'competency_group_type_l1',
                       'mr_pers_compgroupl1_year_comp_score_mean']]
comp_temp_grpl1.drop_duplicates(inplace=True)
comp_temp_grpl2.drop_duplicates(inplace=True)

comp_temp_grpl1 = comp_temp_grpl1.groupby(['employee_global_id', 'year', 
                                           'competency_group']).sum().unstack('competency_group').reset_index().my_flatten_cols()
comp2019 = comp2019.merge(comp_temp_grpl1, how='left', on=['employee_global_id', 'year'])
comp2019.drop(['mr_pers_compgroup_year_comp_score_mean'], axis=1, inplace=True)
comp2019.drop_duplicates(inplace=True, subset=['employee_global_id', 'year'])
comp_temp_grpl2 = comp_temp_grpl2.groupby(['employee_global_id', 'year', 'competency_group', 
                                          'competency_group_type_l1']).sum().unstack(['competency_group', 
                                                                'competency_group_type_l1']).reset_index().my_flatten_cols()
comp2019 = comp2019.merge(comp_temp_grpl2, how='left', on=['employee_global_id', 'year'])
comp2019.drop(['competency_group', 'competency_group_type_l1', 'mr_pers_compgroupl1_year_comp_score_mean'], 
               axis=1, inplace=True)
comp2019.drop_duplicates(inplace=True, subset=['employee_global_id', 'year'])
comp2019.rename({'employee_global_id': 'global_id'}, axis=1, inplace=True)

comp2019.drop(list(comp2019.filter(regex = '_functional_competencies_')), axis = 1, inplace = True)

comp2019_fc = comp2019[['global_id', 'year', 'mr_pers_compgroup_year_comp_score_mean_functional_competencies']]
comp_180_lc = comp2019[['global_id',
                        'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_develop_people',
                        'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_dream_big',
                        'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_live_our_culture',
                        'mr_pers_compgroup_year_comp_score_mean_leadership_competencies']]

###################################################################################################
# processed backups
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_processed_2019.pickle', 'wb') as f:
#     pickle.dump(comp2019_fc, f)
#     f.close()

comp_2019 = open('E:/ADLS/pickles/competency_processed_2019.pickle', 'wb')
pickle.dump(comp2019_fc, comp_2019)
comp_2019.close()

# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_processed_2019.pickle', 'rb') as f:
#     comp2019_fc = pickle.load(f)
#     f.close()

comp_2019 = open('E:/ADLS/pickles/competency_processed_2019.pickle', 'rb')
comp2019_fc = pickle.load(comp_2019)
comp_2019.close()

###################################################################################################

with adlsfsc.open(path + '/2019/Data/Raw_Data/navigate/competency/rating_files/comp2019_lc.xlsx') as f:
    comp2019_lc = helpers.xlsx_read(f)
comp2019_lc = comp2019_lc[['global_id', 'competency', 'average']]

comp2019_lc = comp2019_lc.pivot(index='global_id', columns='competency', values='average').reset_index()
comp2019_lc.columns.name = None
comp2019_lc.columns = ['global_id',
       'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_develop_people',
       'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_dream_big',
       'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_live_our_culture',
                      'mr_pers_compgroup_year_comp_score_mean_leadership_competencies']

## Identifying IDs for which LC needs to be taken from 180 file and appending t0 360 LC data
comp_180_lc.dropna(axis=0, how='all',inplace=True)
ids_lc_360 = list(comp2019_lc['global_id'])
comp_180_lc = comp_180_lc[~comp_180_lc['global_id'].isin(ids_lc_360)]
comp2019_lc = pd.concat([comp2019_lc,comp_180_lc], ignore_index=True)

comp2019 = comp2019_fc.merge(comp2019_lc, how='outer', on=['global_id'])
comp2019['year'] = 2019

###################################################################################################

# processed backups
with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_full_2019.pickle', 'wb') as f:
    pickle.dump(comp2019, f)
    f.close()

comp_2019 = open('E:/ADLS/pickles/competency_full_2019.pickle', 'wb')
pickle.dump(comp2019, comp_2019)
comp_2019.close()
