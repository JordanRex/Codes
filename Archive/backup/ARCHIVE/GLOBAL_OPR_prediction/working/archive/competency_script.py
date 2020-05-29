## competency script

# the competency file sources are rar files. the below snippet needs to be modified a bit to work directly with the rar files.
# for now, for convenience, manually extracted the csvs from the rar files (not recommended)
# import os, rarfile
# def unrar():
#     for rar in os.listdir(dpath):
#         filepath = os.path.join(dpath, rar)
#         opened_rar = rarfile.RarFile(filepath)
#         for f in opened_rar.infolist():
#             print (f.filename, f.file_size)
#         opened_rar.extractall(xpath)
#################################################################################################################

def download_competency_files_from_adls_folder(adls, path, cols):
    files_list = adls.ls(path)
    comp_pattern = re.compile('Comp app')
    files_list = [s for s in files_list if comp_pattern.search(s)]
    filesdict = {}
    for i in tqdm(files_list):
        file_year = i.split('/')[-1].split(' ')[-1].split('.')[0]
        with adls.open(i) as f:
            filesdict[file_year] = helpers.xlsx_read(file_path=f, drop_dup='yes')
            filesdict[file_year] = filesdict[file_year][cols]
            filesdict[file_year]['year'] = int(file_year)
    df = pd.concat(filesdict.values(), ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    df.dropna(subset=['manager_rating_numeric_value', 'employee_global_id', 'year'], how='any', inplace=True)
    df['competency_group_type_l1'] = df['competency'].str.split(' - ').str[0]
    df['competency_group_type_l2'] = df['competency'].str.split(' - ').str[1]
    df = helpers.process_columns(df, cols=['competency_group', 'competency', 
                                                         'competency_group_type_l1', 'competency_group_type_l2'])
    return df
comp_full = download_competency_files_from_adls_folder(adlsfsc, 
                                                       path + '/2019/Data/Raw_Data/navigate/competency/rating_files/',
                                                      cp_cols_to_keep)

compfull = open('E:/ADLS/pickles/competency_16to18_raw.pickle', 'wb')
pickle.dump(comp_full, compfull)
compfull.close()

# # save interim raw competency backup
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_16to18_raw.pickle', 'wb') as f:
#     pickle.dump(comp_full, f)
#     f.close()
    
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_16to18_raw.pickle', 'rb') as f:
#     comp_full = pickle.load(f)
#     f.close()

# compfull = open('E:/ADLS/pickles/competency_16to18_raw.pickle', 'rb')
# comp_full = pickle.load(compfull)
# compfull.close()
##########################################################################################################

############## CREATING THE FEATURES ####################
# creating the grouped dfs and merging the new features (from manager_rating)
comp_full = cust_funcs.group_agg_feats(comp_full, group_cols=['employee_global_id', 'year'], 
                                       agg_col='manager_rating_numeric_value',
                                       new_cols=['employee_global_id', 'year', 'mr_pers_year_comp_score_sum', 
                                                 'mr_pers_year_comp_score_mean'])
comp_full = cust_funcs.group_agg_feats(comp_full, group_cols=['employee_global_id', 'competency_group', 'year'], 
                                       agg_col='manager_rating_numeric_value',
                                       new_cols=['employee_global_id', 'competency_group', 'year', 
                                                 'mr_pers_compgroup_year_comp_score_sum', 'mr_pers_compgroup_year_comp_score_mean'])
comp_full = cust_funcs.group_agg_feats(comp_full, group_cols=['employee_global_id', 'competency_group', 
                                                              'competency_group_type_l1', 'year'], 
                                       agg_col='manager_rating_numeric_value', 
                                       new_cols=['employee_global_id', 'competency_group', 'competency_group_type_l1', 
                                                 'year', 'mr_pers_compgroupl1_year_comp_score_sum', 
                                                 'mr_pers_compgroupl1_year_comp_score_mean'])

# creating the grouped dfs and merging the new features (from employee_rating)
comp_full = cust_funcs.group_agg_feats(comp_full, group_cols=['employee_global_id', 'year'], 
                                       agg_col='employee_rating_numeric_value',
                                       new_cols=['employee_global_id', 'year', 'er_pers_year_comp_score_sum', 
                                                 'er_pers_year_comp_score_mean'])
comp_full = cust_funcs.group_agg_feats(comp_full, group_cols=['employee_global_id', 'competency_group', 'year'], 
                                       agg_col='employee_rating_numeric_value',
                                       new_cols=['employee_global_id', 'competency_group', 'year', 
                                                 'er_pers_compgroup_year_comp_score_sum', 'er_pers_compgroup_year_comp_score_mean'])
comp_full = cust_funcs.group_agg_feats(comp_full, group_cols=['employee_global_id', 'competency_group', 
                                                              'competency_group_type_l1', 'year'], 
                                       agg_col='employee_rating_numeric_value', 
                                       new_cols=['employee_global_id', 'competency_group', 'competency_group_type_l1', 
                                                 'year', 'er_pers_compgroupl1_year_comp_score_sum', 
                                                 'er_pers_compgroupl1_year_comp_score_mean'])

##########################################################################################################
# create the difference features
cols = ['pers_year_comp_score_mean', 'pers_compgroup_year_comp_score_mean', 'pers_compgroupl1_year_comp_score_mean']
for i in cols:
    newcol=str('diff_' + i)
    ercol=str('er_'+i)
    mrcol=str('mr_'+i)
    comp_full[newcol] = comp_full[ercol] - comp_full[mrcol]

comp_full = comp_full[['employee_global_id', 'year', 'competency_group', 'competency_group_type_l1', 
                       'mr_pers_year_comp_score_sum', 'er_pers_year_comp_score_sum', 'mr_pers_year_comp_score_mean',
                       'er_pers_year_comp_score_mean', 'diff_pers_year_comp_score_mean',
                       'mr_pers_compgroup_year_comp_score_mean', 'mr_pers_compgroupl1_year_comp_score_mean', 
                       'er_pers_compgroup_year_comp_score_mean', 'er_pers_compgroupl1_year_comp_score_mean', 
                       'diff_pers_compgroup_year_comp_score_mean', 'diff_pers_compgroupl1_year_comp_score_mean']]
comp_full.drop_duplicates(inplace=True)
comp_full.reset_index(inplace=True, drop=True)

# post-processing and final competency group features added to the instance (some reshaping)
comp_temp_grpl1 = comp_full.copy()
comp_temp_grpl2 = comp_full.copy()
comp_temp_grpl1 = comp_temp_grpl1[['employee_global_id', 'year', 'competency_group', 
                       'mr_pers_compgroup_year_comp_score_mean', 'er_pers_compgroup_year_comp_score_mean',
                                  'diff_pers_compgroup_year_comp_score_mean']]
comp_temp_grpl2 = comp_temp_grpl2[['employee_global_id', 'year', 'competency_group', 'competency_group_type_l1',
                       'mr_pers_compgroupl1_year_comp_score_mean', 'er_pers_compgroupl1_year_comp_score_mean',
                                  'diff_pers_compgroupl1_year_comp_score_mean']]
comp_temp_grpl1.drop_duplicates(inplace=True)
comp_temp_grpl2.drop_duplicates(inplace=True)

comp_temp_grpl1 = comp_temp_grpl1.groupby(['employee_global_id', 'year', 
                                           'competency_group']).sum().unstack('competency_group').reset_index().my_flatten_cols()
comp_full = comp_full.merge(comp_temp_grpl1, how='left', on=['employee_global_id', 'year'])
comp_full.drop(['mr_pers_compgroup_year_comp_score_mean', 'er_pers_compgroup_year_comp_score_mean',
               'diff_pers_compgroup_year_comp_score_mean'], axis=1, inplace=True)
comp_full.drop_duplicates(inplace=True, subset=['employee_global_id', 'year'])
comp_temp_grpl2 = comp_temp_grpl2.groupby(['employee_global_id', 'year', 'competency_group', 
                                          'competency_group_type_l1']).sum().unstack(['competency_group', 
                                                                'competency_group_type_l1']).reset_index().my_flatten_cols()
comp_full = comp_full.merge(comp_temp_grpl2, how='left', on=['employee_global_id', 'year'])
comp_full.drop(['competency_group', 'competency_group_type_l1', 'mr_pers_compgroupl1_year_comp_score_mean', 
                'er_pers_compgroupl1_year_comp_score_mean', 'diff_pers_compgroupl1_year_comp_score_mean'], axis=1, inplace=True)
comp_full.drop_duplicates(inplace=True, subset=['employee_global_id', 'year'])
comp_full.fillna(inplace=True, value=-1)
comp_full.rename({'employee_global_id': 'global_id'}, axis=1, inplace=True)

comp_full.drop(list(comp_full.filter(regex = '_functional_competencies_|_sum')), axis = 1, inplace = True)
##########################################################################################################

compfull = open('E:/ADLS/pickles/competency_16to18_processed.pickle', 'wb')
pickle.dump(comp_full, compfull)
compfull.close()

# # save final processed competency backup
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_16to18_processed.pickle', 'wb') as f:
#     pickle.dump(comp_full, f)
#     f.close()
    
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_16to18_processed.pickle', 'rb') as f:
#     comp_full = pickle.load(f)
#     f.close()

# compfull = open('E:/ADLS/pickles/competency_16to18_processed.pickle', 'rb')
# comp_full = pickle.load(compfull)
# compfull.close()
