## ADS PREPARATION SCRIPT

# datatype coercions (not necessary everywhere, adding for foolproofing)
if comp_full['year'].dtype=='str':
    comp_full['year'] = comp_full['year'].str.extract('(\d+)', expand=False)
comp_full['year'] = comp_full['year'].astype(int)
comp_full['global_id'] = pd.to_numeric(comp_full['global_id'], errors='coerce')
comp_full.dropna(subset=['global_id'], inplace=True)
comp_full['global_id'] = comp_full['global_id'].astype(int)

tar_reshaped['target_year'] = tar_reshaped['target_year'].astype(int)
tar_reshaped['global_id'] = tar_reshaped['global_id'].astype(int)

bp_full['year'] = bp_full['year'].astype(int)
bp_full['global_id'] = bp_full['global_id'].astype(int)
bp_full['target_year'] = bp_full['target_year'].astype(int)
bp_full['prev_year'] = bp_full['year']-1

# adding the previous year features (competency, target)
## competency
comp_full_prev = comp_full.copy()
comp_full_prev = comp_full_prev.add_prefix(prefix='prev_')
comp_full_prev.rename({'prev_global_id':'global_id'}, axis=1, inplace=True)
## target
tar_reshaped_prev = tar_reshaped.copy()
tar_reshaped_prev = tar_reshaped_prev.add_prefix(prefix='prev_')
tar_reshaped_prev.rename({'prev_global_id':'global_id', 'prev_target_year':'target_year'}, axis=1, inplace=True)
tar_reshaped_prev['target_year'] = tar_reshaped_prev['target_year']+1

# ads creation and pre-processing
ads = bp_full.copy()
ads = ads.merge(comp_full, how='left')
ads = ads.merge(tar_reshaped, how='left')
ads = ads.merge(comp_full_prev, how='left')
ads = ads.merge(tar_reshaped_prev, how='left')
ads = ads.merge(belts_grp, how='left')
ads = helpers.process_columns(ads, cols=['position_title', 'employee_band', 'company_code'])
#ads = ads.round(3)
ads = ads.replace([np.inf, -np.inf], np.nan)
ads = ads[ads['employee_band'].isin(psg_bands)]
ads['employee_band'] = ads['employee_band'].map(psg_bands_dict)
ads.reset_index(drop=True, inplace=True)
for i in ['target_year', 'prev_year', 'position_start_date_pa']:
    if i in ads.columns:
        ads.drop(i, inplace=True, axis=1)

# target delta
ads['target_delta'] = ads['prev_net_target'] - ads['net_target']

# position tenure and other related features #####################
## create the monthly flag column and process the date columns available
ads['yearmonth'] = ads['year'].astype(str) + '0701'
ads.position_start_date = pd.to_datetime(ads.position_start_date)
ads['monthly_file_date'] = pd.to_datetime(ads['yearmonth'], format='%Y%m%d')
## treat the position start date
ads = cust_funcs.fillna_df(ads, fill_cols=['position_start_date'], mode='adv_fill', grp_col='global_id')
ads['position_start_date'] = np.where(ads['position_start_date'] > ads['monthly_file_date'], 
                                         ads['monthly_file_date'], ads['position_start_date'])
ads['position_tenure'] = (ads.monthly_file_date - ads.position_start_date).astype('timedelta64[D]')
## time in band
ads.sort_values(['global_id', 'monthly_file_date'], axis=0, inplace=True, kind='mergesort')
ads.reset_index(drop=True, inplace=True)
ads = pd.merge(ads, band_changes, on=['global_id', 'employee_band'], how='left')
ads['time_in_band'] = (ads.monthly_file_date - ads.start_date).astype('timedelta64[D]')
ads.drop(['monthly_file_date', 'yearmonth', 'start_date'], inplace=True, axis=1)
##################################################################

# creating the various date features
ads = cust_funcs.datetime_feats(ads)

# force numeric the ads
ads = cust_funcs.force_numeric(ads, cols=['global_id', 'direct_manager_emp_id'])

# merge the teamsize and teamsize_diff dataframe
ads = ads.merge(teamsize_and_diff_df, on=['direct_manager_emp_id', 'year'], how='left')

# merge the career velocity dataframes
ads = ads.merge(ia_full, on=['global_id', 'year'], how='left')
ads = ads.merge(cv_full, on=['global_id', 'year'], how='left')
ads = ads.merge(pv_full, on=['global_id', 'year'], how='left')

# merge the opr datasets
train = opr_train.merge(ads, how='inner', on=['global_id', 'year'])
valid = opr_valid.merge(ads, how='inner', on=['global_id', 'year'])

# merge the demo datasets
train = train.merge(train_demo, how='left', on=['global_id', 'year'])
train = train.merge(train_demo_mng, how='left', on=['direct_manager_emp_id', 'year'])
valid = valid.merge(valid_demo, how='left', on=['global_id', 'year'])
valid = valid.merge(valid_demo_mng, how='left', on=['direct_manager_emp_id', 'year'])
## create the tenure and age difference features as well
train['emp_mngr_tenure_diff'] = train.emp_tenure_asof_current - train.mngr_tenure_asof_current
train['emp_mngr_dob_diff'] = train.emp_age_asof_current - train.mngr_age_asof_current
valid['emp_mngr_tenure_diff'] = valid.emp_tenure_asof_current - valid.mngr_tenure_asof_current
valid['emp_mngr_dob_diff'] = valid.emp_age_asof_current - valid.mngr_age_asof_current

zone_info = helpers.csv_read('../working/zone_info.csv', cols_to_keep=['macro_entity_l2_code', 'zone'])
zone_info.drop_duplicates(inplace=True)
#zone_info.dropna(inplace=True)

train['macro_entity_l2_code'] = train['macro_entity_l2_code'].astype(int)
valid['macro_entity_l2_code'] = valid['macro_entity_l2_code'].astype(int)

train = train.merge(zone_info, how='left', on=['macro_entity_l2_code'])
valid = valid.merge(zone_info, how='left', on=['macro_entity_l2_code'])
train.dropna(subset=['zone'], inplace=True)
valid.dropna(subset=['zone'], inplace=True)

#################### the final merges ##################################
## for train
train = pd.merge(train, gm_train, on=['global_id', 'year'], how='left')
train = pd.merge(train, tp_train, on=['global_id', 'year'], how='left')
train = pd.merge(train, pdp_train, on=['global_id', 'year'], how='left')
train = pd.merge(train, pdi_train, on=['global_id', 'year'], how='left')
#train = pd.merge(train, mngrkpis_train, on=['global_id', 'year'], how='left')
train = pd.merge(train, movm_train, on=['global_id', 'year'], how='left')
#train = pd.merge(train, normca_train, on=['global_id', 'year'], how='left')
train = pd.merge(train, careerasp_train, on=['global_id', 'year'], how='left')
train = pd.merge(train, band_changes_df_train_agg3, on=['global_id', 'year'], how='left')
#train = pd.merge(train, mngr_lvl_orgfeats_train, on=['global_id', 'year'], how='left')
#train = pd.merge(train, team_lvl_orgfeats_train, on=['global_id', 'year'], how='left')
train = pd.merge(train, mobility, on=['global_id'], how='left')
train = pd.merge(train, tptrainrenom, on=['global_id', 'year'], how='left')
train = pd.merge(train, engfull_train, on=['global_id', 'year'], how='left')

## for valid
valid = pd.merge(valid, gm_valid, on=['global_id', 'year'], how='left')
valid = pd.merge(valid, tp_valid, on=['global_id', 'year'], how='left')
valid = pd.merge(valid, pdp_valid, on=['global_id', 'year'], how='left')
valid = pd.merge(valid, pdi_valid, on=['global_id', 'year'], how='left')
#valid = pd.merge(valid, mngrkpis_valid, on=['global_id', 'year'], how='left')
valid = pd.merge(valid, movm_valid, on=['global_id', 'year'], how='left')
#valid = pd.merge(valid, normca_valid, on=['global_id', 'year'], how='left')
valid = pd.merge(valid, careerasp_valid, on=['global_id', 'year'], how='left')
valid = pd.merge(valid, band_changes_df_valid_agg3, on=['global_id', 'year'], how='left')
#valid = pd.merge(valid, mngr_lvl_orgfeats_valid, on=['global_id', 'year'], how='left')
#valid = pd.merge(valid, team_lvl_orgfeats_valid, on=['global_id', 'year'], how='left')
valid = pd.merge(valid, mobility, on=['global_id'], how='left')
valid = pd.merge(valid, tpvalidrenom, on=['global_id', 'year'], how='left')
valid = pd.merge(valid, engfull_valid, on=['global_id', 'year'], how='left')


train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)

# split the response
ytrain = train.response
yvalid = valid.response
train.drop('response', inplace=True, axis=1)
valid.drop('response', inplace=True, axis=1)

train.drop(columns=['macro_entity_l2_code', 'date_of_birth_dayofyear', 'date_of_birth_month', 'date_of_birth_quarter', 
       'date_of_birth_year', 'original_hire_date_dayofyear',
       'original_hire_date_month', 'original_hire_date_quarter',
       'original_hire_date_year', 'position_start_date_month', 'position_start_date_quarter',
       'position_start_date_year'], inplace=True)
valid.drop(columns=['macro_entity_l2_code', 'date_of_birth_dayofyear', 'date_of_birth_month', 'date_of_birth_quarter', 
       'date_of_birth_year', 'original_hire_date_dayofyear',
       'original_hire_date_month', 'original_hire_date_quarter',
       'original_hire_date_year', 'position_start_date_month', 'position_start_date_quarter',
       'position_start_date_year'], inplace=True)

train.drop(['emp_tenure_asof_current', 'mngr_tenure_asof_current', 'emp_mngr_tenure_diff', 
            'emp_mngr_dob_diff', 'emp_age_asof_current', 'mngr_age_asof_current'], axis=1, inplace=True)
valid.drop(['emp_tenure_asof_current', 'mngr_tenure_asof_current', 'emp_mngr_tenure_diff', 
            'emp_mngr_dob_diff', 'emp_age_asof_current', 'mngr_age_asof_current'], axis=1, inplace=True)

train.drop(list(train.filter(regex = 'prev_er|prev_mr|prev_diff|diff')), axis = 1, inplace = True)
valid.drop(list(valid.filter(regex = 'prev_er|prev_mr|prev_diff|diff')), axis = 1, inplace = True)

# target delta feature
train['target_delta'] = train['net_target'] - train['prev_net_target']
valid['target_delta'] = valid['net_target'] - valid['prev_net_target']

ytrain = np.array(ytrain)
yvalid = np.array(yvalid)

# save backup
ads_backup = open('../working/ads_backup.pkl','wb')
pickle.dump(train, ads_backup)
pickle.dump(valid, ads_backup)
pickle.dump(ytrain, ads_backup)
pickle.dump(yvalid, ads_backup)
ads_backup.close()
