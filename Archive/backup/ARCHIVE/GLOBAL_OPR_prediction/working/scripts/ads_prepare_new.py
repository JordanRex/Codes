## ADS PREPARATION SCRIPT

#############################################################################################
### blueprint pre-processing
bp_full = cust_funcs.force_numeric(df=bp_full,
                                  cols=['global_id', 'direct_manager_emp_id', 'year', 'target_year', 'ebm_level'])
bp_full.dropna(subset=['global_id', 'year'], how='any', inplace=True)
bp_full['prev_year'] = bp_full['year']-1
bp_full['talentpool_year'] = bp_full['year']-1
bp_full['engagement_year'] = bp_full['year']-1
bp_full['pdp_year'] = bp_full['year']-1
bp_full['mngrkpi_year'] = bp_full['year']-1
for i in ['global_id', 'year', 'prev_year', 'target_year', 'talentpool_year','ebm_level', 'engagement_year', 'pdp_year', 'mngrkpi_year']:
    bp_full[i] = bp_full[i].astype(int)
#############################################################################################

#############################################################################################
### datatype coercions and other pre-processing

# for competency
comp_full = cust_funcs.force_numeric(df=comp_full, cols=['global_id', 'year'])
comp_full.dropna(subset=['global_id', 'year'], inplace=True, how='any')
comp_full['global_id'] = comp_full['global_id'].astype(int)
comp_full['year'] = comp_full['year'].astype(int)
comp_full.dropna(inplace=True, subset=['global_id'])
comp_full = comp_full[['global_id', 'year',
       'mr_pers_compgroup_year_comp_score_mean_leadership_competencies',
        'mr_pers_compgroup_year_comp_score_mean_functional_competencies',
       'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_develop_people',
       'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_dream_big',
       'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_live_our_culture']]
comp_full = pd.concat([comp_full, comp2019], axis=0, ignore_index=True)
comp_full.drop_duplicates(inplace=True)

# for target
tar_reshaped = cust_funcs.force_numeric(df=tar_reshaped)
tar_reshaped.dropna(subset=['global_id', 'target_year'], inplace=True, how='any')
tar_reshaped['global_id'] = tar_reshaped['global_id'].astype(int)
tar_reshaped['target_year'] = tar_reshaped['target_year'].astype(int)
tar_reshaped.dropna(inplace=True)

# for opr
opr_reshaped = cust_funcs.force_numeric(opr_reshaped)
opr_reshaped.dropna(subset=['global_id', 'year'], inplace=True, how='any')
opr_reshaped['global_id'] = opr_reshaped['global_id'].astype(int)
opr_reshaped['year'] = opr_reshaped['year'].astype(int)
opr_reshaped.dropna(inplace=True)

# for engagement
eng_es = engfull[engfull['eng_col']=='engagement_score']
eng_me = engfull[engfull['eng_col']=='manager_effectiveness_score']
eng_es.drop(columns='eng_col', inplace=True)
eng_me.drop(columns='eng_col', inplace=True)
eng_es.drop_duplicates(subset=['global_id', 'year'], inplace=True)
eng_me.drop_duplicates(subset=['global_id', 'year'], inplace=True)
eng_es.columns = ['global_id', 'engagement_score', 'engagement_year']
eng_me.columns = ['global_id', 'manager_effectiveness_score', 'engagement_year']

# for pdp
pdp_full.rename(columns={'year':'pdp_year'}, inplace=True)
#############################################################################################

#############################################################################################
### ads creation

ads = bp_full.copy()
ads = helpers.process_columns(ads, cols=['employee_band'])
ads = ads[ads['employee_band'].isin(psg_bands_0to5)]
ads['employee_band'] = ads['employee_band'].map(psg_bands_dict)
ads = ads.merge(comp_full, how='left', on=['global_id', 'year'])
ads = ads.merge(tar_reshaped, how='left', on=['global_id', 'target_year'])
ads = ads.merge(opr_reshaped, how='left', on=['global_id', 'year'])
ads = ads.merge(teamsize_and_diff_df, how='left', on=['direct_manager_emp_id', 'year'])
ads = ads.merge(cv_full, on=['global_id','year'], how='left')
ads = ads.merge(pv_full, on=['global_id', 'year'], how='left')
ads = ads.merge(tib_full, on=['global_id', 'year'], how='left')
ads = ads.merge(belts_grp, how='left', on=['global_id', 'year'])
# ads = ads.merge(gmt, how='left', on=['global_id', 'year'])
# ads = ads.merge(gmba, how='left', on=['global_id', 'year'])
ads = ads.merge(tp_full, how='left', on=['global_id', 'talentpool_year'])
ads = ads.merge(eng_es, how='left', on=['global_id', 'engagement_year'])
ads = ads.merge(eng_me, how='left', on=['global_id', 'engagement_year'])
ads = ads.merge(fs_prom, how='left', on=['global_id', 'mngrkpi_year'])
ads = ads.merge(fs_ho, how='left', on=['global_id', 'mngrkpi_year'])
ads = ads.merge(fs_adherant_perc, how='left', on=['global_id', 'mngrkpi_year'])
ads = ads.merge(fs_to_overall, how='left', on=['global_id', 'mngrkpi_year'])
ads = ads.merge(dr_prom, how='left', on=['global_id', 'mngrkpi_year'])
ads = ads.merge(dr_ho, how='left', on=['global_id', 'mngrkpi_year'])
ads = ads.merge(dr_adherant_perc, how='left', on=['global_id', 'mngrkpi_year'])
ads = ads.merge(dr_to_overall, how='left', on=['global_id', 'mngrkpi_year'])
ads = ads.merge(tt_full, how='left', on=['global_id', 'year'])
ads = ads.merge(pdp_full, how='left', on=['global_id', 'pdp_year'])
ads = ads.replace([np.inf, -np.inf], np.nan)
ads.reset_index(drop=True, inplace=True)

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

# creating the various date features
#ads = cust_funcs.datetime_feats(ads)

#############################################################################################

with adlsfsc.open(path + '/2019/Data/Transformed_Data/ml_helper_files/zone_info.csv') as f:
    zone_info = helpers.csv_read(f, cols_to_keep=zonefile_cols)
zone_info.drop_duplicates(inplace=True)

ads['macro_entity_l2_code'] = ads['macro_entity_l2_code'].astype(int)
ads = ads.merge(zone_info, how='left', on=['macro_entity_l2_code'])
print('ads size before zone filter: ', ads.shape)
ads.dropna(subset=['zone'], inplace=True)
print('ads size after zone filter: ', ads.shape)

# add target delta
prev_target = tar_reshaped.copy()
prev_target.columns = ['target_year', 'global_id', 'prev_net_target']
ads['target_year'] = ads['year']-2
ads = ads.merge(prev_target, on=['global_id', 'target_year'], how='left')
ads['target_delta'] = ads['net_target'] - ads['prev_net_target']

# add previous opr
prev_opr = opr_reshaped.copy()
prev_opr['year'] = prev_opr['year']+1
prev_opr.rename(columns={'opr':'prev_opr'}, inplace=True)
prev_prev_opr = opr_reshaped.copy()
prev_prev_opr['year'] = prev_prev_opr['year']+2
prev_prev_opr.rename(columns={'opr':'prev_prev_opr'}, inplace=True)
ads = ads.merge(prev_opr, on=['global_id', 'year'], how='left')
ads = ads.merge(prev_prev_opr, on=['global_id', 'year'], how='left')

for i in ['target_year', 'prev_year', 'position_start_date_pa', 'talentpool_year', 'position_start_date',
          'macro_entity_id', 'macro_entity_l1_code', 'macro_entity_l2_code',
          'macro_entity_l3_code', 'macro_entity_l4_code', 'local_entity_id',
          'local_entity_l1_code', 'local_entity_l2_code', 'local_entity_l3_code',
          'local_entity_l4_code', 'ab_inbev_entity_id', 'inbev_entity_l2_desc',
          'inbev_entity_l3_desc', 'analysis_block_id', 'analysis_block_l1_code',
          'analysis_block_l2_code', 'analysis_block_l3_code', 'analysis_block_l4_code', 
          'annual_salary', 'currency_key', 'pay_scale_type', 'pay_scale_level', 'pay_scale_area_code', 'country_grouping_p_',
         'position_id', 'direct_manager_emp_id', 'position_title',
         'date_of_birth', 'original_hire_date', 'functional_area_id', 'global_job_code', 'job_family_code',
       'talent_type', 'country_code', 'physical_work_location_code',
       'company_code', 'personnel_area', 'personnel_subarea', 'yearmonth', 'monthly_file_date', 'emp_careervelocity_axis_band1',
         'mrs', 'compare_ratio', 'mod_salary', 'prev_net_target', 'engagement_year', 'pdp_year', 'mngrkpi_year']:
    if i in ads.columns:
        ads.drop(i, inplace=True, axis=1)

ads['prev_opr'] = ads['prev_opr']-1
ads['prev_opr'] = np.where(ads['prev_opr']<0, 0, ads['prev_opr'])
ads['prev_prev_opr'] = ads['prev_prev_opr']-1
ads['prev_prev_opr'] = np.where(ads['prev_prev_opr']<0, 0, ads['prev_prev_opr'])
