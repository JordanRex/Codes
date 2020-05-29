## dictionaries and columns to be subset at each place

##########################################################################################################
# ADHOC active status tweaking for people inactive in blueprint
inactive_to_active_ids = [99733692,10707493,2000060,13046171,41014334,99731672,10307024,10700898]
##########################################################################################################

############################ global dictionaries and lists #####################################################
dep_dict = {'4A': 5, 
            '4B': 4, 
            '3A': 3, 
            '3B': 2, 
            '1A': 1, 
            '1B': 0}
rev_dep_dict_with1B = {0:'1B', 1:'1A', 2:'3B', 3:'3A', 4:'4B', 5:'4A'}
dep_dict_without1B = {'4A':4, '4B':3, '3A':2, '3B':1, '1A':0}
rev_dep_dict_without1B = {4:'4A', 3:'4B', 2:'3A', 1:'3B', 0:'1A'}

psg_bands = ['vi_b', 'vi_a', 'vii_b', 'vii_a', 'v_b', 'v_a', 'iv_b', 'iv_a', 'iii_b', 
             'iii_a', 'ii_b', 'i_b', 'ii_a', 'i_a', '0_b', 'ebm']
psg_bands_0to5 = ['v_b', 'v_a', 'iv_b', 'iv_a', 'iii_b', 'iii_a', 'ii_b', 'i_b', 'ii_a', 'i_a', '0_b', 'ebm']

psg_bands_dict = {'ebm':0,
                  '0_b':1, 
                  'i_a':2, 'i_b':3, 
                  'ii_a':4, 'ii_b':5, 
                  'iii_a':6, 'iii_b':7, 
                  'iv_a':8, 'iv_b':9, 
                  'v_a':10, 'v_b':11,
                  'vi_a':12, 'vi_b':13,
                  'vii_a':14, 'vii_b':15}

curr_dict = {'eur':1.13, 'chf':1, 'cad':0.76, 'usd':1, 'gbp':1.3, 'usd4':1, 'aud':0.71, 'czk':0.044,
       'huf':0.0036, 'jpy':0.009, 'sek':0.11, 'cny':0.15, 'ars':0.025, 'eur4':1.13, 'mxn':0.052, 'rub':0.015, 'uah':0.037,
       'cop':0.00032, 'clp':0.0015, 'inr':0.014, 'krw':0.00089, 'vnd':0.000043, 'uyu':0.031, 'pyg':0.00016, 'bob':0.14, 'pen':0.3,
       'gtq':0.13, 'ecs':1, 'dop':0.02, 'brl':0.27, 'aud5':0.71, 'zar':0.071, 'nad':0.071, 'bwp':0.095, 'mzm':0.016,
       'svc':0.11, 'ghc':0.19, 'ugx':0.00027, 'zmk':0.084, 'mwk':0.0014, 'ngn':0.0028, 'tzs':0.00043, 'mur':0.029, 'pab':1,
       'kzt':0.0027, 'nzd':0.68, 'hnl':0.041}

############################## columns to subset ##################################################################
bp_nlp_cols = ['company_name', 'contract_text', 'cost_center_description', 'functional_area_name', 'global_job_description', 
               'inbev_description', 'inbev_entity_l2_desc', 'inbev_entity_l3_desc', 'inbev_entity_l4_desc', 
               'job_family_description', 'macro_entity_desc', 'macro_entity_l2_desc', 'macro_entity_l3_desc', 
               'macro_entity_l4_desc', 'macro_entity_l5_desc', 'macro_entity_l6_desc', 'org_unit_description', 
               'pay_scale_area_text', 'pers__subarea_text', 'personnel_area_text', 'position_type_text']

bp_cols_to_keep = ['global_id', # employee unique id
                   'position_id', # position unique id
                   'direct_manager_emp_id', 'hierarchy_manager_emp_id', 'parent_org_unit_manager_personnel_no', # manager employee unique ids (3 of them)
                   'position_title', 'position_start_date_pa',
                   'employee_band', # salary band
                   'date_of_birth', 'original_hire_date', # dates for age and tenure 
                   'ebm_level', # ebm level feature for filtering
                   'macro_entity_l1_code', 'macro_entity_l2_code',
                   'inbev_entity_l2_desc', 'inbev_entity_l3_desc',
                   'annual_salary', 'currency_key', 'pay_scale_type', 'pay_scale_level', 'pay_scale_area_code', 'country_grouping_p_',
                   'year']

# read in the competency files for competency
cp_cols_to_keep = ['employee_global_id',
'competency_group',
'competency',
'employee_rating_numeric_value',
'manager_rating_numeric_value']

# belts columns
belts_cols = ['employee_global_id', 'certification_type', 'new_certification_date_dd_mm_yyyy_']

# engagement columns
eng_cols = ['globalid', 
            '2017_engagement_pa', '2016_engagement_pa', 
            '2017_performance_enablement_pa', '2016_performance_enablement_pa', 
            '2017_direct_report_performance_enablement_pa', '2016_direct_report_performance_enablement_pa']

# zone mapping file columns
zonefile_cols = ['macro_entity_l2_code', 'zone']
