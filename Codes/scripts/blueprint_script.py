# def download_blueprint_files_from_adls_folder(adls, path):
#     files = adls.ls(path)
#     bp_pattern = re.compile('blueprint_')
#     files_list = [s for s in files if bp_pattern.search(s)]
#     filesdict = {}
#     for i in tqdm(files_list):
#         file_year = i.split('/')[-1].split('.')[0].split('_')[-1]
#         with adls.open(i) as f:
#             filesdict[file_year] = helpers.xlsx_read(file_path=f, drop_dup='yes', dtype='object')
#             filesdict[file_year]['year'] = int(file_year)
#     df = pd.concat(filesdict.values(), ignore_index=True)
#     df.dropna(subset=['global_id', 'employee_band'], inplace=True)
#     return df
# bp_full = download_blueprint_files_from_adls_folder(adlsfsc, path + '/2019/Data/Raw_Data/sharp/blueprints') 

##########################################################################################################

# # save interim raw blueprint backup
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Blueprint/bp_backup_16to19_raw.pickle', 'wb') as f:
#     pickle.dump(bp_full, f)
#     f.close()
    
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Blueprint/bp_backup_16to19_raw.pickle', 'rb') as f:
#     bp_full = pickle.load(f)
#     f.close()

##########################################################################################################
# bpfull = open('E:/ADLS/pickles/bp_backup_16to19_raw.pickle', 'wb')
# pickle.dump(bp_full, bpfull)
# bpfull.close()

bpfull = open('E:/ADLS/pickles/bp_backup_16to19_raw.pickle', 'rb')
bp_full = pickle.load(bpfull)
bpfull.close()
##########################################################################################################

bp_full['employment_status'] = np.where(bp_full['global_id'].isin(inactive_to_active_ids),
                                        'Active',
                                       bp_full['employment_status'])
bp_full = bp_full[bp_full['employment_status']=='Active']
bp_full = bp_full[bp_cols_to_keep]
bp_full['missing']=bp_full.isnull().sum(axis=1)
bp_full.sort_values(['global_id', 'year', 'missing'], ascending=True, inplace=True)
bp_full.reset_index(drop=True, inplace=True)
bp_full.drop_duplicates(subset=['global_id', 'year'], inplace=True, keep='first')
bp_full.drop(columns=['missing'], inplace=True)
bp_full = cust_funcs.force_numeric(bp_full, cols=['global_id', 'direct_manager_emp_id', 'position_id', 'ebm_level'])

bp_full['date_of_birth'] = pd.to_datetime(bp_full.date_of_birth, format='%m/%d/%Y', errors='coerce')
bp_full['original_hire_date'] = pd.to_datetime(bp_full.original_hire_date, format='%Y-%m-%d', errors='coerce')
bp_full['position_start_date'] = pd.to_datetime(bp_full.position_start_date_pa, format='%Y-%m-%d', errors='coerce')

bp_full['target_year'] = bp_full['year']-1

bp_full['ebm_level'].fillna(-1, inplace=True)
bp_full['ebm_level'] = bp_full['ebm_level'].astype(int)
bp_full['ebm_level'] = np.where(bp_full['ebm_level'].isin([1,2,3]), bp_full['ebm_level'], -1)
##########################################################################################################

# function
with adlsfsc.open(path + '/2019/Data/Transformed_Data/ml_helper_files/function_info.csv') as f:
    function_info = helpers.csv_read(f, cols_to_keep=['inbev_entity_l2_desc', 'function'])
bp_full = bp_full.merge(function_info, how='left', on='inbev_entity_l2_desc')
bp_full['function'] = np.where(bp_full['inbev_entity_l3_desc']=='LOGISTICS', 'logistics', bp_full['function'])
##########################################################################################################

# fix manager id information
for i in ['direct_manager_emp_id', 'hierarchy_manager_emp_id', 'parent_org_unit_manager_personnel_no']:
    bp_full[i] = bp_full[i].replace(0, np.NAN)
    
bp_full['direct_manager_emp_id'] = np.where(bp_full['direct_manager_emp_id'].isna,
                                        bp_full['hierarchy_manager_emp_id'],
                                        bp_full['direct_manager_emp_id'])
bp_full['direct_manager_emp_id'] = np.where(bp_full['direct_manager_emp_id'].isna,
                                        bp_full['parent_org_unit_manager_personnel_no'],
                                        bp_full['direct_manager_emp_id'])

bp_full['direct_manager_emp_id'].fillna(value=0, inplace=True)
bp_full.drop(columns=['hierarchy_manager_emp_id', 'parent_org_unit_manager_personnel_no'], inplace=True)
##########################################################################################################

# fix the dates (hire_date, position_date, DOB)
bp_full['original_hire_date'] = np.where(bp_full['original_hire_date']>bp_full['position_start_date'],
                                        bp_full['position_start_date'], bp_full['original_hire_date'])

# save the processed bp_full backup
with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Blueprint/bp_backup_16to19_processed.pickle', 'wb') as f:
    pickle.dump(bp_full, f)
    f.close()

print(bp_full.year.value_counts())

bpfull = open('E:/ADLS/pickles/bp_backup_16to19_processed.pickle','wb')
pickle.dump(bp_full, bpfull)
bpfull.close()
