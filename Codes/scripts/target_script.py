## target script

with adlsfsc.open(path + '/2019/Data/Raw_Data/miscellaneous/target/TA_16_to_18.xlsx', 'rb') as f:
    target_df = helpers.xlsx_read(f)

target_df = cust_funcs.force_numeric(target_df, cols=target_df.columns)
if 'personnel_number' in target_df.columns:
    target_df.drop(columns=['personnel_number'], inplace=True)
target_df.set_index('global_id', inplace=True)
tar_reshaped = target_df.unstack().reset_index()
tar_reshaped.columns = ['target_year', 'global_id', 'net_target']
tar_reshaped['target_year'] = tar_reshaped['target_year'].str.split('_').str[-1]

with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Miscellaneous/target_backup.pickle', 'wb') as f:
    pickle.dump(tar_reshaped, f)
    f.close()

tarpkl = open('E:/ADLS/pickles/target_backup.pickle', 'wb')
pickle.dump(tar_reshaped, tarpkl)
tarpkl.close()
