# movements script

#########################################################################################################################################
### CAREER VELOCITY

# career velocity files
cv_cols = ['global_id', 'index_average', 'year']
def download_cv_files_from_adls_folder(adls, path, cols):
    files = adls.ls(path)
    cv_pattern = re.compile('cv_')
    files_list = [s for s in files if cv_pattern.search(s)]
    filesdict = {}
    for i in tqdm(files_list):
        file_year = i.split('/')[-1].split('.')[0].split('_')[-1]
        with adls.open(i) as f:
            filesdict[file_year] = helpers.csv_read(file_path=f, drop_dup='yes', cols_to_keep=cols)
    df = pd.concat(filesdict.values(), ignore_index=True)
    return df
cv_full = download_cv_files_from_adls_folder(adlsfsc, path + '/2019/Data/Output_Data/ml_features/career_velocity/', cols=cv_cols)

# position velocity files
pv_cols = ['global_id', 'position_velocity', 'year']
def download_pv_files_from_adls_folder(adls, path, cols):
    files = adls.ls(path)
    pv_pattern = re.compile('pv_')
    files_list = [s for s in files if pv_pattern.search(s)]
    filesdict = {}
    for i in tqdm(files_list):
        file_year = i.split('/')[-1].split('.')[0].split('_')[-1]
        with adls.open(i) as f:
            filesdict[file_year] = helpers.csv_read(file_path=f, drop_dup='yes', cols_to_keep=cols)
    df = pd.concat(filesdict.values(), ignore_index=True)
    return df
pv_full = download_pv_files_from_adls_folder(adlsfsc, path + '/2019/Data/Output_Data/ml_features/career_velocity/', cols=pv_cols)

# time in band files
tib_cols = ['global_id', 'emp_careervelocity_axis_band1', 'emp_time_in_band1', 'year']
def download_tib_files_from_adls_folder(adls, path, cols):
    files = adls.ls(path)
    tib_pattern = re.compile('last')
    files_list = [s for s in files if tib_pattern.search(s)]
    filesdict = {}
    for i in tqdm(files_list):
        file_year = i.split('/')[-1].split('.')[0].split('_')[-1]
        with adls.open(i) as f:
            filesdict[file_year] = helpers.csv_read(file_path=f, drop_dup='yes', cols_to_keep=cols)
    df = pd.concat(filesdict.values(), ignore_index=True)
    return df
tib_full = download_tib_files_from_adls_folder(adlsfsc, path + '/2019/Data/Output_Data/ml_features/career_velocity/', cols=tib_cols)

with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Movements/career_velocity.pkl', 'wb') as f:
    pickle.dump(cv_full, f)
    pickle.dump(pv_full, f)
    pickle.dump(tib_full, f)
    f.close()
