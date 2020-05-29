## opr script

### reading the input files from ADLS
def download_opr_files_from_adls_folder(adls, path):
    files_list = adls.ls(path)
    filesdict = {}
    for i in tqdm(files_list):
        file_year = i.split('/')[-1].split('.')[0].split('_')[-1]
        with adls.open(i) as f:
            filesdict[file_year] = helpers.csv_read(file_path=f, drop_dup='yes')
    df = pd.concat(filesdict.values(), ignore_index=True)
    return df
opr_full = download_opr_files_from_adls_folder(adlsfsc, path + '/2019/Data/Raw_Data/navigate/opr')

### pre-processing the input files and appending them. not dynamic since cadence/structure can change
required_cols = ['employee_global_id', 'year', 'opr_rating_scale']
opr_full = opr_full[required_cols]

opr_full.columns = ['global_id', 'year', 'opr']
opr_full.drop_duplicates(inplace=True, subset=['global_id', 'year'])
opr_full.dropna(how='any', inplace=True)
opr_full = opr_full[opr_full['year'] > 2013]
opr_full.reset_index(inplace=True, drop=True)
opr_full = opr_full[opr_full['opr']!='2']
opr_full['opr'] = opr_full['opr'].map(dep_dict)

### reshaping and creating the pivot version
opr_reshaped = opr_full.pivot(index='global_id', columns='year', values=['opr']).reset_index().my_flatten_cols()
opr_reshaped.columns.name = None
opr_reshaped[['opr_2016', 'opr_2017', 'opr_2018']] = opr_reshaped[['opr_2016', 'opr_2017', 'opr_2018']].apply(pd.to_numeric, errors='coerce')
opr_reshaped = helpers.process_columns(df=opr_reshaped)

opr_reshaped.set_index('global_id', inplace=True)
opr_reshaped = opr_reshaped.unstack().reset_index()
opr_reshaped.columns = ['year', 'global_id', 'opr']
opr_reshaped['year'] = opr_reshaped['year'].str.split('_').str[-1]

with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Miscellaneous/opr_backup_17to18.pickle', 'wb') as f:
    pickle.dump(opr_reshaped, f)
    f.close()

### checks
#(opr_full[opr_full.duplicated(['global_id', 'year'], keep=False)]).shape # should yield zero (uncomment drop duplicates to test)
