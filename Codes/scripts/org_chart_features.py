# manager FS / manager DR summary kpis / team tenure

# manager FS
with adlsfsc.open(path + '/2019/Data/Output_Data/ml_features/org_chart_features/FS_manager_summary_all.csv', 'rb') as f:
    mngrkpis_fs = helpers.csv_read(f)

mngrkpis_1 = mngrkpis_fs.filter(regex = '^ho_jul_|^prom_jul_|^to_overall_jul_\d{4}$|adherent_percentage_jul|global')
mngrkpis_1 = mngrkpis_1.filter(regex = '16|17|18|id')
mngrkpis_1.rename(columns={'iglobalid':'global_id'}, inplace=True)

mngrkpis_1.set_index('global_id', inplace=True)
mngrkpis_1 = mngrkpis_1.reset_index()
mngrkpis_1 = pd.melt(mngrkpis_1, id_vars='global_id', value_vars=['ho_jul_2016', 'ho_jul_2017', 'ho_jul_2018',
       'to_overall_jul_2016', 'to_overall_jul_2017', 'to_overall_jul_2018',
       'adherent_percentage_jul_2016', 'adherent_percentage_jul_2017',
       'adherent_percentage_jul_2018', 'prom_jul_2016', 'prom_jul_2017',
       'prom_jul_2018'])
mngrkpis_1['FS_col'] = mngrkpis_1['variable'].str.split('_2').str[0]
mngrkpis_1['mngrkpi_year'] = mngrkpis_1['variable'].str.split('_').str[-1].astype(int)
mngrkpis_fs = mngrkpis_1[['global_id', 'FS_col', 'mngrkpi_year', 'value']]

################################################################################################

# manager DR
with adlsfsc.open(path + '/2019/Data/Output_Data/ml_features/org_chart_features/DR_manager_summary_all.csv', 'rb') as f:
    mngrkpis_dr = helpers.csv_read(f)

mngrkpis_2 = mngrkpis_dr.filter(regex = '^ho_jul_|^prom_jul_|^to_overall_jul_\d{4}$|adherent_percentage_jul|global')
mngrkpis_2 = mngrkpis_2.filter(regex = '16|17|18|id')
mngrkpis_2.rename(columns={'iglobalid':'global_id'}, inplace=True)

mngrkpis_2.set_index('global_id', inplace=True)
mngrkpis_2 = mngrkpis_2.reset_index()
mngrkpis_2 = pd.melt(mngrkpis_2, id_vars='global_id', value_vars=['ho_jul_2016', 'ho_jul_2017', 'ho_jul_2018',
       'to_overall_jul_2016', 'to_overall_jul_2017', 'to_overall_jul_2018',
       'adherent_percentage_jul_2016', 'adherent_percentage_jul_2017',
       'adherent_percentage_jul_2018', 'prom_jul_2016', 'prom_jul_2017',
       'prom_jul_2018'])
mngrkpis_2['DR_col'] = mngrkpis_2['variable'].str.split('_2').str[0]
mngrkpis_2['mngrkpi_year'] = mngrkpis_2['variable'].str.split('_').str[-1].astype(int)
mngrkpis_dr = mngrkpis_2[['global_id', 'DR_col', 'mngrkpi_year', 'value']]

################################################################################################

# team tenure
def download_tenure_files_from_adls_folder(adls, path):
    files = adls.ls(path)
    tt_pattern = re.compile('team_level_')
    files_list = [s for s in files if tt_pattern.search(s)]
    filesdict = {}
    for i in tqdm(files_list):
        file_year = i.split('/')[-1].split('.')[0].split('_')[-1]
        with adls.open(i) as f:
            filesdict[file_year] = helpers.csv_read(file_path=f, drop_dup='yes')
            filesdict[file_year]['year'] = int(file_year)
    df = pd.concat(filesdict.values(), ignore_index=True)
    return df
tt_full = download_tenure_files_from_adls_folder(adlsfsc, path + '/2019/Data/Output_Data/ml_features/org_chart_features/')

tt_full = tt_full[['rglobalid', 'year', 'lm_and_dr_level_level_mean_tip']]
tt_full.columns = ['global_id', 'year', 'mean_team_tenure']

################################################################################################

fs_prom = mngrkpis_fs[mngrkpis_fs['FS_col']=='prom_jul']
fs_ho = mngrkpis_fs[mngrkpis_fs['FS_col']=='ho_jul']
fs_to_overall = mngrkpis_fs[mngrkpis_fs['FS_col']=='to_overall_jul']
fs_adherant_perc = mngrkpis_fs[mngrkpis_fs['FS_col']=='adherent_percentage_jul']

dr_prom = mngrkpis_dr[mngrkpis_dr['DR_col']=='prom_jul']
dr_ho = mngrkpis_dr[mngrkpis_dr['DR_col']=='ho_jul']
dr_to_overall = mngrkpis_dr[mngrkpis_dr['DR_col']=='to_overall_jul']
dr_adherant_perc = mngrkpis_dr[mngrkpis_dr['DR_col']=='adherent_percentage_jul']

fs_prom.drop(columns=['FS_col'], inplace=True)
fs_ho.drop(columns=['FS_col'], inplace=True)
fs_to_overall.drop(columns=['FS_col'], inplace=True)
fs_adherant_perc.drop(columns=['FS_col'], inplace=True)

dr_prom.drop(columns=['DR_col'], inplace=True)
dr_ho.drop(columns=['DR_col'], inplace=True)
dr_to_overall.drop(columns=['DR_col'], inplace=True)
dr_adherant_perc.drop(columns=['DR_col'], inplace=True)

fs_prom.columns = ['global_id', 'mngrkpi_year', 'fs_prom']
fs_ho.columns = ['global_id', 'mngrkpi_year', 'fs_ho']
fs_to_overall.columns = ['global_id', 'mngrkpi_year', 'fs_to_overall']
fs_adherant_perc.columns = ['global_id', 'mngrkpi_year', 'fs_adherant_perc']

dr_prom.columns = ['global_id', 'mngrkpi_year', 'dr_prom']
dr_ho.columns = ['global_id', 'mngrkpi_year', 'dr_ho']
dr_to_overall.columns = ['global_id', 'mngrkpi_year', 'dr_to_overall']
dr_adherant_perc.columns = ['global_id', 'mngrkpi_year', 'dr_adherant_perc']
