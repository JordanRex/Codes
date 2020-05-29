## navigate features script

################################################################################################################################
### BELTS
################################################################################################################################

with adlsfsc.open(path + '/2019/Data/Raw_Data/navigate/belts/belts_final.xlsx', 'rb') as f:
    belts = helpers.xlsx_read(f, belts_cols, sheet_name='Consolidated')

belts.dropna(inplace=True)
belts.columns = ['global_id', 'belt', 'date_acquired']
belts['date_acquired'] = pd.to_datetime(belts['date_acquired'])
belts['year'] = belts['date_acquired'].dt.year
belts.reset_index(drop=True, inplace=True)
belts['yearmonth'] = belts['year'].astype(str) + '0701'
belts['monthly_file_date'] = pd.to_datetime(belts['yearmonth'], format='%Y%m%d')
belts['year'] = np.where(belts['date_acquired']>belts['monthly_file_date'], belts['year']+1, belts['year'])
belts = cust_funcs.force_numeric(belts, cols=['global_id'])
belts.drop_duplicates(subset=['global_id', 'belt'], inplace=True)

belts_grp = belts.groupby(['global_id', 'year']).size().to_frame(name='count_of_belts').reset_index()
belts_grp = cust_funcs.force_numeric(belts_grp, cols=['global_id', 'year'])
belts_grp.dropna(subset=['global_id', 'year'], how='any', inplace=True)

################################################################################################################################
### GMT-GMBA
################################################################################################################################

# def gm_fn(dfpath, sheet):
#     gm = helpers.xlsx_read(file_path=dfpath, sheet_name=sheet)
#     gm.columns = (gm.columns.str.lower()).str.replace(' ', '_')
#     if sheet=='GMT': gm = gm[['global_id', 'gmt_year', 'status']]
#     if sheet=='GMBA': gm = gm[['global_id', 'gmba_year', 'status']]
#     gm.name = sheet.lower()
#     return gm

# # gmt - gmba datasets
# with adlsfsc.open(path + '/2019/Data/Raw_Data/navigate/gmt_gmba/gmt_gmba.xlsx', 'rb') as f:
#     gmt = gm_fn(dfpath=f, sheet='GMT')
#     gmba = gm_fn(dfpath=f, sheet='GMBA')

# gmt.columns = ['global_id', 'year', 'gmt_status']
# gmba.columns = ['global_id', 'year', 'gmba_status']

################################################################################################################################
### TALENT-POOL
################################################################################################################################

# talent pool features
with adlsfsc.open(path + '/2019/Data/Raw_Data/navigate/talent_pool/Talent Pool History 2009 - 2018.xlsx', 'rb') as f:
    tp = helpers.xlsx_read(f)
tp.columns = (tp.columns.str.lower()).str.replace(r' ', '_')

#########################
tp.rename(columns={'id':'employee_global_id'}, inplace=True)
tp_ads = tp[['employee_global_id', 'talent_pool_2018', 'talent_pool_2017', 'talent_pool_2016', 'talent_pool_2015']].copy()
tp_ads.dropna(inplace=True)
tp_ads.reset_index(drop=True, inplace=True)
tp_ads.set_index('employee_global_id', inplace=True)
tp_ads = tp_ads.reset_index()
tp_ads = pd.melt(tp_ads,
                 id_vars='employee_global_id',
                 value_vars=['talent_pool_2018',
       'talent_pool_2017', 'talent_pool_2016', 'talent_pool_2015'])
tp_ads.columns = ['global_id', 'tp', 'tp_level']
tp_ads['year'] = tp_ads['tp'].str.split('_').str[2]
tp_ads['year'] = tp_ads['year'].astype(int)
tp_ads.dropna(subset=['tp_level'], inplace=True)
tp_ads['talent_pool'] = np.where(tp_ads['tp_level']=='Global Talent Pool', 3,
                                     np.where(tp_ads['tp_level']=='People Bet', 2,
                                             np.where(tp_ads['tp_level']=='Zone Talent Pool', 1, 0)))
tp_ads = tp_ads[['global_id', 'year', 'talent_pool']]

###########################

# talent pool renomination
tp_renominated = tp[['employee_global_id', 'talent_pool_2018',
       'talent_pool_2017', 'talent_pool_2016', 'talent_pool_2015',
       'talent_pool_2014', 'talent_pool_2013', 'talent_pool_2012',
       'talent_pool_2011', 'talent_pool_2010', 'talent_pool_2009']]
tp_renominated.drop_duplicates(inplace=True)
tp_renominated.set_index('employee_global_id', inplace=True)
tp_renominated = tp_renominated.reset_index()
tp_renominated = pd.melt(tp_renominated, 
                         id_vars='employee_global_id', 
                         value_vars=['talent_pool_2018',
       'talent_pool_2017', 'talent_pool_2016', 'talent_pool_2015',
       'talent_pool_2014', 'talent_pool_2013', 'talent_pool_2012',
       'talent_pool_2011', 'talent_pool_2010', 'talent_pool_2009'])
tp_renominated.columns = ['global_id', 'tp', 'tp_level']

tp_renominated['year'] = tp_renominated['tp'].str.split('_').str[2]
tp_renominated['year'] = tp_renominated['year'].astype(int)

tp_renominated['isanull'] = np.where(tp_renominated['tp_level'].isna(), 1, 0)

def tp_renomination(df, year):
    temp = df[df['year']<=year]
    temp['renomination'] = np.where((temp['year']<=year)
                                   & (temp['isanull']==0),
                                   1,
                                   0)
    temp['diff'] = temp.groupby(['global_id'])['isanull'].diff()
    
    temp2 = temp[temp['isanull']==0].copy()
    temp2['meantp'] = temp2.groupby('global_id')['year'].transform(np.mean)
    temp2['mediantp'] = temp2.groupby('global_id')['year'].transform(np.median)
    temp2['renomination_new'] = np.where(temp2['meantp']==temp2['mediantp'], 0, 1)
    temp2 = temp2[['global_id', 'renomination_new']]
    temp2.drop_duplicates(inplace=True)
    
    temp = temp[['global_id', 'renomination']]
    temp = temp.merge(temp2, on='global_id', how='left')
    temp['talentpool_renomination'] = np.where(temp['renomination']==1, temp['renomination_new'], 0)
    temp = temp[['global_id', 'talentpool_renomination']]
    temp['year'] = year
    temp.drop_duplicates(subset=['global_id', 'year'], inplace=True)
    return temp

def tp_renom_years(df, years):
    temp = {}
    for i in years:
        temp[i] = tp_renomination(df, i)
    final_df = pd.concat(temp.values(), ignore_index=True)
    return final_df
tp_full = tp_renom_years(tp_renominated, [2015, 2016, 2017, 2018])

tp_full = tp_full.merge(tp_ads, how='outer')
tp_full.drop_duplicates(subset=['global_id', 'year'], inplace=True)
tp_full.columns = ['global_id', 'talentpool_renomination', 'talentpool_year', 'talentpool']

################################################################################################################################
### ENGAGEMENT
################################################################################################################################

with adlsfsc.open(path + '/2019/Data/Raw_Data/miscellaneous/engagement/engagement_2016-18.xlsx') as f:
    engagement_file = helpers.xlsx_read(f)
    
# make engagement file copy to avoid inplace operations
dummy = engagement_file.copy()

# create the engagement score dataframe
dummy_eng = dummy.iloc[:, 29:32]
dummy_eng = dummy_eng.drop([1], axis='rows')
dummy_eng.reset_index(drop=True, inplace=True)
dummy_eng.columns = dummy_eng.iloc[0]
dummy_eng = dummy_eng.reindex(dummy_eng.index.drop(0))
dummy_eng.columns = ['engagement_score_2018', 'engagement_score_2017', 'engagement_score_2016']

# create the manager effectiveness score dataframe
dummy_me = dummy.iloc[:, 65:68]
dummy_me = dummy_me.drop([1], axis='rows')
dummy_me.reset_index(drop=True, inplace=True)
dummy_me.columns = dummy_me.iloc[0]
dummy_me = dummy_me.reindex(dummy_me.index.drop(0))
dummy_me.columns = ['manager_effectiveness_score_2018', 'manager_effectiveness_score_2017', 'manager_effectiveness_score_2016']

# create the global ids dataframe
dummy_ids = dummy.iloc[1:, [1,4]]
dummy_ids.reset_index(drop=True, inplace=True)
dummy_ids.columns = dummy_ids.iloc[0]
dummy_ids = dummy_ids.reindex(dummy_ids.index.drop(0))
dummy_ids.columns = ['nsize', 'global_id']

engagement_full = pd.concat([dummy_ids.reset_index(drop=True), dummy_eng.reset_index(drop=True)], axis=1)
engagement_full = pd.concat([engagement_full.reset_index(drop=True), dummy_me.reset_index(drop=True)], axis=1)

engagement_full.sort_values(['global_id', 'nsize'], inplace=True)
engagement_full.reset_index(drop=True, inplace=True)

engfull = engagement_full.copy()
engfull.drop(columns=['nsize'], inplace=True)
engfull.drop_duplicates(inplace=True)
engfull.set_index('global_id', inplace=True)
engfull.reset_index(inplace=True)
engfull = pd.melt(engfull,
                 id_vars='global_id',
                 value_vars=['engagement_score_2018', 'engagement_score_2017',
       'engagement_score_2016', 'manager_effectiveness_score_2018',
       'manager_effectiveness_score_2017', 'manager_effectiveness_score_2016'])
engfull['eng_col'] = engfull['variable'].str.split('_2').str[0]
engfull['year'] = engfull['variable'].str.split('_').str[-1].astype(int)
engfull.drop(columns='variable', inplace=True)

################################################################################################################################
### PDP
################################################################################################################################

# read from adls
# with adlsfsc.open(path + '/2019/Data/RawData/navigate/pdp/pdp2016.xlsx') as f:
#     pdp16 = helpers.xlsx_read(f)
# with adlsfsc.open(path + '/2019/Data/RawData/navigate/pdp/pdp2017.xlsx') as f:
#     pdp17 = helpers.xlsx_read(f)
# with adlsfsc.open(path + '/2019/Data/RawData/navigate/pdp/pdp2018.xlsx') as f:
#     pdp18 = helpers.xlsx_read(f)

# # save backup
# pdpbackup = open('E:/performance/input/pickle_files/pdpbackup.pkl','wb')
# pickle.dump(pdp16, pdpbackup)
# pickle.dump(pdp17, pdpbackup)
# pickle.dump(pdp18, pdpbackup)
# pdpbackup.close()

# # load backup
# pdpbackup = open('E:/performance/input/pickle_files/pdpbackup.pkl', 'rb')
# pdp16 = pickle.load(pdpbackup)
# pdp17 = pickle.load(pdpbackup)
# pdp18 = pickle.load(pdpbackup)
# pdpbackup.close()

# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/pdpbackup.pkl', 'wb') as f:
#     pickle.dump(pdp16, f)
#     pickle.dump(pdp17, f)
#     pickle.dump(pdp18, f)
#     f.close()

# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/pdpbackup.pkl', 'rb') as f:
#     pdp16 = pickle.load(f)
#     pdp17 = pickle.load(f)
#     pdp18 = pickle.load(f)
#     f.close()

# pdp16.name=2016
# pdp17.name=2017
# pdp18.name=2018

# pdpcols = ['employee_global_id', 'leadership_competency', 'functional_competency']

# def pdp_feats(df, pdpcols):
#     year=df.name
#     df = df[pdpcols]
#     df[['leadership_competency', 'functional_competency']] = ((df[['leadership_competency', 'functional_competency']].notnull()).astype('int'))
#     df = df.groupby(['employee_global_id'], as_index=False).agg({'leadership_competency':'sum', 'functional_competency':'sum'})
#     df['year'] = year
#     df.columns = ['global_id', 'lc_count', 'fc_count', 'year']
#     df.drop_duplicates(subset=['global_id'], inplace=True)
#     return df

# pdp16 = pdp_feats(pdp16, pdpcols)
# pdp17 = pdp_feats(pdp17, pdpcols)
# pdp18 = pdp_feats(pdp18, pdpcols)

# pdp_full = pd.concat([pdp16, pdp17], ignore_index=True, axis=0)
# pdp_full = pd.concat([pdp_full, pdp18], ignore_index=True, axis=0)

# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/pdpfull.pkl', 'wb') as f:
#     pickle.dump(pdp_full, f)
#     f.close()

with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/pdpfull.pkl', 'rb') as f:
    pdp_full = pickle.load(f)
    f.close()

################################################################################################################################
    
with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/navigate.pkl', 'wb') as f:
    pickle.dump(belts_grp, f)
    pickle.dump(tp_full, f)
    pickle.dump(engfull, f)
    pickle.dump(pdp_full, f)
    f.close()
    
navigate = open('E:/ADLS/pickles/navigate.pickle','wb')
pickle.dump(belts_grp, navigate)
pickle.dump(tp_full, navigate)
pickle.dump(engfull, navigate)
pickle.dump(pdp_full, navigate)
navigate.close()
