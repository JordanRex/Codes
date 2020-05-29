## rules and shape module

def apply_rules(df,opr16_col,opr17_col,pred_col,tib_col,mei_col,ta17_col,mean_ca_col):
    #Rule 1 : If OPR 2016 = 4B and OPR 2017 = 4B on the same position/band, then OPR 2018 = 4A
    conditions  = [(df[opr16_col] == '4B') & df[opr17_col].isin(['4B']) & (df[tib_col] > 2)]
    choices     = ['4A']
    df["rules_prediction"] = np.select(conditions, choices, default='nan')
    #Rule2 - If OPR 2016 = OPR 2017 = 3A on the same band, and results on first quartile, then OPR 2018 = 4B, if not, then OPR 2018 = 3B
    conditions  = [(df[opr16_col] == '3A') & (df[opr17_col] =='3A') & (df[tib_col] > 2) & df[pred_col].isin(['4B','3A']) ]
    choices     = ['4B']
    df["rules_prediction_1"] = np.select(conditions, choices, default='nan')
    df["rules_prediction"] = np.where(df['rules_prediction']=='nan',df["rules_prediction_1"],df["rules_prediction"])
    #Rule4 - If time in band > 5 years and OPR Suggestion = 4B, OPR = 4A
    conditions  =[ (df[pred_col].isin(['4B'])) & (df[tib_col]>5)]
    choices     = ['4A']
    df["rules_prediction_1"] = np.select(conditions, choices, default='nan')
    df["rules_prediction"] = np.where(df['rules_prediction']=='nan',df["rules_prediction_1"],df["rules_prediction"])
    #Rule5 - If CA < 3.2 and OPR Suggestion is 4A/4B then OPR = 3A
    conditions  =[ (df[pred_col].isin(['4B','4A'])) & (df[mean_ca_col]<3.2)]
    choices     = ['3A']
    df["rules_prediction_1"] = np.select(conditions, choices, default='nan')
    df["rules_prediction"] = np.where(df['rules_prediction']=='nan',df["rules_prediction_1"],df["rules_prediction"])
    #Rule6 - If TA < 70% and OPR Suggestion is 4A/4B then OPR = 3A
    conditions  =[ (df[pred_col].isin(['4B','4A'])) & (df[ta17_col]<70)]
    choices     = ['3A']
    df["rules_prediction_1"] = np.select(conditions, choices, default='nan')
    df["rules_prediction"] = np.where(df['rules_prediction']=='nan',df["rules_prediction_1"],df["rules_prediction"])
    # Rule7 - If Eng < 45% and OPR Suggestion is 4A/4B then OPR = 3A
    conditions  =[ (df[pred_col].isin(['4B','4A'])) & (df[mei_col]<0.45)]
    choices     = ['3A']
    df["rules_prediction_1"] = np.select(conditions, choices, default='nan')
    df["rules_prediction"] = np.where(df['rules_prediction']=='nan',df["rules_prediction_1"],df["rules_prediction"])
    df.drop(columns=['rules_prediction_1'], inplace=True)
    return(df)


def apply_shape(dset, zone_col, function_col, band_col, opr18_col,
                l1=[0, 1, 2, 3, 4, 5, 6, 7, 8], l2=[9, 10], l3=[], group_len=20):
    # dataset to have columns with probabilities of OPR from 1B to 4A for all ids with same name as OPR    
    # checks for minimum presence of atleast 1 4A in group and len of group greater than 25 by default
    # pass a blank list if band_l3 is out of scope
    # zone and function level shape
    # output dataset would have column named s_opr
    
    l = pd.DataFrame()
    band_l1 = l1
    band_l2 = l2
    band_l3 = l3
    for x in [band_l1,band_l2,band_l3]:
        bp2 = dset[dset[band_col].isin(x)]
        for i in set(bp2.dropna(subset=[zone_col])[zone_col]):
            df = bp2[bp2[zone_col]==i]
            for j in ['sales','supply','marketing','solutions','people']:
                df1 = df[df[function_col]==j]
                # Checking for 4a presence in data
                try:
                    count_4a = df1[opr18_col].value_counts()['4A']
                except KeyError:
                    count_4a = 0

                if ((len(df1)>=group_len) & (count_4a > 1)):
                    df1['len'] = len(df1)
                    try:
                        val_1a = 1-df1[opr18_col].value_counts(normalize=True)['1A']
                    except KeyError:
                        val_1a = 1.01
                    try:
                        val_3b = 1-df1[opr18_col].value_counts(normalize=True)['3B']
                    except KeyError:
                        val_3b = 1.01
                    try:
                        val_3a = 1-df1[opr18_col].value_counts(normalize=True)['3A']
                    except KeyError:
                        va_3a = 1.01
                    try:
                        val_4b = 1-df1[opr18_col].value_counts(normalize=True)['4B']
                    except KeyError:
                        val_4b = 1.01
                    try:
                        val_4a = 1-df1[opr18_col].value_counts(normalize=True)['4A']
                    except KeyError:
                        val_4a = 1.01

                    df1['rank'] = df1['4A'].rank(pct=True)
                    df1['s_opr'] = '4A'
                    df1.sort_values(by='rank',inplace=True,ascending=False)
                    l = l.append(df1[df1['rank']>=(val_4a)])

                    df1 = df1[~df1['Global ID'].isin(set(l['Global ID']))]
                    df1['rank'] = df1['4B'].rank(pct=True)
                    df1.sort_values(by='rank',inplace=True,ascending=False)
                    df1['s_opr'] = '4B'
                    l = l.append(df1[df1['rank']>=(val_4b)])

                    df1 = df1[~df1['Global ID'].isin(set(l['Global ID']))]
                    df1['rank'] = df1['1A'].rank(pct=True)
                    df1.sort_values(by='rank',inplace=True,ascending=False)
                    df1['s_opr'] = '1A'
                    l = l.append(df1[df1['rank']>=(val_1a)])

                    df1 = df1[~df1['Global ID'].isin(set(l['Global ID']))]
                    df1['rank'] = df1['3A'].rank(pct=True)
                    df1.sort_values(by='rank',inplace=True,ascending=False)
                    df1['s_opr'] = '3A'
                    l = l.append(df1[df1['rank']>=(val_3a)])

                    df1 = df1[~df1['Global ID'].isin(set(l['Global ID']))]
                    df1['s_opr'] = '3B'
                    l = l.append(df1)
            else :
                if l.empty:
                    df1 = df
                else :
                    df1 = df[~df['Global ID'].isin(set(l['Global ID']))]
                df1['len'] = len(df1)
                try:
                    val_1a = 1-df1[opr18_col].value_counts(normalize=True)['1A']
                except KeyError:
                    va_1a = 1.01
                try:
                    val_3b = 1-df1[opr18_col].value_counts(normalize=True)['3B']
                except KeyError:
                    val_3b = 1.01
                try:
                    val_3a = 1-df1[opr18_col].value_counts(normalize=True)['3A']
                except KeyError:
                    va_3a = 1.01
                try:
                    val_4b = 1-df1[opr18_col].value_counts(normalize=True)['4B']
                except KeyError:
                    val_4b = 1.01
                try:
                    val_4a = 1-df1[opr18_col].value_counts(normalize=True)['4A']
                except KeyError:
                    val_4a = 1.01

                df1['rank'] = df1['4A'].rank(pct=True)
                df1['s_opr'] = '4A'
                df1.sort_values(by='rank',inplace=True,ascending=False)
                l = l.append(df1[df1['rank']>=(val_4a)])

                df1 = df1[~df1['Global ID'].isin(set(l['Global ID']))]
                df1['rank'] = df1['4B'].rank(pct=True)
                df1.sort_values(by='rank',inplace=True,ascending=False)
                df1['s_opr'] = '4B'
                l = l.append(df1[df1['rank']>=(val_4b)])

                df1 = df1[~df1['Global ID'].isin(set(l['Global ID']))]
                df1['rank'] = df1['1A'].rank(pct=True)
                df1.sort_values(by='rank',inplace=True,ascending=False)
                df1['s_opr'] = '1A'
                l = l.append(df1[df1['rank']>=(val_1a)])

                df1 = df1[~df1['Global ID'].isin(set(l['Global ID']))]
                df1['rank'] = df1['3A'].rank(pct=True)
                df1.sort_values(by='rank',inplace=True,ascending=False)
                df1['s_opr'] = '3A'
                l = l.append(df1[df1['rank']>=(val_3a)])

                df1 = df1[~df1['Global ID'].isin(set(l['Global ID']))]
                df1['s_opr'] = '3B'
                l = l.append(df1)
    return(l)

def rules_shape_rules_fn(valid, yvalid, pred, predprobs):
    opr_dict = {4.:'4A', 3.:'4B', 2.:'3A', 1.:'3B', 0.:'1A'}
    
    ##########################################################################################
    # preprocessing for rules
    rules_df = valid[['global_id', 'opr_prev', 'opr_prev_prev', 'time_in_band', 'engagement_score', 
               'net_target', 'mr_pers_year_comp_score_mean', 'employee_band', 'zone', 'function']]
    rules_df['response'] = yvalid
    rules_df['time_in_band'] = rules_df['time_in_band']/365
    rules_df['pred'] = pred
    rules_df['pred'] = rules_df['pred'].map(opr_dict)
    rules_df['opr_prev'] = rules_df['opr_prev'].map(opr_dict)
    rules_df['opr_prev_prev'] = rules_df['opr_prev_prev'].map(opr_dict)
    rules_df['response'] = rules_df['response'].map(opr_dict)
    rules_df = cust_funcs.force_numeric(rules_df, cols=['engagement_score'])
    # rules function call
    rules_df = apply_rules(df=rules_df.copy(), 
                           opr16_col='opr_prev_prev', 
                           opr17_col='opr_prev', 
                           opr18_col='response', 
                           pred_col='pred',
                           tib_col='time_in_band', mei_col='engagement_score', 
                           ta17_col='net_target', mean_ca_col='mr_pers_year_comp_score_mean')
    shape_df = pd.DataFrame(predprobs)
    shape_df.columns = ['1A', '3B', '3A', '4B', '4A']
    ##########################################################################################
    
    # preprocessing for shape
    rules_and_shape_df = pd.concat([rules_df.reset_index(drop=True), shape_df.reset_index(drop=True)], axis=1)
    rules_and_shape_df.rename(columns={'global_id':'Global ID'}, inplace=True)
    # shape function call
    rules_and_shape_df = apply_shape(dset=rules_and_shape_df.copy(), 
                                     zone_col='zone', 
                                     function_col='function', 
                                     band_col='employee_band',
                                     opr18_col='response')
    rules_and_shape_df = rules_and_shape_df[['zone', 'response', 'rules_prediction', 's_opr', 'Global ID']]
    rules_and_shape_df['pred'] = np.where(rules_and_shape_df['rules_prediction']=='nan', 
                                          rules_and_shape_df['s_opr'], rules_and_shape_df['rules_prediction'])
    rules_and_shape_df.sort_index(inplace=True)
    
    rules_and_shape_df['pred_mapped'] = rules_and_shape_df['pred'].map(dep_dict)
    rules_and_shape_df['response_mapped'] = rules_and_shape_df['response'].map(dep_dict)
    print(skm.accuracy_score(y_pred=rules_and_shape_df['pred_mapped'], y_true=rules_and_shape_df['response_mapped']))
    print(skm.confusion_matrix(y_pred=rules_and_shape_df['pred_mapped'], y_true=rules_and_shape_df['response_mapped']))
    ##########################################################################################
    
    # call rules again
    rules_df2 = valid[['global_id', 'opr_prev', 'opr_prev_prev', 'time_in_band', 'engagement_score', 
               'net_target', 'mr_pers_year_comp_score_mean', 'employee_band', 'zone', 'function']].copy()
    rules_df2['response'] = yvalid.copy()
    rules_df2['time_in_band'] = rules_df2['time_in_band']/365
    rules_df2['pred'] = np.array(rules_and_shape_df['pred'])
    rules_df2['opr_prev'] = rules_df2['opr_prev'].map(opr_dict)
    rules_df2['opr_prev_prev'] = rules_df2['opr_prev_prev'].map(opr_dict)
    rules_df2['response'] = rules_df2['response'].map(opr_dict)
    rules_df2 = cust_funcs.force_numeric(rules_df2, cols=['engagement_score'])

    rules_df2 = apply_rules(df=rules_df2.copy(), opr16_col='opr_prev_prev', opr17_col='opr_prev', opr18_col='response', 
                           pred_col='pred',
                       tib_col='time_in_band', mei_col='engagement_score', ta17_col='net_target', 
                        mean_ca_col='mr_pers_year_comp_score_mean')
    ##########################################################################################
    
    rules_and_shape_df['rules_again_prediction'] = np.array(rules_df2['rules_prediction'])
    rules_and_shape_df['pred'] = np.where(rules_and_shape_df['rules_again_prediction']=='nan',
                                          rules_and_shape_df['pred'],
                                          rules_and_shape_df['rules_again_prediction'])
    print(skm.accuracy_score(y_pred=rules_and_shape_df['pred'], y_true=rules_and_shape_df['response']))
    ##########################################################################################
    
    ##########################################################################################
    ##########################################################################################
    
    ##########################################################################################
    distribution_df = pd.DataFrame(np.array(np.unique(rules_and_shape_df.pred, return_counts=True)).T)
    distribution_df.columns = ['a', 'b']
    distribution_df['dist'] = 100*(distribution_df['b']/distribution_df['b'].sum())
    print('the rules and shape predictions distribution')
    print(distribution_df)
    print('\n')

    #rules_and_shape_df['pred'].value_counts()
    #rules_and_shape_df.response.value_counts()
    rules_and_shape_df['pred_mapped'] = rules_and_shape_df['pred'].map(dep_dict)
    rules_and_shape_df['response_mapped'] = rules_and_shape_df['response'].map(dep_dict)

    print('the rules and shape confusion matrix')
    print(skm.confusion_matrix(y_pred=rules_and_shape_df['pred_mapped'], 
                         y_true=rules_and_shape_df['response_mapped']))
    print('\n')

    print('the rules and shape precision score in the order -> [1A, 3B, 3A, 4B, 4A]')
    print(100*(skm.precision_score(y_pred=rules_and_shape_df['pred_mapped'], 
                             y_true=rules_and_shape_df['response_mapped'], average=None)))
    print('\n')
    ##########################################################################################
    
    preddf = rules_and_shape_df[['zone', 'response_mapped', 'pred_mapped']].copy()
    preddf['exact'] = np.where(preddf['response_mapped']==preddf['pred_mapped'], 1, 0)
    preddf['dev'] = (preddf['response_mapped'] - preddf['pred_mapped']).astype(int).abs()
    preddf['dev1'] = np.where(preddf['dev']<2, 1, 0)

    preddf2 = preddf.groupby(['zone']).agg({'response_mapped':np.size, 'exact':np.sum, 'dev1':np.sum}).reset_index()
    preddf2['accuracy_exact'] = (preddf2['exact']/preddf2['response_mapped'])*100
    preddf2['accuracy_dev1'] = (preddf2['dev1']/preddf2['response_mapped'])*100
    print('the zone level statistics')
    print(preddf2)
    print('\n')

    print('total accuracy: ', preddf2.exact.sum()/preddf2.response_mapped.sum())
    print('dev 1 accuracy: ', preddf2.dev1.sum()/preddf2.response_mapped.sum())
    
    return rules_and_shape_df
