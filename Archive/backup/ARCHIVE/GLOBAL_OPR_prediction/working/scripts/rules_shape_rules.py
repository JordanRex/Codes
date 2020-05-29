## rules and shape and rules (final)

# Rules snippet
def apply_rules(dset,
                prev_prev_opr_col,
                prev_opr_col,
                pred_col,
                tib_col,
                mei_col,
                ta_col,
                mean_ca_col,
                predictions=None,
                iteration='first'):
    # mei_col = enagagement
    # Ranking for OPR 3A
    # time in band used in years
    
    if iteration=='first':
        df = dset.copy()
        df[prev_opr_col] = df[prev_opr_col].map(rev_dep_dict_without1B)
        df[prev_prev_opr_col] = df[prev_prev_opr_col].map(rev_dep_dict_without1B)

        predictions['rank'] = predictions['3A'].rank(pct=True,method='max')
        df = df.merge(predictions,on='global_id',how='left')
        df[pred_col] = df[pred_col].map(rev_dep_dict_without1B)

        df[mei_col] = pd.to_numeric(df[mei_col],errors='coerce')
        df[mean_ca_col] = pd.to_numeric(df[mean_ca_col],errors='coerce')
        df[ta_col] = pd.to_numeric(df[ta_col],errors='coerce')
    else:
        df = dset.copy()
        df[prev_opr_col] = df[prev_opr_col].map(rev_dep_dict_without1B)
        df[prev_prev_opr_col] = df[prev_prev_opr_col].map(rev_dep_dict_without1B)
        df[pred_col] = df[pred_col].map(rev_dep_dict_without1B)

        df[mei_col] = pd.to_numeric(df[mei_col],errors='coerce')
        df[mean_ca_col] = pd.to_numeric(df[mean_ca_col],errors='coerce')
        df[ta_col] = pd.to_numeric(df[ta_col],errors='coerce')
        
    rank_col='rank'
    
    # Rule1 - If Eng < 45%, TA < 70%, CA < 3.2 and OPR Suggestion is 4A/4B then OPR = 3A    
    conditions = [(df[pred_col].isin(['4B','4A'])) & ((df[mei_col]<0.45) | (df[ta_col]<70) | (df[mean_ca_col]<3.2))]
    choices = ['3A']
    df["rules_prediction"] = np.select(conditions, choices, default='nan')
    
    #Rule 2 : If OPR 2016 = 4B and OPR 2017 = 4B on the same position/band, then OPR 2018 = 4A
    conditions = [(df[prev_prev_opr_col] == '4B') & (df[prev_opr_col].isin(['4B'])) & (df[tib_col] > 2)]
    choices = ['4A']
    df["rules_prediction_1"] = np.select(conditions, choices, default='nan')
    df["rules_prediction"] = np.where(df['rules_prediction']=='nan',df["rules_prediction_1"],df["rules_prediction"])
    
    #Rule 3 : If OPR 2016 = OPR 2017 = 3A on the same band, and results on first quartile, then OPR 2018 = 4B, if not, then OPR 2018 = 3B
    conditions = [((df[prev_prev_opr_col] == '3A') & (df[prev_opr_col] =='3A') & (df[tib_col] > 2) & (df[pred_col].isin(['3A'])) & (df[rank_col] >= 0.5)), ((df[prev_prev_opr_col] == '3A') & (df[prev_opr_col] =='3A') & (df[tib_col] > 2) & (df[pred_col].isin(['3A'])) & (df[rank_col] < 0.5))]
    choices = ['4B', '3B']
    df["rules_prediction_1"] = np.select(conditions, choices, default='nan')
    df["rules_prediction"] = np.where(df['rules_prediction']=='nan',df["rules_prediction_1"],df["rules_prediction"])
    
    #Rule4 - If time in band > 5 years and OPR Suggestion = 4B, OPR = 4A
    conditions = [(df[pred_col].isin(['4B'])) & (df[tib_col]>5)]
    choices = ['4A']
    df["rules_prediction_1"] = np.select(conditions, choices, default='nan')
    df["rules_prediction"] = np.where(df['rules_prediction']=='nan',df["rules_prediction_1"],df["rules_prediction"])
    
    #df["rules_prediction"] = np.where(df['rules_prediction']=='nan',df[pred_col],df["rules_prediction"])
    
    if 'rank' in dset.columns:
        dset = dset.merge(df[['global_id','rules_prediction']],on='global_id',how='left')
    else:
        dset = dset.merge(df[['global_id','rules_prediction', 'rank']],on='global_id',how='left')
    return(dset)

#########################################################################################################################
