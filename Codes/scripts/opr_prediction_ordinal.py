## OPR PREDICTION MODULE
### USE THIS IF ORDINAL CLASSIFICATION MODELLING MODULE WAS USED

# predict function (use this for 20xx year prediction)

def predict_opr_ordinal(df, ordinal_class):
    clfs = ordinal_class.clfs
    encoder = ordinal_class.enc
    scaler = ordinal_class.scalerobj
    featnames= ordinal_class.feat_names
    
    # save ids
    pred_ids = df.global_id
    # drop unnecessary columns
    df.drop(['global_id', 'year'], inplace=True, axis=1)

    # encoding and other preprocessing
    if {'opr_prev', 'opr_prev_prev'}.issubset(df.columns) :
        cat_columns = ['zone', 'function', 'opr_prev', 'opr_prev_prev', 'ebm_level']
    else :
        cat_columns = ['zone', 'function', 'ebm_level']
    # convert some object columns to numeric
    df = cust_funcs.force_numeric(df, cols=['engagement_score', 'manager_effectiveness_score',
        'mr_pers_compgroup_year_comp_score_mean_functional_competencies',
       'mr_pers_compgroup_year_comp_score_mean_leadership_competencies',
       'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_develop_people',
       'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_dream_big',
       'mr_pers_compgroupl1_year_comp_score_mean_leadership_competencies_live_our_culture',
       'net_target', 'teamsize', 'teamsize_delta', 'index_average',
       'position_velocity', 'emp_time_in_band1', 'count_of_belts',
       'talentpool_renomination', 'talentpool', 'engagement_score',
       'manager_effectiveness_score', 'fs_prom', 'fs_ho', 'fs_adherant_perc',
       'fs_to_overall', 'dr_prom', 'dr_ho', 'dr_adherant_perc',
       'dr_to_overall', 'mean_team_tenure', 'lc_count', 'fc_count',
       'position_tenure', 'target_delta'])

    # encoding
    ## for categorical
    ### split
    df_cat = df[cat_columns]
    ### fillna
    df_cat.fillna(value='none', axis=1,  inplace=True)
    df_df_cat = encoder.transform(df_cat)
    ## for numerical
    ### split
    num_cols = list(set(df.columns)-set(df_cat.columns))
    df_num = df[num_cols]
    # reset all indices (better safe than sorry)
    df_df_cat.reset_index(drop=True, inplace=True)
    df_num.reset_index(drop=True, inplace=True)
    ### combine with *_cat dfs
    df_new = pd.concat([df_df_cat, df_num], axis=1)

    ### missing value treatment
    miss = DataFrameImputer()
    df = df_new.fillna(value=-1)

    df_new = pd.DataFrame(scaler.transform(df), columns=df.columns.values)
    df_new = df_new[featnames]
    
    def predict(X, clfs):
        unique_class = np.array([0,1,2,3,4])
        clfs_predict = {k:clfs[k].best_estimator_.predict_proba(X) for k in clfs}
        clfs_predict_class1 = {k:clfs[k].best_estimator_.predict_proba(X)[:,1] for k in clfs}
        
        predicted = []
        for i,y in enumerate(unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - (clfs_predict[y][:,1]))
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        probs = np.vstack(predicted).T
        preds = np.argmax(probs, axis=1)
        return probs, preds, clfs_predict_class1
    
    return predict(df_new, clfs)
