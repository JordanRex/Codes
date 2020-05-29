## xgboost model

def get_xgb_imp(xgb):
    imp_vals = xgb.get_score(importance_type='total_gain')
    feats_imp = pd.DataFrame(imp_vals,index=np.arange(2)).T
    feats_imp.iloc[:,0]= feats_imp.index    
    feats_imp.columns=['feature','importance']
    feats_imp.sort_values('importance',inplace=True,ascending=False)
    feats_imp.reset_index(drop=True,inplace=True)
    return feats_imp

def quick_model_xgb(train, ytrain, 
                    params={'objective':'multi:softprob', 'max_depth':4, 'silent':1, 'nthread':-1, 'num_class':5, 
             'learning_rate':0.2, 'eval_metric':['mlogloss', 'merror'], 'n_jobs': -1,
            'tree_method':'exact', 'seed':1, 'grow_policy':'lossguide', 'max_delta_step': 3,
            'max_bin':300, 'alpha': 0.02, 'base_score': 0.4, 'colsample_bylevel': 0.75,
             'colsample_bytree': 0.8, 'gamma': 0.03, 'lambda': 0.01,
             'min_child_weight': 8, 'subsample': 0.85}, num_round=100):
    # drop unnecessary columns
    train.drop(['global_id', 'year'], inplace=True, axis=1)

    # encoding and other preprocessing
    #cat_columns = train.select_dtypes(include=['object']).columns.values
    if {'opr_prev', 'opr_prev_prev'}.issubset(train.columns) :
        cat_columns = ['zone', 'function', 'opr_prev', 'opr_prev_prev', 'ebm_level']
    else :
        cat_columns = ['zone', 'function', 'ebm_level']
    # convert some object columns to numeric
    train = cust_funcs.force_numeric(train, cols=['engagement_score', 'manager_effectiveness_score',
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

    ## for categorical
    ### split
    train_cat = train[cat_columns]
    ### fillna
    train_cat.fillna(value='none', axis=1,  inplace=True)
    ### encoding
    encoding='ohe'
    if encoding in ['be', 'bne', 'he', 'oe', 'ohe']:
        train_df_cat, encoderobj = ce_encodings(train_df=train_cat, encoding=encoding)
    else :
        print('Not supported. Use one of [be, bne, he, oe, ohe]', '\n')
    
    ## for numerical
    ### split
    num_cols = list(set(train.columns)-set(train_cat.columns))
    train_num = train[num_cols]

    # reset all indices (better safe than sorry)
    train_df_cat.reset_index(drop=True, inplace=True)
    train_num.reset_index(drop=True, inplace=True)

    ### combine with *_cat dfs
    train_new = pd.concat([train_df_cat, train_num], axis=1)

    ### missing value treatment
    miss = DataFrameImputer()
    train = train_new.fillna(value=-1)
        
    ### featnames
    featnames = train.columns.values
    
    #train, valid = feat_sel.variance_threshold_selector(train=train, valid=valid, threshold=0.1)
    train_new, scalerobj = scalers(train, 'ss')
    
    weights_dict = {0:1.4, 1:0.7, 2:0.75, 3:1.8, 4:2}
    ytrain_weights = np.copy(ytrain)
    for k, v in weights_dict.items(): 
        ytrain_weights[ytrain==k]=v
    
    feat_names = train_new.columns.values
    xg_train = xgb.DMatrix(train_new, label=ytrain, weight=ytrain_weights)
    
    # setup parameters for xgboost
    param = params
    cv_results = xgb.cv(param, xg_train, num_boost_round=num_round, 
                        verbose_eval=50, early_stopping_rounds=25, stratified=True, nfold=4)
    model = xgb.train(param, xg_train, num_boost_round=cv_results.shape[0])
    
    #print(f'CV error using softprob = {error_rate}')    
    return model, featnames, cv_results, encoderobj, scalerobj
