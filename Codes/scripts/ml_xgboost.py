## xgboost model

def get_xgb_imp(xgb):
    imp_vals = xgb.get_fscore()
    feats_imp = pd.DataFrame(imp_vals,index=np.arange(2)).T
    feats_imp.iloc[:,0]= feats_imp.index    
    feats_imp.columns=['feature','importance']
    feats_imp.sort_values('importance',inplace=True,ascending=False)
    feats_imp.reset_index(drop=True,inplace=True)
    return feats_imp

def quick_model_xgb(train, valid, ytrain, yvalid):
    # drop unnecessary columns
    train.drop(['global_id', 'year'], inplace=True, axis=1)
    valid.drop(['global_id', 'year'], inplace=True, axis=1)

    # encoding and other preprocessing
    #cat_columns = train.select_dtypes(include=['object']).columns.values
    cat_columns = ['zone', 'function', 'pdi_score_category', 'opr_prev', 'opr_prev_prev', 'ebm_level']
    
    # convert some object columns to numeric
    train = cust_funcs.force_numeric(train, cols=['engagement_score', 'manager_effectiveness_score'])
    valid = cust_funcs.force_numeric(valid, cols=['engagement_score', 'manager_effectiveness_score'])

    ## for categorical
    ### split
    train_cat = train[cat_columns]
    valid_cat = valid[cat_columns]
    ### fillna
    train_cat.fillna(value='none', axis=1,  inplace=True)
    valid_cat.fillna(value='none', axis=1,  inplace=True)
    ### encoding
    encoding='ohe'
    if encoding in ['be', 'bne', 'he', 'oe', 'ohe']:
        train_df_cat, valid_df_cat, enc = ce_encodings(train_cat, valid_cat, encoding)
    else :
        print('Not supported. Use one of [be, bne, he, oe, ohe]', '\n')
    
    ## for numerical
    ### split
    num_cols = list(set(train.columns)-set(train_cat.columns))
    train_num = train[num_cols]
    valid_num = valid[num_cols]

    # reset all indices (better safe than sorry)
    train_df_cat.reset_index(drop=True, inplace=True)
    valid_df_cat.reset_index(drop=True, inplace=True)
    train_num.reset_index(drop=True, inplace=True)
    valid_num.reset_index(drop=True, inplace=True)

    ### combine with *_cat dfs
    train_new = pd.concat([train_df_cat, train_num], axis=1)
    valid_new = pd.concat([valid_df_cat, valid_num], axis=1)

    ### missing value treatment
    miss = DataFrameImputer()
    train = train_new.fillna(value=-1)
    valid = valid_new.fillna(value=-1)

    feat_names = train.columns.values
    
    #train, valid = feat_sel.variance_threshold_selector(train=train, valid=valid, threshold=0.1)
    train_new, valid_new = scalers(train, valid, 'ss')
    
    weights_dict = {0:1.7, 1:3, 2:0.8, 3:0.85, 4:1.8, 5:2.5}
    ytrain_weights = np.copy(ytrain)
    yvalid_weights = np.copy(yvalid)
    for k, v in weights_dict.items(): 
        ytrain_weights[ytrain==k]=v
        yvalid_weights[yvalid==k]=v
    
    xg_train = xgb.DMatrix(train_new, label=ytrain, weight=ytrain_weights)
    xg_test = xgb.DMatrix(valid_new, label=yvalid, weight=yvalid_weights)

    # setup parameters for xgboost
    param = {'objective':'multi:softprob', 'max_depth':10, 'silent':1, 'nthread':-1, 'num_class':6, 
             'learning_rate':0.1, 'eval_metric':['mlogloss', 'merror'], 'n_jobs': -1,
            'tree_method':'exact', 'seed':1, 'grow_policy':'lossguide', 'max_delta_step': 3,
            'max_bin':400, 'alpha': 0.036, 'base_score': 0.4, 'colsample_bylevel': 0.8,
             'colsample_bytree': 0.7, 'gamma': 0.002, 'lambda': 0.03,
             'min_child_weight': 4, 'subsample': 0.85}

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 200
    model = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50, verbose_eval=10)
    # get prediction
    pred_probs = model.predict(xg_test)
    pred = np.argmax(pred_probs, 1)
    error_rate = np.sum(pred != yvalid) / yvalid.shape[0]
    print('Test error using softprob = {}'.format(error_rate))

    print(skm.accuracy_score(y_pred=pred, y_true=yvalid))
    print(skm.confusion_matrix(y_pred=pred, y_true=yvalid))
    return model, feat_names, xg_train, xg_test, pred, pred_probs
