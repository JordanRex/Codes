from catboost import CatBoostClassifier, Pool

def quick_model_cat(train, valid, ytrain, yvalid):
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
        traindfcat, validdfcat, enc = ce_encodings(train_cat, valid_cat, encoding)
    else :
        print('Not supported. Use one of [be, bne, he, oe, ohe]', '\n')

    ## for numerical
    ### split
    num_cols = list(set(train.columns)-set(train_cat.columns))
    train_num = train[num_cols]
    valid_num = valid[num_cols]

    # reset all indices (better safe than sorry)
    train_cat.reset_index(drop=True, inplace=True)
    valid_cat.reset_index(drop=True, inplace=True)
    train_num.reset_index(drop=True, inplace=True)
    valid_num.reset_index(drop=True, inplace=True)

    ### combine with *_cat dfs
    trainnew = pd.concat([traindfcat, train_num], axis=1)
    validnew = pd.concat([validdfcat, valid_num], axis=1)

    ### missing value treatment
    miss = DataFrameImputer()
    trainnew = trainnew.fillna(value=-1)
    validnew = validnew.fillna(value=-1)

    feat_names = trainnew.columns.values
    
    #train, valid = feat_sel.variance_threshold_selector(train=train, valid=valid, threshold=0.1)
    trainnew, validnew = scalers(trainnew, validnew, 'ss')

    weights_dict = {0:1.5, 1:1.7, 2:0.8, 3:0.9, 4:1.4, 5:2}
    ytrain_weights = np.copy(ytrain)
    yvalid_weights = np.copy(yvalid)
    for k, v in weights_dict.items(): 
        ytrain_weights[ytrain==k]=v
        yvalid_weights[yvalid==k]=v
    
    # pool
    trainpool = Pool(data=trainnew, label=ytrain)
    validpool = Pool(data=validnew, label=yvalid)
    
    # catboost classifier
    model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.2, verbose=20, random_strength=0.2,
                               loss_function='MultiClassOneVsAll', eval_metric='Accuracy')
    model.fit(trainpool, eval_set=validpool)
    pred = model.predict(validnew)
    predprobs = model.predict_proba(validnew)

    print(skm.accuracy_score(y_pred=pred, y_true=yvalid))
    print(skm.confusion_matrix(y_pred=pred, y_true=yvalid))
    return model, pred, predprobs
