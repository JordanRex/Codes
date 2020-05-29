from sklearn.neighbors import KNeighborsClassifier

def quick_model_knn(train, valid, ytrain, yvalid):
    # drop unnecessary columns
    train.drop(['global_id', 'year'], inplace=True, axis=1)
    valid.drop(['global_id', 'year'], inplace=True, axis=1)

    # encoding and other preprocessing
    cat_columns = train.select_dtypes(include=['object']).columns.values

    ## for categorical
    ### split
    train_cat = train[cat_columns]
    valid_cat = valid[cat_columns]
    ### fillna
    train_cat.fillna(value='none', axis=1,  inplace=True)
    valid_cat.fillna(value='none', axis=1,  inplace=True)
    ### encoding
    encoding='oe'
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
    
    knnmodel = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knnmodel.fit(train_new, ytrain)
    print(knnmodel.score(valid_new, yvalid))
    
    pred_probs = knnmodel.predict_proba(valid_new)
    pred = knnmodel.predict(valid_new)
    
    return knnmodel, pred, pred_probs