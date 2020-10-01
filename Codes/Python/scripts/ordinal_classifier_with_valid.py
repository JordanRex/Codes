## ORDINAL CLASSIFIER MODULE - FOR TRAIN/VALID ##

from sklearn.model_selection import GridSearchCV

class OrdinalClassifier():
    
    def __init__(self, train, ytrain, valid, yvalid, clf, params):
        self.clf = clf
        self.params = params
        self.clfs = {}
        self.clfs_scores = {}
        self.clfs_params = {}
        
        self.X = train
        self.y = ytrain
        self.XV = valid
        self.yV = yvalid

        self.preprocessing()
        
    def preprocessing(self):
        train=self.X
        valid=self.XV
        self.valid_ids=np.array(valid['id'])
        # drop unnecessary columns
        train.drop(['id'], inplace=True, axis=1)
        valid.drop(['id'], inplace=True, axis=1)

        # encoding and other preprocessing
        cat_columns = train.select_dtypes(include=['object']).columns.values

        # convert some object columns to numeric
        train = cust_funcs.force_numeric(train, cols=['a', 'b'])
        valid = cust_funcs.force_numeric(valid, cols=['a', 'b'])

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
            train_df_cat, valid_df_cat, self.enc = ce_encodings(train_df=train_cat, valid_df=valid_cat, encoding=encoding)
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
        train_new, valid_new, self.scalerobj = scalers(train=train, valid=valid, which_method='ss')
        self.X = train_new
        self.XV = valid_new
        
    def fit(self):
        X = self.X
        y = self.y
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in tqdm(range(self.unique_class.shape[0]-1)):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                model = clone(self.clf)
                clf = GridSearchCV(model, self.params, n_jobs=-1, cv=3, scoring='roc_auc', verbose=2, refit=True)
                clf.fit(X, binary_y)
                self.clfs_params[i] = clf.best_params_
                self.clfs_scores[i] = clf.best_score_
                self.clfs[i] = clf
    
    def predict(self):
        X = self.XV
        clfs_predict = {k:self.clfs[k].best_estimator_.predict_proba(X) for k in self.clfs}
        self.clfs_predict_class1 = {k:self.clfs[k].best_estimator_.predict_proba(X)[:,1] for k in self.clfs}
        
        predicted = []
        for i,y in enumerate(self.unique_class):
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
        return probs, preds
