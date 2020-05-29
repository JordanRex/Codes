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
        self.valid_ids=np.array(valid['global_id'])
        # drop unnecessary columns
        train.drop(['global_id', 'year'], inplace=True, axis=1)
        valid.drop(['global_id', 'year'], inplace=True, axis=1)

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
        valid = cust_funcs.force_numeric(valid, cols=['engagement_score', 'manager_effectiveness_score',
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

##### USE BELOW SNIPPET WHILE DOING ITERATIONS USING THIS MODULE #####
# # create the predictions, scaler, encoder
# probs_all, preds_all = xgb_ordinal.predict()
# clfs_predict_df = pd.DataFrame.from_dict(xgb_ordinal.clfs_predict_class1)
# clfs_predict_df['global_id'] = xgb_ordinal.valid_ids
# scalerobj = xgb_ordinal.scalerobj
# encoderobj = xgb_ordinal.enc

# np.array(np.unique(preds_all, return_counts=True)).T
# skm.accuracy_score(y_pred=preds_all, y_true=yvalid_0to5_2018)

# for i in range(4):
#     clfs_predict_df.sort_values(by=[3-i], inplace=True, ascending=False, kind='mergesort')
#     clfs_predict_df.reset_index(inplace=True, drop=True)
#     clfs_predict_df[str(3-i) + '_new_index'] = clfs_predict_df.index
#     clfs_predict_df[str(3-i) + '_index_perc'] = clfs_predict_df[str(3-i) + '_new_index'].rank(pct=True)
#     clfs_predict_df[str(3-i) + '_flag'] = np.where(clfs_predict_df[str(3-i) + '_index_perc'] < 
#                                                  global_distribution_df.iloc[i, 2], 1, 0)
    
# clfs_predict_df['class'] = np.where(clfs_predict_df['3_flag']==1, 4,
#                                    np.where(clfs_predict_df['2_flag']==1, 3,
#                                            np.where(clfs_predict_df['1_flag']==1, 2,
#                                                    np.where(clfs_predict_df['0_flag']==1, 1, 0))))

# shape_predictions_df = clfs_predict_df[['global_id', 'class']].copy()

# valid_labels = valid_0to5_2018[['global_id']].copy()
# valid_labels['actuals'] = yvalid_0to5_2018.copy()

# shape_predictions_df = shape_predictions_df.merge(valid_labels, how='left', on=['global_id'])

# skm.accuracy_score(y_pred=shape_predictions_df['class'], y_true=shape_predictions_df['actuals'])
# skm.confusion_matrix(y_pred=shape_predictions_df['class'], y_true=shape_predictions_df['actuals'])
