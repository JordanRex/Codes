## ORDINAL CLASSIFIER MODULE - FOR ONLY TRAIN AND DEPLOYMENT ##

from sklearn.model_selection import GridSearchCV

class OrdinalClassifier():
    
    def __init__(self, train, ytrain, clf, params):
        self.clf = clf
        self.params = params
        self.clfs = {}
        self.clfs_scores = {}
        self.clfs_params = {}
        
        self.X = train
        self.y = ytrain

        self.preprocessing()
        
    def preprocessing(self):
        train=self.X
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
            train_df_cat, self.enc = ce_encodings(train_df=train_cat, encoding=encoding)
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

        self.feat_names = train.columns.values

        #train, valid = feat_sel.variance_threshold_selector(train=train, valid=valid, threshold=0.1)
        train_new, self.scalerobj = scalers(train=train, which_method='ss')
        train_new = train_new[self.feat_names]
        
        self.X = train_new
        
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
