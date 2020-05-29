## xgb cv tuning script

class xgbclass():
    ################################################### INIT ###############################################
    def __init__(self, train, ytrain):
        self.randomseed = 1
        self.N_FOLDS = 3
        self.trials = Trials()
        self.MAX_EVALS = 50
        
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
        train = train_new.fillna(value=0)

        #train, valid = feat_sel.variance_threshold_selector(train=train, valid=valid, threshold=0.1)
        train_new, scalerobj = scalers(train, 'ss')

        weights_dict = {0:1.5, 1:0.85, 2:0.9, 3:1.6, 4:2}
        ytrain_weights = np.copy(ytrain)
        for k, v in weights_dict.items(): 
            ytrain_weights[ytrain==k]=v

        feat_names = train_new.columns.values
            
        self.train = train_new
        self.ytrain = ytrain
        self.weights = ytrain_weights
        self.dtrain = xgb.DMatrix(train_new, label=ytrain, weight=ytrain_weights, feature_names=feat_names)

        self.optimize()
        
        self.model = xgb.train(self.trials.best_trial['result']['params'], dtrain=self.dtrain, maximize=False, 
                               num_boost_round=self.num_rounds) #, feval=self.cust_score)
    ####################################################################################################
    
    def cust_score(self, preds, dtrain):
        labels = dtrain.get_label()
        score = skm.log_loss(y_pred=preds, y_true=labels)
        return 'score', score
    
    # function to be minimized and sent to the optimize function of hyperopt
    def xgb_score(self, params):
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['max_depth', 'max_bin']:
            params[parameter_name] = int(params[parameter_name])
            
        ## to tune on cv results (the right method)
        xgb_cv = xgb.cv(params=params, num_boost_round=100, nfold=self.N_FOLDS, dtrain=self.dtrain, early_stopping_rounds=10,
                       maximize=False, stratified=True, verbose_eval=10, metrics=['merror'])#, feval=self.cust_score
        num_rounds = len(xgb_cv['test-merror-mean'])
        bst_score = xgb_cv['test-merror-mean'][num_rounds-1]
        print('evaluation metric score of iteration is: ', bst_score, '\n')
        print('validation accuracy is: ', 1-xgb_cv['test-merror-mean'][num_rounds-1])
        return {'loss': bst_score, 'status': STATUS_OK, 'params': params, 'num_boost': num_rounds-1, 
                'valid_acc': 1-xgb_cv['test-merror-mean'][num_rounds-1]}        
    
    # function to do hyperparameter tuning with hyperopt (bayesian based method)
    def optimize(self):
        # space to be traversed for the hyperopt function
        space = {
            'base_score' : hp.uniform('base_score', 0.4, 0.5),
             'learning_rate' : 0.05,#hp.uniform('learning_rate', 0.05, 0.5),
             #'max_depth' : hp.choice('max_depth', np.arange(3, 8, dtype=int)),
            'max_depth' : hp.quniform('max_depth', 3, 10, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 3, 9, 0.5),
             'subsample' : hp.uniform('subsample', 0.85, 0.95),
            'alpha': hp.uniform('alpha', 0.05, 0.1),
             'gamma' : hp.uniform('gamma', 0, 0.1),
            'lambda' : hp.uniform ('lambda', 0, 0.1),
             'colsample_bytree' : hp.uniform('colsample_bytree', 0.75, 0.95),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.6, 0.8),
            'objective' : 'multi:softprob',
            'num_class':5,
            'grow_policy': 'lossguide', #hp.choice('grow_policy', ['depthwise', 'lossguide']),
            'max_bin': hp.quniform('max_bin', 250, 400, 50),
            'max_delta_step': hp.quniform('max_delta_step', 0, 3, 1),
            'booster': 'gbtree',#hp.choice('booster', ['gbtree', 'dart']),
            'tree_method': 'exact',#hp.choice('tree_method', ['exact', 'hist']),
            'n_jobs': -1
        }
        
        self.best = fmin(self.xgb_score, space, algo=tpe.suggest, trials=self.trials, max_evals=self.MAX_EVALS,
                    rstate=np.random.RandomState(self.randomseed))
        self.num_rounds = self.trials.best_trial['result']['num_boost']
        return None # results of all the iterations, the best one and the number of rounds for the best run
    
    # function to return cv results for train dataset (recall/precision/f1/accuracy)
    def xgb_cv(self):
        best=self.trials.best_trial['result']['params']
        model = xgb.XGBClassifier(**best, silent=True)
        xgb_cv_scores = sklearn.model_selection.cross_val_predict(model, self.train, self.ytrain, cv=5)
        print('accuracy: ', sklearn.metrics.accuracy_score(y_pred=xgb_cv_scores, y_true=self.ytrain))
        return None
    
    def get_xgb_imp(self):
        imp_vals = self.model.get_fscore()
        feats_imp = pd.DataFrame(imp_vals,index=np.arange(2)).T
        feats_imp.iloc[:,0]= feats_imp.index
        feats_imp.columns=['feature','importance']
        feats_imp.sort_values('importance',inplace=True,ascending=False)
        feats_imp.reset_index(drop=True,inplace=True)
        return feats_imp
