## xgboost / lightgbm / catboost class

import xgboost as xgb
from catboost import CatBoostClassifier, Pool, cv
from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as skm
from sklearn.model_selection import cross_val_predict
from hyperopt import hp, tpe, STATUS_OK, fmin, Trials, space_eval
from hyperopt.pyll.stochastic import sample


#############################################################################################################################
## xgboost class for tuning parameters and returning the best model
class xgboost_model():
    
    def __init__(self, df, response, iter=1, rand=7, N_FOLDS=5, rounds=10, lr=0.3, df_test=None, response_test=None):
        self.df = df
        self.response = response
        self.iter = iter
        self.N_FOLDS = N_FOLDS
        self.nrounds = rounds
        self.lr = lr
        
        self.init_iter = 0
        self.rand = np.random.RandomState(rand)
        
        self.optimize()
        self.xgb_train()
        
    # define your custom evaluation metric here
    # currently defined: recall, precision, f1, roc-auc, weighted of recall/precision metrics
    def cust_score(self, preds, dtrain):
        labels = dtrain.get_label()
        y_preds = [1 if y >= self.cur_params['base_score'] else 0 for y in preds] # binaryzing your output
        rscore = skm.recall_score(y_pred=y_preds, y_true=labels)
        pscore = skm.precision_score(y_pred=y_preds, y_true=labels)
        #score = skm.roc_auc_score(y_score=preds, y_true=labels)
        score = (2*rscore + pscore)/3
        return 'score', score
    
    # function to be minimized and sent to the optimize function of hyperopt
    def xgb_score(self, params):
        self.init_iter += 1
        print(self.init_iter)
        self.cur_params = params
        
        metrics = ['auc']
        
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['max_depth', 'scale_pos_weight']:
            params[parameter_name] = int(params[parameter_name])

        dtrain = xgb.DMatrix(data=self.df.values, feature_names=self.df.columns.values, label=self.response)
        xgb_cv = xgb.cv(params = params, num_boost_round=self.nrounds, nfold=self.N_FOLDS, dtrain=dtrain, early_stopping_rounds=10,
                       feval = self.cust_score, maximize = True, stratified = True, verbose_eval=50, metrics=metrics) # may tune on the stratified flag
        num_rounds = len(xgb_cv['test-score-mean'])
        bst_score = xgb_cv['test-score-mean'][num_rounds-1]
        return {'loss': 1-bst_score, 'params': params, 'num_boost': num_rounds, 'status': STATUS_OK,
                'base_score': params['base_score']}
    
    # function to do hyperparameter tuning with hyperopt (bayesian based method)
    def optimize(self):
        trials = Trials()
        # space to be traversed for the hyperopt function
        space = {
            'base_score' : hp.quniform('base_score', 0.4, 0.51, 0.01),
             'learning_rate' : self.lr,
            'max_depth' : hp.quniform('max_depth', 3, 7, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 2, 5, 0.5),
             'subsample' : hp.quniform('subsample', 0.75, 0.85, 0.05),
             'gamma' : hp.uniform('gamma', 0, 0.5),
            'reg_lambda' : hp.uniform ('reg_lambda', 0, 0.3),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.85, 0.95, 0.01),
             'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.8, 0.9, 0.01),
            'scale_pos_weight' : hp.quniform('scale_pos_weight', 1, 5, 1),
            'objective' : 'binary:logistic',
            'n_jobs': -1,
            'tree_method': hp.choice('tree_method', ['exact']),
            'max_delta_step': hp.quniform('max_delta_step', 1, 5, 1),
            }
        
        self.best = fmin(self.xgb_score, space, algo=tpe.suggest, trials=trials, max_evals=self.iter,
                    rstate=self.rand)
        #best = trials.best_trial['result']['params']
        self.best_params = space_eval(space, self.best)
        self.num_rounds = trials.best_trial['result']['num_boost']
        self.trials = trials
            
    # train and return a model with the best params
    def xgb_train(self):
        self.dtrain = xgb.DMatrix(data=self.df.values, feature_names=self.df.columns.values, label=self.response)
        params = self.best_params
        params['max_depth'] = int(params['max_depth'])
        params['learning_rate'] = 0.01
        nrounds = 500 #self.num_rounds
        self.model = xgb.train(params, dtrain=self.dtrain, maximize=True, num_boost_round=nrounds, feval=self.cust_score, verbose_eval=100)
        
    def get_xgb_imp(self):
        imp_vals = self.model.get_fscore()
        feats_imp = pd.DataFrame(imp_vals,index=np.arange(2)).T
        feats_imp.iloc[:,0]= feats_imp.index    
        feats_imp.columns=['feature','importance']
        feats_imp.sort_values('importance',inplace=True,ascending=False)
        feats_imp.reset_index(drop=True,inplace=True)
        
        self.imp = feats_imp

#############################################################################################################################

class cat_model():
    
    def __init__(self, df, response, itr=1, rand=7, N_FOLDS=5, iters=10, df_test=None, response_test=None):
        self.df = df
        self.response = response
        self.itr = itr
        self.N_FOLDS = N_FOLDS
        self.iterations = iters
        
        self.init_iter = 0
        self.rand = np.random.RandomState(rand)
        
        self.optimize()
        self.cat_train()
        
    
    # function to be minimized and sent to the optimize function of hyperopt
    def cat_score(self, params):
        self.init_iter += 1
        print(self.init_iter)
        self.cur_params = params
                
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['depth', 'l2_leaf_reg']:
            params[parameter_name] = int(params[parameter_name])
        bootstrap_type = params['type']['bootstrap_type']
        params['bootstrap_type']=bootstrap_type
        if bootstrap_type=='Bernoulli':
            params['subsample']=params['type']['subsample']
        else:
            params['bagging_temperature']=params['type']['bagging_temperature']
        del params['type']
        
        pool = Pool(data=self.df.values, label=self.response)
        cat_cv = cv(pool, params, nfold=self.N_FOLDS, early_stopping_rounds=50,
                    stratified = True, verbose_eval=50, as_pandas=True) # may tune on the stratified flag
        iters = cat_cv.shape[0]
        bst_score = cat_cv.loc[iters-1,'test-AUC-mean']
        return {'loss': 1-bst_score, 'params': params, 'iterations': iters, 'status': STATUS_OK}
    
    # function to do hyperparameter tuning with hyperopt (bayesian based method)
    def optimize(self):
        trials = Trials()
        # space to be traversed for the hyperopt function
        space = {
             'learning_rate' : hp.quniform('learning_rate', 0.1, 0.5, 0.01),
            'depth' : hp.quniform('depth', 3, 12, 1),
             'l2_leaf_reg' : hp.quniform('l2_leaf_reg', 1, 10, 1),
             'border_count' : hp.quniform('border_count', 100, 350, 50),
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'iterations': self.iterations,
            'rsm': hp.quniform('rsm', 0.75, 0.95, 0.05),
            'type': hp.choice('type', 
                            [{'bootstrap_type': 'Bernoulli', 'subsample': hp.quniform('bernoulli_subsample', 0.75, 0.95, 0.05)}, 
                            {'bootstrap_type': 'Bayesian', 'bagging_temperature': hp.quniform('bayesian_bagtemp', 0.75, 0.95, 0.05)}]),
            'random_strength': hp.quniform('random_strength', 0, 0.5, 0.1),
            'class_weights': hp.choice('class_weights', [[0.5, 2], [0.1, 2], [1, 1], [0.4, 4]]),
            }
        
        self.best = fmin(self.cat_score, space, algo=tpe.suggest, trials=trials, max_evals=self.itr,
                    rstate=self.rand)
        #best = trials.best_trial['result']['params']
        self.best_params = space_eval(space, self.best)
        self.iters = trials.best_trial['result']['iterations']
        self.trials = trials
            
    # train and return a model with the best params
    def cat_train(self):
        self.trainpool = Pool(data=self.df.values, label=self.response)
        params=self.best_params
        params['bootstrap_type'] = params['type']['bootstrap_type']
        if params['bootstrap_type']=='Bernoulli':
            params['subsample']=params['type']['subsample']
        else:
            params['bagging_temperature']=params['type']['bagging_temperature']
        del params['type']
        
        print(params)
        if params['bootstrap_type']=='Bernoulli':
            model = CatBoostClassifier(depth=int(params['depth']), learning_rate=0.05, iterations=1000, random_strength=params['random_strength'],
                                   l2_leaf_reg=params['l2_leaf_reg'], border_count=params['border_count'],
                                   rsm=params['rsm'], class_weights=params['class_weights'], bootstrap_type='Bernoulli', subsample=params['subsample'],
                                   loss_function='Logloss', eval_metric='AUC', verbose=False)
        else:
            model = CatBoostClassifier(depth=int(params['depth']), learning_rate=0.05, iterations=1000, random_strength=params['random_strength'],
                                   l2_leaf_reg=params['l2_leaf_reg'], border_count=params['border_count'],
                                   rsm=params['rsm'], class_weights=params['class_weights'], bootstrap_type='Bayesian', 
                                       bagging_temperature=params['bagging_temperature'],
                                   loss_function='Logloss', eval_metric='AUC', verbose=False)
        model.fit(self.trainpool)
        self.model = model

#############################################################################################################################
        
class rf_model():
    
    def __init__(self, df, response, itr=1, rand=7, N_FOLDS=10, df_test=None, response_test=None):
        self.df = df
        self.response = response
        self.itr = itr
        self.N_FOLDS = N_FOLDS
        
        self.init_iter = 0
        self.rand = np.random.RandomState(rand)
        
        self.optimize()
        self.rf_train()
        
    
    # function to be minimized and sent to the optimize function of hyperopt
    def rf_score(self, params):
        self.init_iter += 1
        print(self.init_iter)
        self.cur_params = params

    # Make sure parameters that need to be integers are integers
        for parameter_name in ['max_depth', 'n_estimators']:
            params[parameter_name] = int(params[parameter_name])
                
        rf_results = RandomForestClassifier(**params, random_state=self.rand)
        rf_cv_scores = sklearn.model_selection.cross_val_predict(rf_results, self.df, self,response, cv=self.N_FOLDS, verbose=50, stratified=True)       
        recall_score = sklearn.metrics.recall_score(y_pred=rf_cv_scores, y_true=self.response)
        precision_score = sklearn.metrics.precision_score(y_pred=rf_cv_scores, y_true=self.response)
        score = (2*recall_score + precision_score)/3

        return {'loss': (1 - score), 'status': STATUS_OK, 'params': params}
    
    # function to do hyperparameter tuning with hyperopt (bayesian based method)
    def optimize(self):
        trials = Trials()
        # space to be traversed for the hyperopt function
        space = {
            'max_depth' : hp.quniform('max_depth', 3, 10, 1),
            'max_features': hp.choice('max_features', range(20, int((self.df.shape[:][1])/5))),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'n_estimators': hp.choice('n_estimators', np.arange(200, 500, 1000))
        }
        
        # Run optimization
        best = fmin(fn = self.rf_score, space = space, algo = tpe.suggest, 
            max_evals = self.itr, trials = trials, rstate = np.random.RandomState(self.rand))
        best = trials.best_trial['result']['params']
        return best, trials
        #best = trials.best_trial['result']['params']
        self.best_params = space_eval(space, self.best)
        self.iters = trials.best_trial['result']['iterations']
        self.trials = trials
            
    # train and return a model with the best params
    def rf_train(self):
        model = RandomForestClassifier(random_state = self.rand)
        model.set_params(**self.best_params)
        model.fit(self.df, self.response)
        self.model = model

#############################################################################################################################
