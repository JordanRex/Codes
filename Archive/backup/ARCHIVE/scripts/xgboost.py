import xgboost as xgb

## xgboost class for tuning parameters and returning the best model

class xgboost_model():
    
    def __init__():
        """ this class initializes some functions used in the xgboost pipeline """
    
    # define your custom evaluation metric here
    # currently defined: recall, precision, f1, roc-auc, weighted of recall/precision metrics
    def f1_score(preds, dtrain):
        labels = dtrain.get_label()
        #y_preds = [1 if y >= 0.5 else 0 for y in preds] # binaryzing your output
        #rscore = sklearn.metrics.recall_score(y_pred=y_preds, y_true=labels)
        #pscore = sklearn.metrics.precision_score(y_pred=y_preds, y_true=labels)
        #score = sklearn.metrics.f1_score(y_pred=y_preds, y_true=labels)
        score = sklearn.metrics.roc_auc_score(y_score=preds, y_true=labels)
        #score = (4*rscore + pscore)/5
        return 'score', score
    
    # function to be minimized and sent to the optimize function of hyperopt
    def xgb_score(params):
        global ITERATION
        ITERATION += 1
        randomseed = 1
        
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['max_depth', 'scale_pos_weight']:
            params[parameter_name] = int(params[parameter_name])
                    
        dtrain = xgb.DMatrix(data=X_train.values, feature_names=X_train.columns.values, label=y_train)
        xgb_cv = xgb.cv(params = params, num_boost_round=1000, nfold=N_FOLDS, dtrain=dtrain, early_stopping_rounds=5,
                       feval = xgboost_model.f1_score, maximize = True, stratified = True, verbose_eval=False) # may tune on the stratified flag
        num_rounds = len(xgb_cv['test-score-mean'])
        bst_score = xgb_cv['test-score-mean'][num_rounds-1]
        #print('evaluation metric score of iteration is: ', bst_score, '\n')
        return {'loss': (1 - bst_score), 'status': STATUS_OK, 'params': params, 'num_boost': num_rounds, 
                'bst_score': bst_score, 'base_score': params['base_score']}
    
    # function to do hyperparameter tuning with hyperopt (bayesian based method)
    def optimize(X_train, y_train):
        # Keep track of evals
        global ITERATION
        ITERATION = 0
        global trials
        trials = Trials()
        
        # space to be traversed for the hyperopt function
        space = {
            'base_score' : hp.quniform('base_score', 0.1, 0.9, 0.01),
             'learning_rate' : hp.uniform('learning_rate', 0.001, 0.2),
             #'max_depth' : hp.choice('max_depth', np.arange(3, 8, dtype=int)),
            'max_depth' : hp.quniform('max_depth', 5, 20, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 0, 5, 0.2),
             'subsample' : hp.quniform('subsample', 0.7, 0.85, 0.05),
             'gamma' : hp.quniform('gamma', 0, 1, 0.1),
            'reg_lambda' : hp.uniform ('reg_lambda', 0, 1),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.7, 0.85, 0.05),
            'scale_pos_weight' : hp.quniform('scale_pos_weight', 1, 5, 1),
            'objective' : 'binary:logistic'}
        
        best = fmin(xgboost_model.xgb_score, space, algo=tpe.suggest, trials=trials, max_evals=MAX_EVALS,
                    rstate=np.random.RandomState(randomseed))
        best = trials.best_trial['result']['params']
        num_rounds = trials.best_trial['result']['num_boost']
        
        return trials, best, num_rounds # results of all the iterations, the best one and the number of rounds for the best run
    
    # train and return a model with the best params
    def xgb_train(best_params, num_rounds):
        dtrain = xgb.DMatrix(data=X_train.values, feature_names=X_train.columns.values, label=y_train)
        model = xgb.train(best_params, dtrain=dtrain, maximize=True, num_boost_round=num_rounds, feval=xgboost_model.f1_score)
        return model

    # function to input a model and test matrix to output predictions and score parameters
    def xgb_predict(X_test, y_test, model, trials, mode = "validate", threshold = 0.2):
        dtest = xgb.DMatrix(data=X_test, feature_names=X_test.columns.values)
        pred = model.predict(dtest)
        #predict = np.where(pred > trials.best_trial['result']['base_score'], 1, 0)
        predict = np.where(pred > threshold, 1, 0)
        
        if mode == "validate":
            recall_score = sklearn.metrics.recall_score(y_pred=predict, y_true=y_test)
            precision_score = sklearn.metrics.precision_score(y_pred=predict, y_true=y_test)
            f1_score = sklearn.metrics.f1_score(y_pred=predict, y_true=y_test)
            auc_score = roc_auc_score(y_test, pred)
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_pred=predict, y_true=y_test).ravel()
            print(sklearn.metrics.confusion_matrix(y_pred=predict, y_true=y_test), '\n')
            print('recall score is: ', recall_score)
            print('precision score is: ', precision_score)
            print('f1_score is: ', f1_score)
            print('accuracy score: ', sklearn.metrics.accuracy_score(y_true=y_test, y_pred=predict))
            print('The final AUC after taking the best params and num_rounds when it stopped is {:.4f}.'.format(auc_score), '\n')
            return pred, predict, tn, fp, fn, tp
        else:
            return pred
        
    # function to return cv results for train dataset (recall/precision/f1/accuracy)
    def xgb_cv(X_train, y_train, best_params):
        model = xgb.XGBClassifier(**best, silent=True)
        xgb_cv_scores = sklearn.model_selection.cross_val_predict(model, X_train, y_train, cv=5)
        print('recall: ', sklearn.metrics.recall_score(y_pred=xgb_cv_scores, y_true=y_train))
        print('precision: ', sklearn.metrics.precision_score(y_pred=xgb_cv_scores, y_true=y_train))
        print('f1: ', sklearn.metrics.f1_score(y_pred=xgb_cv_scores, y_true=y_train))
        print('accuracy: ', sklearn.metrics.accuracy_score(y_pred=xgb_cv_scores, y_true=y_train))


""" calling the model creation functions to return the trials (results object) and the best parameters.
the best parameters are used to train the model and the predicted results are returned with the .xgb_predict call """

# return the trials and best parameters
trials, best, num_rounds = xgboost_model.optimize(X_train=X_train, y_train=y_train)
print('best score was: ', 1 - trials.average_best_error(), '\n')
#print(trials.best_trial['result']['bst_score'])

# return the model object trained with the best parameters
model = xgboost_model.xgb_train(best, num_rounds)

# uncomment below line if you went ahead with the train/test split approach over the CV based approach
#pred, predict, tn, fp, fn, tp = xgboost_model.xgb_predict(X_test=X_test, model=model, y_test=y_test, mode='validate', trials=trials)

# cv results
xgboost_model.xgb_cv(X_train, y_train, best)

# predictions on valid
t=0.1
xgb_pred, xgb_predict, tn, fp, fn, tp = xgboost_model.xgb_predict(X_test=X_valid, model=model, y_test=y_valid, mode='validate',
                                                                  trials=trials, threshold = t)

## important features from the best model above
xgb.plot_importance(booster=model, max_num_features=20, show_values=False)