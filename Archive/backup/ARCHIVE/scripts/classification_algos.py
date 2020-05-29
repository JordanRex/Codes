## importing the relevant packages:

# clear the workspace
%reset -f

# print list of files in directory
import os
print(os.listdir())

# print/display all plots inline
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# the base packages
import collections # for the Counter function
import csv # for reading/writing csv files
import pandas as pd, numpy as np, time, gc

# the various packages/modules used across processing (sklearn) and bayesian optimization (hyperopt, bayes_opt)
from sklearn import metrics
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from bayes_opt import BayesianOptimization
from tqdm import tqdm
from hyperopt import hp, tpe, STATUS_OK, fmin, Trials
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample

# modelling algorithms
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Evaluation of the model
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer

# define the global variables used later
MAX_EVALS = 10 # number of iterations/parameter sets created towards tuning
N_FOLDS = 5 # number of cv folds
randomseed = 1 # the value for the random state used at various points in the pipeline
pd.options.display.max_rows = 100 # specify if you want the full output in cells rather the truncated list
pd.options.display.max_columns = 200


###########################################################################################################################################
###########################################################################################################################################
""" XGBOOST """
## xgboost class for tuning parameters and returning the best model ##
###########################################################################################################################################
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
       ###########################################################################################################################################

## xgboost execute below lines to get the best params and results from the xgboost model
""" calling the model creation functions to return the trials (results object) and the best parameters.
the best parameters are used to train the model and the predicted results are returned with the xgboost_model.xgb_predict call """

# return the trials and best parameters
trials, best, num_rounds = xgboost_model.optimize(X_train=X_train, y_train=y_train)
print('best score was: ', 1 - trials.average_best_error(), '\n')
#print(trials.best_trial['result']['bst_score'])

# return the model object trained with the best parameters
model = xgboost_model.xgb_train(best, num_rounds)

# cv results
xgboost_model.xgb_cv(X_train, y_train, best)

# print results with confusion matrix for the validation set
xgb_pred, xgb_predict, tn, fp, fn, tp = xgboost_model.xgb_predict(X_test=X_valid, model=model, y_test=y_valid, mode='validate',
                                                                  trials=trials, threshold = 0.04)

print('true negatives: ', tn)
print('false positives: ', fp)
print('false negatives: ', fn)
print('true positives: ', tp)

p, r, thresholds = metrics.precision_recall_curve(y_true=y_valid, probas_pred=xgb_pred)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
# calling above function to decide the best decision threshold for your needs
plot_precision_recall_vs_threshold(p, r, thresholds)

# execute below snippet to save model for later use (a model is usually just a few mb so saving is good as backupif hard to recreate)
import pickle
pickle.dump(model, open("xgb_june17.pickle.dat", "wb"))
#loaded_model = pickle.load(open("xgb_june17.pickle.dat", "rb"))

## important features from the best model above
xgb.plot_importance(booster=model, max_num_features=25, show_values=False)

###########################################################################################################################################
###########################################################################################################################################


###########################################################################################################################################
###########################################################################################################################################
""" LIGHTGBM """
## lightgbm class for tuning parameters and returning the best model ##
###########################################################################################################################################
class lightgbm_model():
    def __init__():
        """ this class initializes some functions used in the lightgbm pipeline """
    
    def score(preds, train_set):
        labels = train_set.get_label()
        y_preds = [1 if y >= 0.5 else 0 for y in preds] # binaryzing your output

        rscore = sklearn.metrics.recall_score(y_pred=y_preds, y_true=labels)
        pscore = sklearn.metrics.precision_score(y_pred=y_preds, y_true=labels)
        #score = sklearn.metrics.f1_score(y_pred=y_preds, y_true=labels)
        #score = sklearn.metrics.roc_auc_score(y_score=y_preds, y_true=labels)
        score = (4*rscore + pscore)/5
        
        return 'score', score, True
    
    def lgbm_score(params):        
        global ITERATION
        ITERATION += 1
        
        # Retrieve the subsample if present otherwise set to 1.0
        subsample = params['boosting_type'].get('subsample', 1.0)
        # Extract the boosting type
        params['boosting_type'] = params['boosting_type']['boosting_type']
        params['subsample'] = subsample
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
            params[parameter_name] = int(params[parameter_name])
        
        start = timer()
        # Perform n_folds cross validation
        cv_results = lgb.cv(params, train_set, num_boost_round = 1000, nfold = N_FOLDS, 
                            early_stopping_rounds = 10, feval = lightgbm_model.score, seed = randomseed)
        run_time = timer() - start
        
        # Extract the best score
        best_score = np.max(cv_results['score-mean'])
        # Loss must be minimized
        loss = 1 - best_score
        # Boosting rounds that returned the highest cv score
        n_estimators = int(np.argmax(cv_results['score-mean']) + 1)

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'iteration': ITERATION,
                'estimators': n_estimators, 
                'train_time': run_time, 'status': STATUS_OK}
    
    def optimize():
        # Keep track of evals
        global ITERATION
        ITERATION = 0
        global trials
        trials = Trials()
        space = {
            'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.75, 0.9)}, 
                                                         {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.75, 0.9)},
                                                         {'boosting_type': 'goss', 'subsample': 1.0}]),
            'num_leaves': hp.quniform('num_leaves', 100, 1000, 50),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 30000, 300000, 20000),
            'min_child_samples': hp.quniform('min_child_samples', 1, 3, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'subsample': hp.uniform('subsample', 0.7, 0.9),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 0.8),
            'scale_pos_weight': hp.quniform('scale_pos_weight', 1, 3, 1),
            'objective': 'binary'
        }
        
        # Run optimization
        best = fmin(fn = lightgbm_model.lgbm_score, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = trials, rstate = np.random.RandomState(randomseed))
        best = trials.best_trial['result']['params']
        nestimators = trials.best_trial['result']['estimators']
        return best, trials, nestimators
    
    def lgbm_train(best_params, nestimators):
        train_set = lgb.Dataset(X_train, label = y_train)
        #model = lgb.LGBMClassifier(silent = False, random_state = randomseed, objective = 'binary', n_estimators=nestimators)
        model = lgb.train(best_params, train_set=train_set, num_boost_round=nestimators, feval=lightgbm_model.score)
        #model.set_params(**best_params)
        #model.fit(X_train, y_train, eval_metric = "auc")
        return model
    
    def lgbm_predict(X_test, y_test, model, mode = "validate"):
        #test_set = lgb.Dataset(X_test.values, feature_name=X_test.columns.values, label=y_test)
        pred = model.predict(X_test)
        predict = np.where(pred > 0.044, 1, 0)
        
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
        
    def lgbm_cv(X_train, y_train, best):
        model = lgb.LGBMClassifier(**best, silent=True)
        lgb_cv_scores = sklearn.model_selection.cross_val_predict(model, X_train, y_train, cv=5)
        print('recall: ', sklearn.metrics.recall_score(y_pred=lgb_cv_scores, y_true=y_train))
        print('precision: ', sklearn.metrics.precision_score(y_pred=lgb_cv_scores, y_true=y_train))
        print('f1: ', sklearn.metrics.f1_score(y_pred=lgb_cv_scores, y_true=y_train))
        print('accuracy: ', sklearn.metrics.accuracy_score(y_pred=lgb_cv_scores, y_true=y_train))
        
###########################################################################################################################################

# Create a lgb dataset
train_set = lgb.Dataset(X_train, label = y_train)

# calling the lightgbm function and best model
best, trials, nestimators = lightgbm_model.optimize()
print(1 - trials.average_best_error(), '\n')
model = lightgbm_model.lgbm_train(best, nestimators)

# cv results
lightgbm_model.lgbm_cv(X_train, y_train, best)

# using lightgbm model on the validation set
lgb_pred, lgb_predict, tn, fp, fn, tp = lightgbm_model.lgbm_predict(X_test=X_valid, model=model, y_test=y_valid, mode='validate')
p, r, thresholds = metrics.precision_recall_curve(y_true=y_valid, probas_pred=lgb_pred)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
plot_precision_recall_vs_threshold(p, r, thresholds)

###########################################################################################################################################
###########################################################################################################################################


###########################################################################################################################################
###########################################################################################################################################
""" RANDOM FOREST """
# random forest class for tuning
###########################################################################################################################################
class rf_model():
    
    def __init__():
        """ this class initializes some functions used in the random forest pipeline """
        
    def rf_score(params):        
        global ITERATION
        ITERATION += 1

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['max_depth', 'n_estimators']:
            params[parameter_name] = int(params[parameter_name])
                
        rf_results = RandomForestClassifier(**params, random_state=randomseed)
        #rf_results.fit(X_train, y_train)
        rf_cv_scores = sklearn.model_selection.cross_val_predict(rf_results, X_train, y_train, cv=5, verbose=False)        
        recall_score = sklearn.metrics.recall_score(y_pred=rf_cv_scores, y_true=y_train)
        precision_score = sklearn.metrics.precision_score(y_pred=rf_cv_scores, y_true=y_train)
        f1_score = sklearn.metrics.f1_score(y_pred=rf_cv_scores, y_true=y_train)

        return {'loss': (1 - recall_score), 'status': STATUS_OK, 'params': params, 'iteration': ITERATION}
    
    def optimize():
        # Keep track of evals
        global ITERATION
        ITERATION = 0
        
        global trials
        trials = Trials()
        space = {
            'max_depth' : hp.quniform('max_depth', 5, 10, 1),
            'max_features': hp.choice('max_features', range(20, int((X_train.shape[:][1])/5))),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'n_estimators': hp.choice('n_estimators', np.arange(200, 1000))
        }
        
        # Run optimization
        best = fmin(fn = rf_model.rf_score, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = trials, rstate = np.random.RandomState(randomseed))
        best = trials.best_trial['result']['params']
        return best, trials
    
    def rf_train(best_params):
        model = RandomForestClassifier(random_state = randomseed)
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        return model
    
    def rf_predict(X_test, y_test, model, mode = "validate"):
        pred = model.predict_proba(X_test)[:, 1]
        predict = np.where(pred > 0.12, 1, 0)
        
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
        
    def rf_cv(X_train, y_train, best):
        model = RandomForestClassifier(**best, verbose=False)
        rf_cv_scores = sklearn.model_selection.cross_val_predict(model, X_train, y_train, cv=5)
        print('recall: ', sklearn.metrics.recall_score(y_pred=rf_cv_scores, y_true=y_train))
        print('precision: ', sklearn.metrics.precision_score(y_pred=rf_cv_scores, y_true=y_train))
        print('f1: ', sklearn.metrics.f1_score(y_pred=rf_cv_scores, y_true=y_train))
        print('accuracy: ', sklearn.metrics.accuracy_score(y_pred=rf_cv_scores, y_true=y_train))
        
###########################################################################################################################################

# calling the randomforest function and returning the best model
best, trials = rf_model.optimize()
print(1 - trials.average_best_error(), '\n')
model = rf_model.rf_train(best)

# cv results
rf_model.rf_cv(X_train, y_train, best)

# predicting using the best random forest model on the validation set
rf_pred, rf_predict, tn, fp, fn, tp = rf_model.rf_predict(X_test=X_valid, model=model, y_test=y_valid, mode='validate')

print('true negatives: ', tn)
print('false positives: ', fp)
print('false negatives: ', fn)
print('true positives: ', tp)

###########################################################################################################################################
###########################################################################################################################################


###########################################################################################################################################
###########################################################################################################################################
""" SVC """
# support vector classifier here (dont tune much - very computationally expensive for kernel methods. stick to linear)
###########################################################################################################################################
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_valid = scaling.transform(X_valid)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100]}]
"""  several more parameters need to be included 
        1. other kernel types (rbf, poly)
        2. class balancing parameter (class weight) 
        3. cv (stratified, non-stratified, KFolds) 
tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'degree': [5, 10, 20]},
                    {'kernel': ['rbf'], 'gamma': ['auto']}]
        """
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=StratifiedKFold(y=y_train, n_folds=5),
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:", '\n')
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_valid, clf.predict(X_valid)
    print(classification_report(y_true, y_pred))
    
svc_pred = clf.best_estimator_.predict_proba(X=X_valid)[:, 1]
svc_predict = clf.best_estimator_.predict(X=X_valid)
print('recall: ', sklearn.metrics.recall_score(y_pred=svc_predict, y_true=y_valid))
print('precision: ', sklearn.metrics.precision_score(y_pred=svc_predict, y_true=y_valid))
print('f1: ', sklearn.metrics.f1_score(y_pred=svc_predict, y_true=y_valid))
print('accuracy: ', sklearn.metrics.accuracy_score(y_pred=svc_predict, y_true=y_valid))

###########################################################################################################################################
###########################################################################################################################################

predictions_df_probs = pd.DataFrame({'xgb_pred': xgb_pred,
                                  'lgb_pred': lgb_pred,
                                  'rf_pred': rf_pred,
                                    'svc_pred': svc_pred,
                                    'actual': y_valid})
# just to see what each modelling algo is really learning from our data here. use it to create your manual ensemble if needed #