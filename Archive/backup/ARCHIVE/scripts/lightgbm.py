import lightgbm as lgb

# lightgbm class for tuning

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
    
    def lgbm_train(best_params, nestimators, X_train, y_train):
        model_2 = lgb.LGBMClassifier(silent = False, random_state = randomseed, objective = 'binary', n_estimators=nestimators)
        model_2.set_params(**best_params)
        model_2.fit(X_train, y_train, eval_metric = lightgbm_model.score)
        
        train_set = lgb.Dataset(X_train, label = y_train)
        model = lgb.train(best_params, train_set=train_set, num_boost_round=nestimators, feval=lightgbm_model.score)
        #model.set_params(**best_params)
        return model, model_2
    
    def lgbm_predict(X_test, y_test, model, mode = "validate", threshold = 0.17):
        #test_set = lgb.Dataset(X_test.values, feature_name=X_test.columns.values, label=y_test)
        try:
            pred = model.predict_proba(X_test)[:,1]
        except:
            pred = model.predict(X_test)
        
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

        
    def lgbm_cv(X_train, y_train, best):
        model = lgb.LGBMClassifier(**best, silent=True)
        lgb_cv_scores = sklearn.model_selection.cross_val_predict(model, X_train, y_train, cv=5)
        print('recall: ', sklearn.metrics.recall_score(y_pred=lgb_cv_scores, y_true=y_train))
        print('precision: ', sklearn.metrics.precision_score(y_pred=lgb_cv_scores, y_true=y_train))
        print('f1: ', sklearn.metrics.f1_score(y_pred=lgb_cv_scores, y_true=y_train))
        print('accuracy: ', sklearn.metrics.accuracy_score(y_pred=lgb_cv_scores, y_true=y_train))
        return model
    
    
# Create a lgb dataset
train_set = lgb.Dataset(X_train, label = y_train)

# calling the lightgbm function and best model
best, trials, nestimators = lightgbm_model.optimize()
print(1 - trials.average_best_error(), '\n')
model, model_2 = lightgbm_model.lgbm_train(best, nestimators, X_train, y_train)

# cv results
lightgbm_model.lgbm_cv(X_train, y_train, best)

#identify the  threshold for maximum F1 score
from sklearn.metrics import f1_score
f1_max = 0
ideal_threshold = 0
for i in np.linspace(0,1,101):
    lgb_pred, lgb_predict, tn, fp, fn, tp = lightgbm_model.lgbm_predict(X_test=X_valid, model=model_2, y_test=y_valid, mode='validate', threshold=i)
    f1 = f1_score(lgb_predict, y_valid)
    if f1 >= f1_max:
        ideal_threshold = i
        f1_max = f1
ideal_threshold, f1_max

# using lightgbm model on the validation set
lgb_pred, lgb_predict, tn, fp, fn, tp = lightgbm_model.lgbm_predict(X_test=X_valid, model=model_2, y_test=y_valid, mode='validate', threshold = ideal_threshold)
tn, fp, fn, tp

# precision-recall curve
p, r, thresholds = metrics.precision_recall_curve(y_true=y_valid, probas_pred=lgb_pred)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
plot_precision_recall_vs_threshold(p, r, thresholds)

# eli5 interpreter
#import eli5 to explain the model outputs
import eli5

#check whether the predictions from model_2 (lgb_pred) is same as that given by eli5.show_prediction for individual observations
lgb_predict[0]
1 - lgb_pred[0]

eli5.formatters.explain_weights_df(model_2)
eli5.show_prediction(model_2, X_valid.iloc[0])
eli5.show_prediction(model_2, X_valid.iloc[1])
eli5.show_prediction(model_2, X_valid.iloc[2])
eli5.show_prediction(model_2, X_valid.iloc[3000])
# eli5.formatters.explain_prediction_df(model_2, X_valid.iloc[0])
# eli5.formatters.explain_prediction_df(model_2, X_valid.iloc[1])
# eli5.formatters.explain_prediction_df(model_2, X_valid.iloc[2])
# eli5.formatters.explain_prediction_df(model_2, X_valid.iloc[3000])

# #get weights of all the variables and store it in a csv
# eli5.formatters.explain_weights_df(model_2, top = 100).to_csv(path_model_results + '//lgbm_model_weights_eli5_5.csv', index=False)