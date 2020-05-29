from sklearn.ensemble import RandomForestClassifier

# random forest class for tuning

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