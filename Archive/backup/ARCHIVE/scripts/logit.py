from sklearn.linear_model import LogisticRegression

# implementing only the baseline logistic model
# need to add support for parameter tuning

#kfold = model_selection.KFold(n_splits=5, random_state=1)
modelCV = LogisticRegression()
# Create regularization penalty space
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = np.logspace(0, 5, 10)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# Create grid search using 5-fold cross validation
clf = GridSearchCV(modelCV, hyperparameters, cv=5, verbose=0)

# Fit grid search
model_fit = clf.fit(X_train, y_train)

# View best hyperparameters
print('Best Penalty:', model_fit.best_estimator_.get_params()['penalty'])
print('Best C:', model_fit.best_estimator_.get_params()['C'])

#scoring = 'recall' # give precision or f1
#results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
#print("5-fold cross validation average accuracy: %.3f" % (results.mean()))

# getting the predictions for the actual test set
log_pred = model_fit.best_estimator_.predict_proba(X=X_valid)[:, 1]
log_predict = np.where(log_pred > 0.5, 1, 0)

# print the various evaluation metrics
print('auc: ', sklearn.metrics.roc_auc_score(y_score=log_pred, y_true=y_valid))
print('recall: ', sklearn.metrics.recall_score(y_pred=log_predict, y_true=y_valid))
print('precision: ', sklearn.metrics.precision_score(y_pred=log_predict, y_true=y_valid))
print('f1: ', sklearn.metrics.f1_score(y_pred=log_predict, y_true=y_valid))