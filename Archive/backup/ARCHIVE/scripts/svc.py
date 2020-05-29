from sklearn.svm import SVC
from sklearn.metrics import classification_report

# scale the features for SVC
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_valid = scaling.transform(X_valid)


# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10]},
                   {'kernel': ['poly'], 'degree': [5, 10]},
                   {'kernel': ['rbf'], 'gamma': ['auto']}]
"""  several more parameters need to be included 
        1. other kernel types (rbf, poly)
        2. class balancing parameter (class weight) 
        3. cv (stratified, non-stratified, KFolds) 

tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100]},
                    {'kernel': ['poly'], 'degree': [5, 10, 20]},
                    {'kernel': ['rbf'], 'gamma': }]
        """

# tune between precision and recall
scores = ['precision', 'recall'
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

# predictions
svc_pred = clf.best_estimator_.predict_proba(X=X_valid)[:, 1]
svc_predict = clf.best_estimator_.predict(X=X_valid)
print('recall: ', sklearn.metrics.recall_score(y_pred=svc_predict, y_true=y_valid))
print('precision: ', sklearn.metrics.precision_score(y_pred=svc_predict, y_true=y_valid))
print('f1: ', sklearn.metrics.f1_score(y_pred=svc_predict, y_true=y_valid))
print('accuracy: ', sklearn.metrics.accuracy_score(y_pred=svc_predict, y_true=y_valid))