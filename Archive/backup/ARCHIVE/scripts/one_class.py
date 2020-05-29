from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

### ONE-CLASS METHODS ###

class oneclass_models():
    
    def __init__():
        """ this class contains several modelling algorithms for one-class classification/anomaly detection """

    def data_prepare(X_train, X_valid):
        # split and create 2 dataframes corresponing to positive/negative classes
        Negatives=X_train[X_train['response']==0]
        Positives=X_train[X_train['response']==1]
        Negatives.drop(['response'], axis=1, inplace=True)
        Positives.drop(['response'], axis=1, inplace=True)
        print(Negatives.shape)
        print(Positives.shape)
        
        # remove response from validation df too
        X_v = X_valid.drop(['response'], axis=1, inplace=False)
        print(X_v.shape)
        
        # take a random fraction of the negatives to reduce computation time
        Negatives = Negatives.sample(frac=0.1, replace=False, random_state=1)
        
        return Positives, Negatives, X_v
        
    def uni_svm(X_train, X_valid):
        """ one-class svm by training separately on positives and negatives """
        
        Positives, Negatives, X_v = oneclass_models.data_prepare(X_train, X_valid)
        
        # Set the parameters by cross-validation
        params = [{'kernel': ['rbf'],
                   'gamma': [0.01, 0.1, 0.5],
                   'nu': [0.01, 0.1, 0.5]}]

        clf_P = GridSearchCV(OneClassSVM(), cv=3, param_grid=params, scoring='accuracy', verbose=True)
        clf_N = GridSearchCV(OneClassSVM(), cv=3, param_grid=params, scoring='accuracy', verbose=True)
        clf_P.fit(X=Positives, y=np.full(len(Positives),1))
        print('positive model fit \n')
        clf_N.fit(X=Negatives, y=np.full(len(Negatives),1))
        print('negative model fit \n')
        clf_AD_P = OneClassSVM(gamma=clf_P.best_params_['gamma'],
                                      kernel=clf_P.best_params_['kernel'], nu=clf_P.best_params_['nu'], verbose=True)
        clf_AD_P.fit(Positives)
        clf_AD_N = OneClassSVM(gamma=clf_N.best_params_['gamma'],
                                      kernel=clf_N.best_params_['kernel'], nu=clf_N.best_params_['nu'], verbose=True)
        clf_AD_N.fit(Negatives)

        valid_pred_P=clf_AD_P.predict(X_v)
        valid_pred_N=clf_AD_N.predict(X_v)
        
        return valid_pred_P, valid_pred_N, clf_AD_P, clf_AD_N
    
    def score_table(valid_pred_P, valid_pred_N):
        table = pd.DataFrame({'P': valid_pred_P,
                              'N': -1*valid_pred_N,
                              'O': y_valid})
        table['P_N'] = np.where((table['P'] == 1) & (table['N'] == -1), 1, 0)

        print(sklearn.metrics.accuracy_score(y_pred=table['P_N'], y_true=table['O']))
        print(sklearn.metrics.precision_score(y_pred=table['P_N'], y_true=table['O']))
        print(sklearn.metrics.recall_score(y_pred=table['P_N'], y_true=table['O']))
        
        return table

# predictions
p, n, clf_p, clf_n = oneclass_models.uni_svm(X_train=X_train, X_valid=X_valid)
table=oneclass_models.score_table(valid_pred_N=n, valid_pred_P=p)

# ISOLATION FOREST
IFA=IsolationForest(n_estimators=200, max_features=0.3)
IFA.fit(Negatives)
train_IFA=IFA.predict(Negatives)
test_IFA=IFA.predict(Positives)

# accuracy custom function
def Train_Accuracy(Mat):
    Sum=0
    for i in Mat:
        if(i==1):
            Sum+=1.0
    return (Sum/len(Mat)*100)
def Test_Accuracy(Mat):
    Sum=0
    for i in Mat:
        if(i==-1):
            Sum+=1.0
    return (Sum/len(Mat)*100)

print("Training: Isolation Forest: ",(Train_Accuracy(train_IFA)),"%")
print("Test: Isolation Forest: ",(Test_Accuracy(test_IFA)),"%")