#### FEATURE SELECTION ####
#- near zero variance columns are removed (threshold=0.1)
#- rf based rfecv with depth=7, column_sampling=0.25, estimators=100 (optional=True/False)

from IMPORT_MODULES import *

class feat_selection():
    
    def __init__():
        """ this module is for dynamic feature selection after all the processing and feat engineering phases. ideally this
        module is followed by the modelling phase immediately """

    # removing near zero variance columns
    def variance_threshold_selector(train, valid, threshold):
        print('input data shape is: ', train.shape, '\n')
        selector = VarianceThreshold(threshold)
        selector.fit(train)
        X = train[train.columns[selector.get_support(indices=True)]]
        Y = valid[valid.columns[selector.get_support(indices=True)]]
        #display(pd.DataFrame(X.head(5)))
        print('output data shape is: ', X.shape, '\n')
        return X, Y

    # using RFECV
    def rfecv(train, valid, y_train):
        # Create the RFE object and compute a cross-validated score.
        #model = LogisticRegression(C=0.1, penalty='l1')
        model = RandomForestClassifier(max_depth=7, max_features=0.25, n_estimators=100, n_jobs=-1)
        rfecv = RFECV(estimator=model, step=1, scoring='roc_auc', verbose=True)
        rfecv.fit(train, y_train)
        print("Optimal number of features : %d" % rfecv.n_features_, '\n')

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (roc-auc)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

        features = [f for f,s in zip(train.columns, rfecv.support_) if s]
        train = train[features]
        valid = valid[features]
        return train, valid
    
    def feat_selection(train, valid, y_train, t=0.2):
        # read in the train, valid and y_train objects
        X, Y = feat_selection.variance_threshold_selector(train, valid, threshold=t)
        X, Y = feat_selection.rfecv(train=X, valid=Y, y_train=y_train)
        return X, Y