## misc.py

import pandas as pd, numpy as np

## Encoding
import category_encoders as ce

def ce_encodings(train_df, encoding, valid_df=None, cols=None):
    print('category encoding is happening ...', '\n')
    if encoding=='bne':          
        enc=ce.BaseNEncoder(base=3, handle_missing=True)
    elif encoding=='be':
        enc=ce.BinaryEncoder()
    elif encoding=='he':
        enc=ce.HashingEncoder()
    elif encoding=='oe':
        enc=ce.OrdinalEncoder()
    elif encoding=='ohe':
        enc=ce.BaseNEncoder(base=1)
    enc.fit(train_df)
    
    if cols is None:
        train_df=enc.transform(train_df)
        if valid_df is not None: valid_df=enc.transform(valid_df)
    else:
        train_df[cols]=enc.transform(train_df[cols])
        if valid_df is not None: valid_df[cols]=enc.transform(valid_df[cols])
    print('category encoding completed', '\n')
    if valid_df is not None: 
        return train_df, valid_df, enc
    else:
        return train_df, enc
    

## Missing Value Treatment class
from sklearn.base import TransformerMixin
#from fancyimpute import KNN, NuclearNormMinimization, IterativeImputer

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
        
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], 
                              index=X.columns)
        return None

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
    def num_missing(self):
        return sum(self.isnull())
    
    def imputer_method(self, column, method=['mean', 'median', 'most_frequent']):
        x = Imputer(missing_values = 'NaN', strategy = method, axis = 0)
        return x.fit_transform(self[[column]]).ravel()
    
#     def fancy_impute(self, X, Y=None, which_method='IterativeImputer'):
#         """ currently supported algorithms are KNN, NNM and MICE from the fancyimpute package
#         which_method = ['KNN', 'NNM', 'IterativeImputer']
#         """
#         print(which_method, ' based missing value imputation is happening ...', '\n')
        
#         if which_method == 'NNM': X = NuclearNormMinimization().complete(X) # NNM method
#         if which_method == 'KNN': X = KNN(k=5, verbose=False).complete(X) # KNN method
        
#         if which_method == 'IterativeImputer':
#             imputer = IterativeImputer()
#             imputer.fit(X.values)
#             X_new = pd.DataFrame(data=imputer.transform(X.values), columns=X.columns)
#             Y_new = pd.DataFrame(data=imputer.transform(Y.values), columns=Y.columns)
#         print('missing value imputation completed', '\n')
#         return X_new, Y_new


## Feature Selection
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.linear_model import LogisticRegression

class feat_selection():
    
    def __init__(self):
        """ this module is for dynamic feature selection after all the processing and feat engineering phases. ideally this module is followed by the modelling phase immediately """

    # removing near zero variance columns
    def variance_threshold_selector(self, train, valid, threshold):
        print('input data shape is: ', train.shape, '\n')
        self.selector = VarianceThreshold(threshold)
        self.selector.fit(train)
        X = train[train.columns[self.selector.get_support(indices=True)]]
        Y = valid[valid.columns[self.selector.get_support(indices=True)]]
        #display(pd.DataFrame(X.head(5)))
        print('output data shape is: ', X.shape, '\n')
        return X, Y

    # using RFECV
    def rfecv(self, train, valid, y_train):
        # Create the RFE object and compute a cross-validated score.
        model = LogisticRegression(C=0.1, penalty='l1')
        #model = RandomForestClassifier(max_depth=10, max_features=0.3, n_estimators=200, n_jobs=-1)
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
    
    def feat_selection(self, train, valid, y_train, t=0.1):
        # read in the train, valid and y_train objects
        X, Y = self.variance_threshold_selector(train, valid, threshold=t)
        X, Y = self.rfecv(train=X, valid=Y, y_train=y_train)
        return X, Y

## Scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# for scaling features
def scalers(train, which_method, valid=None):
    if which_method == 'ss':
        sc = StandardScaler()
        sc.fit(train)
        train_new = pd.DataFrame(sc.transform(train), columns=train.columns.values)
        if valid is not None: valid_new = pd.DataFrame(sc.transform(valid), columns=valid.columns.values)
        if valid is not None: 
            return train_new, valid_new, sc
        else:
            return train_new, sc # scale all variables to zero mean and unit variance, required for PCA and related
    if which_method == 'mm':
        mm = MinMaxScaler()
        mm.fit(train)
        train_new = pd.DataFrame(mm.transform(train), columns=train.columns.values)
        if valid is not None: valid_new = pd.DataFrame(mm.transform(valid), columns=valid.columns.values)
        if valid is not None: 
            return train_new, valid_new, mm
        else:
            return train_new, mm # use this method to iterate
    
## some random functions
def mask_first(x):
    result = np.ones_like(x)
    result[0] = 0
    return result

def neg_mean(x):
    return -1 * np.mean(x)
