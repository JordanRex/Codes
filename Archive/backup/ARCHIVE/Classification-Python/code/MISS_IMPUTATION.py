## MISSING VALUE IMPUTATION CLASS ##

#- implements multiple methods to perform missing value treatment
#- simple methods (mean/median/mode) imputation for categorical and numerical features.
#    - slightly tweak for adding grouped level if needed
#    - currently built to perform imputation consistent to below process (after encoding)
#- fancy methods (NNM/KNN/MICE) imputation for numerical features only
#    - requires encoding to be done prior

from IMPORT_MODULES import *
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
        
    def fit(self, X, y=None):
#         X.groupby(['pay_scale_group', 'abinbev_entity2'])
#         self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], 
#                               index=X.columns)
#         X.reset_index(drop=True)
#         X.groupby('abinbev_entity2')
#         self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], 
#                               index=X.columns)
#         X.reset_index(drop=True)
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], 
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
    def num_missing(self):
        return sum(self.isnull())
    
    def imputer_method(self, column, method=['mean', 'median', 'most_frequent']):
        x = Imputer(missing_values = 'NaN', strategy = method, axis = 0)
        return x.fit_transform(self[[column]]).ravel()
    
    def fancy_impute(X, which_method):
        """ currently supported algorithms are KNN, NNM and MICE from the fancyimpute package
        which_method = ['KNN', 'NNM', 'MICE']
        """
        print(which_method, ' based missing value imputation is happening ...', '\n')
        
        if which_method == 'NNM': X = NuclearNormMinimization().complete(X) # NNM method
        if which_method == 'KNN': X = KNN(k=5, verbose=False).complete(X) # KNN method
        if which_method == 'MICE':
            X_complete_df = X.copy()
            mice = MICE(verbose=False)
            X_complete = mice.complete(np.asarray(X.values, dtype=float))
            X_complete_df.loc[:, X.columns] = X_complete[:][:]
            X = X_complete_df
        print('missing value imputation completed', '\n')
        return X