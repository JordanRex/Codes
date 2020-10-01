# function to bucket sparse levels in categorical features to the 'others' category as well as handle new values in the valid df

from sklearn.base import TransformerMixin, BaseEstimator
from collections import defaultdict

class CategoryGrouper(BaseEstimator, TransformerMixin):  
    """A tranformer for combining low count observations for categorical features.
    This transformer will preserve category values that are above a certain threshold, while bucketing together all the other values. This will fix issues where new data may have an unobserved category value that the training data did not have.
    """
    
    def __init__(self, threshold=0.05):
        """ Initialize method.
        Args: threshold (float): The threshold to apply the bucketing when categorical values drop below that threshold.
        """
        self.d = defaultdict(list)
        self.threshold = threshold

    def transform(self, X, **transform_params):
        """Transforms X with new buckets.
        Args: X (obj): The dataset to pass to the transformer.
        Returns: The transformed X with grouped buckets.
        """
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].apply(lambda x: x if x in self.d[col] else 'others')
        return X_copy

    def fit(self, X, y=None, **fit_params):
        """ Fits transformer over X.
        Builds a dictionary of lists where the lists are category values of the
        column key for preserving, since they meet the threshold.
        """
        df_rows = len(X.index)
        for col in X.columns:
            calc_col = X.groupby(col)[col].agg(lambda x: (len(x) * 1.0) / df_rows)
            self.d[col] = calc_col[calc_col >= self.threshold].index.tolist()
        return self
    
# dfs with 100 elements in cat1 and cat2
# note how df_test has elements 'g' and 't' in the respective categories (unknown values)
df_train = pd.DataFrame({'cat1': ['a'] * 20 + ['b'] * 30 + ['c'] * 40 + ['d'] * 3 + ['e'] * 4 + ['f'] * 3,
                         'cat2': ['z'] * 25 + ['y'] * 25 + ['x'] * 25 + ['w'] * 20 +['v'] * 5})
df_test = pd.DataFrame({'cat1': ['a'] * 10 + ['b'] * 20 + ['c'] * 5 + ['d'] * 50 + ['e'] * 10 + ['g'] * 5,
                        'cat2': ['z'] * 25 + ['y'] * 55 + ['x'] * 5 + ['w'] * 5 + ['t'] * 10})

catgrouper = CategoryGrouper()
catgrouper.fit(df_train)
df_test_transformed = catgrouper.transform(df_test)
df_train_transformed = catgrouper.transform(df_train)

df_train_transformed

###########################################################################################################################################
###########################################################################################################################################

# global function to flatten columns after a grouped operation and aggregation
# outside all classes since it is added as an attribute to pandas DataFrames

def __my_flatten_cols(self, how="_".join, reset_index=True):
    how = (lambda iter: list(iter)[-1]) if how == "last" else how
    self.columns = [how(filter(None, map(str, levels))) for levels in self.columns.values] \
    if isinstance(self.columns, pd.MultiIndex) else self.columns
    return self.reset_index(drop=True) if reset_index else self
pd.DataFrame.my_flatten_cols = __my_flatten_cols

###########################################################################################################################################

# functions

import pandas as pd, numpy as np, sys
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from geopy.distance import vincenty
from sklearn.cluster import KMeans

def readcsv(path, cols=None):
    df = pd.read_csv(path)
    if cols is not None:
        return df.filter(cols)
    else: return df

    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def datetime_feats(df, cols=None):
    if cols is None: cols = [s for s in df.columns.values if 'DATE' in s]
    def dt_feats(df, col):
        df[col] = pd.to_datetime(df[i])
        df[str(col+'_'+'dayofweek')] = df[col].dt.dayofweek
        df[str(col+'_'+'dayofyear')] = df[col].dt.dayofyear
        #df = df.drop([col], axis = 1)
        return df
    # loop function over all raw date columns
    for i in cols:
        df = dt_feats(df, i)
    return df

def to_date(df, cols):
    for i in cols:
        df[i] = pd.to_datetime(df[i])
    return df

############################################################################################################################
# ENCODING CLASS
# - various types of encoding here
#    - binary encoder
#    - base encoder
#    - hashing encoder
#    - ordinal encoder
#    - one-hot encoder

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

class categ_encoders():
    
    def  __init__(self, df, y=None, which='oe', test=None, ytest=None, cols=None):
        self.train=df
        if test is not None: self.test=test
        else: self.test=None
        if ytest is not None: self.ytest=ytest
        if y is not None: self.y=y
        if cols is not None:
            self.cols=cols
        else:
            self.cols=self.train.select_dtypes('category').columns
        options_df = pd.DataFrame({'which': ['bne', 'be', 'he', 'oe', 'ohe', 'cat', 'js', 'te', 'woe', 'mest'],
                                  'description': ['base n', 'binary', 'hasing', 'ordinal', 'onehot', 'catboost', 
                                                  'james-stein', 'target', 'weight-of-evidence', 'm-estimator']})
        display('The different options available are: ', options_df)
        if which!='tgt_enc':
            self.encoder(which)
        else:
            df['target'] = y
            for i in cols:
                df[i], test[i] = self.target_encode(trn_series=df[i], 
                      tst_series=test[i], 
                      target=df.target, 
                      min_samples_leaf=10, 
                      smoothing=10,
                      noise_level=0.01)
            df.drop(columns=['target'], inplace=True)
            self.train, self.test = df, test
        print('category encoding completed', '\n')
           
        
    def encoder(self, encoding):
        print(str(encoding) + ' encoding is happening ...', '\n')
        if encoding=='bne':
            enc=ce.BaseNEncoder(base=3, cols=self.cols)
        elif encoding=='be':
            enc=ce.BinaryEncoder(cols=self.cols)
        elif encoding=='he':
            enc=ce.HashingEncoder(cols=self.cols)
        elif encoding=='oe':
            enc=ce.OrdinalEncoder(cols=self.cols)
        elif encoding=='ohe':
            enc=ce.OneHotEncoder(cols=self.cols)
        elif encoding=='cat':
            enc=ce.CatBoostEncoder(cols=self.cols)
        elif encoding=='js':
            enc=ce.JamesSteinEncoder(cols=self.cols)
        elif encoding=='te':
            enc=ce.TargetEncoder(cols=self.cols)
        elif encoding=='woe':
            enc=ce.WOEEncoder(cols=self.cols)
        elif encoding=='mest':
            enc=ce.MEstimateEncoder(cols=self.cols)

        if self.y is None:
            enc.fit(self.train)
        else:
            enc.fit(self.train, self.y)
        self.train=enc.transform(self.train)
        if self.test is not None: self.test=enc.transform(self.test)
        self.enc=enc
        
    def target_encode(self, trn_series, 
                  tst_series, 
                  target, 
                  min_samples_leaf, 
                  smoothing,
                  noise_level):
        """
        Smoothing is computed like in the following paper by Daniele Micci-Barreca
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        trn_series : training categorical feature as a pd.Series
        tst_series : test categorical feature as a pd.Series
        target : target data as a pd.Series
        min_samples_leaf (int) : minimum samples to take category average into account
        smoothing (int) : smoothing effect to balance categorical average vs prior  
        """ 
        assert len(trn_series) == len(target)
        assert trn_series.name == tst_series.name
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean 
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index 
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
        ################################################################
        
        
# helper class for feature encoding
class df_enc():
    
    def __init__(self, df, response, ohe, num, catemb, test=None, ytest=None, enc_method='he', split=True, scale=True):
        self.ohe, self.num, self.catemb = ohe, num, catemb
        self.enc_method = enc_method
        self.scale = scale
        self.split = split
        
        if scale==True:
            self.scaler = StandardScaler()
            
        if split==False:
            self.train = df
            self.ytrain = response
            self.main_nosplit()
        elif test is None:
            self.train, self.test, self.ytrain, self.ytest = train_test_split(df, response, test_size=0.33, random_state=1)
            self.main_split()
        else:
            self.train, self.test, self.ytrain, self.ytest = df, test, response, ytest
            self.main_split()
        self.test = self.test[self.train.columns]

    ############################################################################################################################################
        
    def main_split(self):
        ce_ohe = categ_encoders(df=self.train, y=self.ytrain, test=self.test, ytest=self.ytest, cols=self.ohe, which='ohe')
        ce_ohe.train[self.num] = ce_ohe.train[self.num].apply(pd.to_numeric)
        ce_ohe.test[self.num] = ce_ohe.test[self.num].apply(pd.to_numeric)
        if self.scale==True:
            ce_ohe.train[self.num] = self.scaler.fit_transform(ce_ohe.train[self.num])
            ce_ohe.test[self.num] = self.scaler.transform(ce_ohe.test[self.num])
        ce_ohe.train[self.catemb] = ce_ohe.train[self.catemb].astype(str)
        ce_ohe.test[self.catemb] = ce_ohe.test[self.catemb].astype(str)
        ce_catemb = categ_encoders(df=ce_ohe.train, y=self.ytrain, test=ce_ohe.test, ytest=self.ytest, cols=self.catemb, which=self.enc_method)
        self.train = ce_catemb.train.apply(pd.to_numeric)
        self.test = ce_catemb.test.apply(pd.to_numeric)
        #self.transform_objs = {'ohe': ce_ohe.enc, 'scaler': self.scaler, 'cat': ce_catemb.enc}
        
    def main_nosplit(self):
        ce_ohe = categ_encoders(df=self.train, y=self.ytrain, cols=self.ohe, which='ohe')
        ce_ohe.train[self.num] = ce_ohe.train[self.num].apply(pd.to_numeric)
        if self.scale==True:
            ce_ohe.train[self.num] = self.scaler.fit_transform(ce_ohe.train[self.num])
        ce_ohe.train[self.catemb] = ce_ohe.train[self.catemb].astype(str)
        ce_catemb = categ_encoders(df=ce_ohe.train, y=self.ytrain, cols=self.catemb, which=self.enc_method)
        self.train_full = ce_catemb.train.apply(pd.to_numeric)
        #self.transform_objs = {'ohe': ce_ohe.enc, 'scaler': self.scaler, 'cat': ce_catemb.enc}

############################################################################################################################

## MISSING VALUE IMPUTATION CLASS ##
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value in column.
        Columns of other types are imputed with mean of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

    def num_missing(self):
        return sum(self.isnull())

############################################################################################################################

from sklearn.base import TransformerMixin
from itertools import repeat
import scipy


class ThermometerEncoder(TransformerMixin):
    """
    Assumes all values are known at fit
    """
    def __init__(self, sort_key=None):
        self.sort_key = sort_key
        self.value_map_ = None
    
    def fit(self, X, y=None):
        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}
        return self
    
    def transform(self, X, y=None):
        values = X.map(self.value_map_)
        
        possible_values = sorted(self.value_map_.values())
        
        idx1 = []
        idx2 = []
        
        all_indices = np.arange(len(X))
        
        for idx, val in enumerate(possible_values[:-1]):
            new_idxs = all_indices[values > val]
            idx1.extend(new_idxs)
            idx2.extend(repeat(idx, len(new_idxs)))
            
        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")
            
        return result
    
############################################################################################################################

# to get distance between 2 coordinates
def distance_calc(row):
    start = (row['REST_LAT'], row['REST_LONG'])
    stop = (12.972442, 77.580643)
    return vincenty(start, stop).km


def kmeans_clusterer(train_df, valid_df, n):
    clusterer = KMeans(n, random_state=1, init='k-means++')
    # fit the clusterer
    clusterer.fit(train_df)
    train_clusters = clusterer.predict(train_df)
    valid_clusters = clusterer.predict(valid_df)
    return train_clusters, valid_clusters

def kmeans_feats(train_df, valid_df, m=5):
    for i in range(2, m):
        t, v = kmeans_clusterer(train_df, valid_df, n=i)
        col_name = str('kmeans_' + str(i))
        t = pd.DataFrame({col_name: t})
        v = pd.DataFrame({col_name: v})
        train_df = pd.concat([train_df.reset_index(drop=True), t], axis=1)
        valid_df = pd.concat([valid_df.reset_index(drop=True), v], axis=1)
    return train_df, valid_df

################################################################################################################################
## PIPE example
### missing value imputation
fillna_zero_cols = ['cat1', 'cat2']
train[fillna_zero_cols] = train[fillna_zero_cols].astype(float)
test[fillna_zero_cols] = test[fillna_zero_cols].astype(float)
fillna_null_cols = list(train.select_dtypes(include=['object']).columns)
fillna_mean_cols = list(train.select_dtypes(include=[np.float64, np.int64]).columns)

pipe = pipe([
    # add a binary variable to indicate missing information for the 2 variables below
    ('continuous_var_imputer', msi.ArbitraryNumberImputer(arbitrary_number=0, variables = fillna_zero_cols)),
    # replace NA by the mean in the 3 variables below, they are numerical
    ('continuous_var_median_imputer', msi.MeanMedianImputer(imputation_method='median', variables = fillna_mean_cols)),
    # replace NA by adding the label "Missing" in categorical variables (transformer will skip those variables where there is no NA)
    ('categorical_imputer', msi.CategoricalVariableImputer(variables = fillna_null_cols))
])

pipe.fit(train)
train = pipe.transform(train)
test = pipe.transform(test)
################################################################################################################################
###########################################################################################################################################

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
    
###########################################################################################################################################

## HELPER FUNCTIONS CLASS ##

from IMPORT_MODULES import *

class helper_funcs():
    
    def __init__():
        """ helper functions used across the pipeline """
        return None
    
    ## find and append multiple dataframes of the type specified in string
    def append_datasets(cols_to_remove, string = ['train', 'valid']):
        # pass either train or valid as str argument
        temp_files = [name for name in os.listdir('../input/') if name.startswith(string)]
        temp_dict = {}
        for i in temp_files:
            df_name = re.sub(string=i, pattern='.csv', repl='')
            temp_dict[df_name] = pd.read_csv(str('../input/' + str(i)), na_values=['No Data', ' ', 'UNKNOWN'])
            temp_dict[df_name].columns = map(str.lower, temp_dict[df_name].columns)
            temp_dict[df_name].drop(cols_to_remove, axis = 1, inplace = True)
            chars_to_remove = [' ', '.', '(', ')', '__', '-']
            for i in chars_to_remove:
                temp_dict[df_name].columns = temp_dict[df_name].columns.str.strip().str.lower().str.replace(i, '_')
        temp_list = [v for k,v in temp_dict.items()]
        if len(temp_list) > 1 :
            temp = pd.concat(temp_list, axis=0, sort=True, ignore_index=True)
        else :
            temp = temp_list[0]
        return temp
    
    ## datetime feature engineering
    def datetime_feats(train, valid):
        cols = [s for s in train.columns.values if 'date' in s]
        print('datetime feature engineering is happening ...', '\n')
        # nested function to derive the various datetime features for a given date column
        def dt_feats(df, col):
            df[col] = pd.to_datetime(df[i])
            #df[str(col+'_'+'day')] = df[col].dt.day
            df[str(col+'_'+'day_name')] = df[col].dt.day_name
            #df[str(col+'_'+'dayofweek')] = df[col].dt.dayofweek
            df[str(col+'_'+'dayofyear')] = df[col].dt.dayofyear
            #df[str(col+'_'+'days_in_month')] = df[col].dt.days_in_month
            #df[str(col+'_'+'month')] = df[col].dt.month
            df[str(col+'_'+'month_name')] = df[col].dt.month_name
            df[str(col+'_'+'quarter')] = df[col].dt.quarter
            df[str(col+'_'+'week')] = df[col].dt.week
            #df[str(col+'_'+'weekday')] = df[col].dt.weekday
            df[str(col+'_'+'year')] = df[col].dt.year
            #df[col] = df[col].dt.date
            df = df.drop([col], axis = 1)
            return df
        # loop function over all raw date columns
        for i in cols:
            train = dt_feats(train, i)
            valid = dt_feats(valid, i)
        return train, valid
    
    ## function to get frequency count of elements in a vector/list
    def freq_count(input_vector):
        return collections.Counter(input_vector)
    
    ## function to make deviation encoding features
    def categ_feat_eng(train_df, valid_df, cat_columns):
        print('categorical feature engineering is happening ...', '\n')
        global iter
        iter = 0
        for i in tqdm(cat_columns):
            grouped_df = pd.DataFrame(train_df.groupby([i])['label'].agg(['mean', 'std'])).reset_index()
            grouped_df.rename(columns={'mean': str('mean_' + cat_columns[iter]),
                                       'std': str('std_' + cat_columns[iter])}, inplace=True)
            train_df = pd.merge(train_df, grouped_df, how='left')
            valid_df = pd.merge(valid_df, grouped_df, how='left')
            iter += 1
        return train_df, valid_df


#### LOOP BREAK FUNCTION ####
""" 
To allow early exit of loops or conditional statements to handle exceptions/errors
Allows exit() to work if script is invoked with IPython without
raising NameError Exception. Keeps kernel alive.   
"""

class IpyExit(SystemExit):
    """Exit Exception for IPython.

    Exception temporarily redirects stderr to buffer.
    """
    def __init__(self):
        # print("exiting")  # optionally print some message to stdout, too
        # ... or do other stuff before exit
        sys.stderr = StringIO()

    def __del__(self):
        sys.stderr.close()
        sys.stderr = sys.__stderr__  # restore from backup

def ipy_exit():
    raise IpyExit

if get_ipython():    # ...run with IPython
    exit = ipy_exit  # rebind to custom exit
else:
    exit = exit      # just make exit importable
    
###########################################################################################################################################

## helpers

import pandas as pd, numpy as np
from sklearn.feature_selection import VarianceThreshold

# removing near zero variance columns
def variance_threshold_selector(train, threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(train)
    X = train[train.columns[selector.get_support(indices=True)]]
#     print(f'features kicked: {len(train.columns) - len(X.columns)}')
    return X

# correlation function
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return dataset

## MAPE function
def mape(a, b): 
    mask = a != 0
    return (np.fabs(a - b)/a)[mask].mean() * 100

## json normalize product des
def jsonnorm_prod(df):
    df['product_des3'] = df['product_des2'].str.replace(r' "', '"').astype(str)
    df['product_des3'] = df['product_des2'].str.findall(r'("[a-zA-Z0-9 ]+": "[a-zA-Z0-9 ]+")').astype(str)
    df['product_des3'] = df['product_des3'].str.replace(r'[', '{').str.replace(r']', '}').str.replace("'", "").astype(str)
    df['product_des3'] = df['product_des3'].map(eval)
    df['product_des4'] = df['product_des2'].str.replace(r'[a-zA-Z0-9 ]+": "[a-zA-Z0-9 ]+', '').str.replace(r'"",,', '').astype(str)

    temp = json_normalize(df['product_des3'])
    temp.isna().sum().to_csv('missing.csv')

    temp = temp.add_prefix('attr_')
    temp.rename(columns=lambda x: x.strip(), inplace=True)
    temp = temp.fillna('.').groupby(temp.columns, axis=1).max()
    
    df = pd.concat([df.reset_index(drop=True), temp], axis=1)
    df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))
    df = df.replace('nan', '.')
    return df

# function to replace some string and get extract for comparison with base data
def match_product_title(obj_str):
    extracted = [i for i in brands if i in obj_str]
    if len(extracted)>0:
        extracted = max(extracted, key=len)
    else:
        extracted = ''
    return extracted

## trend functions
def trendline(index,data, order=1):
    coeffs = np.polyfit(index, list(data), order)
    slope = coeffs[-2]
    return float(slope)
def trend_fn(row):
    lst = [row['oct'], row['nov'], row['dec']]
    index = [1,2,3]
    return trendline(index, lst)
table['trend_3months'] = table.apply(lambda row: trend_fn(row), axis=1)

###########################################################################################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
from scipy.linalg import svd
from collections import defaultdict


class Imputer(object):
    def __init__(self):

    def drop(self, x, missing_data_cond):
        """ Drops all observations that have missing data

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        """

        # drop observations with missing values
        return x[np.sum(missing_data_cond(x), axis=1) == 0]

    def replace(self, x, missing_data_cond, in_place=False):
        """ Replace missing data with a random observation with data

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        """
        if in_place:
            data = x
        else:
            data = np.copy(x)

        for col in xrange(x.shape[1]):
            nan_ids = missing_data_cond(x[:, col])
            val_ids = np.random.choice(np.where(~nan_ids)[0],  np.sum(nan_ids))
            data[nan_ids, col] = data[val_ids, col]
        return data

    def summarize(self, x, summary_func, missing_data_cond, in_place=False):
        """ Substitutes missing values with a statistical summary of each
        feature vector

        Parameters
        ----------
        x : numpy.array
            Assumes that each feature column is of single type. Converts
            digit string features to float.
        summary_func : function
            Summarization function to be used for imputation
            (mean, median, mode, max, min...)
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # replace missing values with the summarization function
        for col in xrange(x.shape[1]):
            nan_ids = missing_data_cond(x[:, col])
            if True in nan_ids:
                val = summary_func(x[~nan_ids, col])
                data[nan_ids, col] = val

        return data

    def one_hot(self, x, missing_data_cond, weighted=False, in_place=False):
        """Create a one-hot row for each observation

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        weighted : bool
            Replaces one-hot by n_classes-hot.

        Returns
        -------
        data : np.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # find rows and columns with missing data
        _, miss_cols = np.where(missing_data_cond(data))
        miss_cols_uniq = np.unique(miss_cols)

        for miss_col in miss_cols_uniq:
            uniq_vals, indices = np.unique(data[:, miss_col],
                                           return_inverse=True)
            if weighted:
                data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                        dtype=int)[indices]*uniq_vals.shape[0]))
            else:
                data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                                     dtype=int)[indices]))

        # remove categorical columns with missing data
        data = np.delete(data, miss_cols, 1)
        return data

    def knn(self, x, k, summary_func, missing_data_cond, cat_cols,
            weighted=False, in_place=False):
        """ Replace missing values with the summary function of K-Nearest
        Neighbors

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        k : int
            Number of nearest neighbors to be used
        summary_func : function
            Summarization function to be used for imputation
            (mean, median, mode, max, min...)
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        cat_cols : int tuple
            Index of columns that are categorical
        """
        if in_place:
            data = x
        else:
            data = np.copy(x)

        imp = Imputer()

        # first transform features with categorical missing data into one hot
        data_complete = imp.one_hot(data, missing_data_cond, weighted=weighted)

        # binarize complete categorical variables and convert to int
        col = 0
        cat_ids_comp = []
        while col < max(cat_cols):
            if isinstance(data_complete[0, col], basestring) \
                    and not data_complete[0, col].isdigit():
                cat_ids_comp.append(col)
            col += 1

        data_complete = imp.binarize_data(data_complete,
                                          cat_ids_comp).astype(float)

        # normalize features
        scaler = StandardScaler().fit(data_complete)
        data_complete = scaler.transform(data_complete)
        # create dict with missing rows and respective columns
        missing = defaultdict(list)
        map(lambda (x, y): missing[x].append(y),
            np.argwhere(missing_data_cond(data)))
        # create mask to build NearestNeighbors with complete observations only
        mask = np.ones(len(data_complete), bool)
        mask[missing.keys()] = False
        # fit nearest neighbors and get knn ids of missing observations
        print 'Computing k-nearest neighbors'
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(
            data_complete[mask])
        ids = nbrs.kneighbors(data_complete[missing.keys()],
                              return_distance=False)

        def substituteValues(i):
            row = missing.keys()[i]
            cols = missing[row]
            data[row, cols] = mode(data[mask][ids[i]][:, cols])[0].flatten()

        print 'Substituting missing values'
        map(substituteValues, xrange(len(missing)))
        return data

    def predict(self, x, cat_cols, missing_data_cond, clf, inc_miss=True,
                in_place=False):
        """ Uses random forest for predicting missing values

        Parameters
        ----------
        cat_cols : int tuple
            Index of columns that are categorical
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        clf : object
            Object with fit and predict methods, e.g. sklearn's Decision Tree
        inc_miss : bool
            Include missing data in fitting the model?
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # find rows and columns with missing data
        miss_rows, miss_cols = np.where(missing_data_cond(data))
        miss_cols_uniq = np.unique(miss_cols)

        if inc_miss:
            valid_cols = np.arange(data.shape[1])
        else:
            valid_cols = [n for n in xrange(data.shape[1])
                          if n not in miss_cols_uniq]

        # factorize valid cols
        data_factorized = np.copy(data)

        # factorize categorical variables and store transformation
        factor_labels = {}
        for cat_col in cat_cols:
            # factors, labels = pd.factorize(data[:, cat_col])
            labels, factors = np.unique(data[:, cat_col], return_inverse=True)
            factor_labels[cat_col] = labels
            data_factorized[:, cat_col] = factors

        # values are integers, convert accordingly
        data_factorized = data_factorized.astype(int)

        # update each column with missing features
        for miss_col in miss_cols_uniq:
            # extract valid observations given current column missing data
            valid_obs = [n for n in xrange(len(data))
                         if data[n, miss_col] != '?']

            # prepare independent and dependent variables, valid obs only
            data_train = data_factorized[:, valid_cols][valid_obs]
            y_train = data_factorized[valid_obs, miss_col]

            # train random forest classifier
            clf.fit(data_train, y_train)

            # given current feature, find obs with missing vals
            miss_obs_iddata = miss_rows[miss_cols == miss_col]

            # predict missing values
            y_hat = clf.predict(data_factorized[:, valid_cols][miss_obs_iddata])

            # replace missing data with prediction
            data_factorized[miss_obs_iddata, miss_col] = y_hat

        # replace values on original data
        for col in factor_labels.keys():
            data[:, col] = factor_labels[col][data_factorized[:, col]]

        return data

    def factor_analysis(self, x, cat_cols, missing_data_cond, threshold=0.9,
                        technique='SVD', in_place=False):
        """ Performs low-rank matrix approximation via dimensioality reduction
        and replaces missing data with values obtained from the data projected
        onto N principal components or singular values or eigenvalues...

        cat_cols : int tuple
            Index of columns that are categorical
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        threshold : float
            Variance threshold that must be explained by eigen values.
        technique : str
            Technique used for low-rank approximation. 'SVD' is supported
        """

        def _mode(d):
            return mode(d)[0].flatten()

        if in_place:
            data = x
        else:
            data = np.copy(x)

        data_summarized = self.summarize(x, _mode, missing_data_cond)

        # factorize categorical variables and store encoding
        factor_labels = {}
        for cat_col in cat_cols:
            labels, factors = np.unique(data_summarized[:, cat_col],
                                        return_inverse=True)
            factor_labels[cat_col] = labels
            data_summarized[:, cat_col] = factors

        data_summarized = data_summarized.astype(float)
        if technique == 'SVD':
            lsvec, sval, rsvec = svd(data_summarized)
            # find number of singular values that explain 90% of variance
            n_singv = 1
            while np.sum(sval[:n_singv]) / np.sum(sval) < threshold:
                n_singv += 1

            # compute low rank approximation
            data_summarized = np.dot(
                lsvec[:, :n_singv],
                np.dot(np.diag(sval[:n_singv]), rsvec[:n_singv, ]))
        else:
            raise Exception("Technique {} is not supported".format(technique))

        # get missing data indices
        nans = np.argwhere(missing_data_cond(x))

        # update data given projection
        for col in np.unique(nans[:, 1]):
            obs_ids = nans[nans[:, 1] == col, 0]
            # clip low rank approximation to be within factor labels
            proj_cats = np.clip(
                data_summarized[obs_ids, col], 0, len(factor_labels[col])-1)
            # round categorical variable factors to int
            proj_cats = proj_cats.round().astype(int)
            data[obs_ids, col] = factor_labels[col][proj_cats]

        return data

    def factorize_data(self, x, cols, in_place=False):
        """Replace column in cols with factors of cols

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data
        cols: tuple <int>
            Index of columns with categorical data

        Returns
        -------
        d : np.ndarray
            Matrix with categorical data replaced with factors
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        factors_labels = {}
        for col in cols:
            # factors, labels = pd.factorize(data[:, col])
            labels, factors = np.unique(data[:, col], return_inverse=True)
            factors_labels[col] = labels
            data[:, col] = factors

        return data, factors_labels

    def binarize_data(self, x, cols, miss_data_symbol=False,
                      one_minus_one=True, in_place=False):
        """Replace column in cols with one-hot representation of cols

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        cols: tuple <int>
            Index of columns with categorical data

        Returns
        -------
        d : np.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)
        for col in cols:
            uniq_vals, indices = np.unique(data[:, col], return_inverse=True)

            if one_minus_one:
                data = np.column_stack(
                    (data,
                     (np.eye(uniq_vals.shape[0], dtype=int)[indices] * 2) - 1))
            else:
                data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                                     dtype=int)[indices]))
            # add missing data column to feature
            if miss_data_symbol is not False and \
                    miss_data_symbol not in uniq_vals:
                data = np.column_stack(
                    (data, -one_minus_one * np.ones((len(data), 1), dtype=int)))

        # remove columns with categorical variables
        val_cols = [n for n in xrange(data.shape[1]) if n not in cols]
        data = data[:, val_cols]
        return data
###########################################################################################################################################

## MILLIFY
import math
millnames = ['',' Thousand',' Million']
def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])



## CORR PLOT
corr = df[ind_vars + [depvar]].corr()
cmap=sns.diverging_palette(5, 250, as_cmap=True)
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '50px'),
                        ('font-size', '12pt')])
]
display(corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '40px', 'font-size': '8pt', 'max-height' : '200px'})\
    .set_caption("Hover to magnify")\
    .set_precision(2)\
    .set_table_styles(magnify()))


def std_scaler(v):
    scaler = StandardScaler()
    return scaler.fit_transform(v.reshape(-1, 1))

                
def anova_fn(self, v1, v2):
    stat, p = f_oneway(v1, v2)
    if p < 0.05:
        return 'group means are similar. the binary feature adds no value', p, ('stat=%.3f, p=%.4f' % (stat, p))
    else:
        return 'group means are dissimilar. feature has relationship with dependant variable', p, ('stat=%.3f, p=%.4f' % (stat, p))    

###########################################################################################################################################

#### SAMPLING ####
#- Oversampling (ADASYN, SMOTE)
#- Undersampling (ENN, RENN, AllKNN)
#- Oversampling and then Undersampling (SMOTE and ENN/TOMEK)

#*it's okay if you have no idea what the above mean. the only thing that is important is to understand why over/undersampling
#is done and why or what ratio between*
#    - why over/under sampling is done in a classification context
#    - what ratio between the 2 classes is important to You in your context
#    - how much information loss (or gain) are you willing to tolerate? (create More data than what you have at hand?)

#""" Explicitly doing sampling. Use with care if going ahead with the CV based approach. Keep ratio low if so (recommended)

#oversampling the minority class using techniques from SMOTE (for oversampling) and ENN/Tomek (for undersampling/cleaning)
#ENN worked out better than Tomek
#added support for undersampling with ENN/RENN/AllKNN """

from IMPORT_MODULES import *

def sampler(X_train, y_train, which='smote_enn', frac=0.75):
    """ which = ['adasyn', smote_tomek', 'smote_enn', 'enn', 'renn', 'allknn'] """
    
    feat_names = X_train.columns.values
    print('Sampling is being done..\n')

    ### OVERSAMPLING (ADASYN) ###
    if which=='adasyn':
        # Apply ADASYN
        ada = ADASYN(random_state=0)
        X_train, y_train = ada.fit_sample(X_train, y_train)

    ### OVERSAMPLING (SMOTE) AND THEN UNDERSAMPLING (ENN/Tomek) ###
    if which=='smote_tomek':
        # Apply SMOTE + Tomek links
        sm = SMOTETomek(random_state=0, ratio=frac)
        X_train, y_train = sm.fit_sample(X_train, y_train)
    if which=='smote_enn':
        # Apply SMOTE + ENN
        smote_enn = SMOTEENN(random_state=0, ratio=frac)
        X_train, y_train = smote_enn.fit_sample(X_train, y_train)

    ### UNDERSAMPLING (ENN/RENN/AllKNN) ###
    if which=='enn':
        # Apply ENN
        enn = EditedNearestNeighbours(random_state=0)
        X_train, y_train = enn.fit_sample(X_train, y_train)
    if which=='renn':
        # Apply RENN
        renn = RepeatedEditedNearestNeighbours(random_state=0)
        X_train, y_train = renn.fit_sample(X_train, y_train)
    if which=='allknn':
        # Apply AllKNN
        allknn = AllKNN(random_state=0)
        X_train, y_train = allknn.fit_sample(X_train, y_train)

    X_train = pd.DataFrame(data=X_train,columns=feat_names)
    print(X_train.shape, y_train.shape, collections.Counter(y_train))
    
    return X_train, y_train

###########################################################################################################################################

####################################################
## EDA functions script ##
####################################################

# clear the workspace
%reset -f

import pandas as pd
import numpy as np
#import xgboost as xgb
import pickle, collections
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# print list of files in directory
import os
print(os.listdir())

# print/display all plots inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pd.options.display.max_columns=100
pd.options.display.max_rows=1000

####################################################

#define a function to return all the stats required for univariate analysis of continuous variables
def univariate_stats_continuous(df_raw_data, var_cont):

    #for each column, check the following -> 1) number of rows in each variable, 2) number of rows with missing values and 3) % of rows with missing values
    df_variable_stats = pd.DataFrame(df_raw_data[var_cont].dtypes).T.rename(index={0:'column type'})
    df_variable_stats = df_variable_stats.append(pd.DataFrame(df_raw_data[var_cont].isnull().sum()).T.rename(index={0:'null values (nb)'}))
    df_variable_stats = df_variable_stats.append(pd.DataFrame(df_raw_data[var_cont].isnull().sum()/df_raw_data[var_cont].shape[0])
                             .T.rename(index={0:'null values (%)'}))
    
    #get stats for every continuous variable 
    df_variable_stats = df_variable_stats.append(df_raw_data[var_cont].agg(['count', 'size', 'nunique', 'mean','median','std', 'var', 'skew', 'kurtosis', 'min', 'max']))
    
    #get mode for every variable - manual since there were some unresolved errors
    temp_list_1 = []
    temp_list_2 = []
    for i in list(df_raw_data[var_cont].columns):
        #print(i)
        temp_list_1.append(df_raw_data[i].mode()[0])
        temp_list_2.append(i)
    temp_list_1 = pd.Series(temp_list_1)
    temp_list_1.index = temp_list_2
    temp_list_1.name = 'mode'
    
    df_variable_stats = df_variable_stats.append(pd.DataFrame(temp_list_1).T)

    def return_percentile(df_name, percentile_array, index_array):
        """
        This function returns different percentiles for all the columns of a given DataFrame
        This function is built to function only for continuous variables
        """
        df_quantile = df_name.quantile(percentile_array)
        df_quantile['rows'] = index_array
        df_quantile = df_quantile.reset_index()
        df_quantile.drop('index', axis=1, inplace=True)
        df_quantile.set_index(['rows'], inplace=True)
        
        return df_quantile
    
    percentile_array = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.25,0.3,0.33,0.4,0.5,0.6,0.66,0.7,0.75,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]
    index_array = ['0%','1%','2%','3%','4%','5%','6%','7%','8%','9%','10%','20%','25%','30%','33%','40%','50%','60%','66%','70%','75%','80%','90%','91%','92%','93%','94%','95%','96%','97%','98%','99%','100%']
    
    df_quantile = return_percentile(df_raw_data[var_cont], percentile_array, index_array)

    df_variable_stats = df_variable_stats.append(df_quantile).T

    df_variable_stats.reset_index(inplace=True)
    df_variable_stats.drop('column type', axis=1, inplace=True)
    df_variable_stats.dtypes
    
    df_variable_stats = df_variable_stats[['index','nunique','null values (nb)','null values (%)','mean','median','mode','std','var','max','min','count','kurtosis','skew','0%','1%','2%','3%','4%','5%','6%','7%','8%','9%','10%','20%','25%','30%','33%','40%','50%','60%','66%','70%','75%','80%','90%','91%','92%','93%','94%','95%','96%','97%','98%','99%','100%']]
    df_variable_stats.columns = ['Variable','Unique values','Missing values','Missing percent','Mean','Median','Mode','Std. Dev.','Variance','Max','Min','Range','Kurtosis','Skewness','0%','1%','2%','3%','4%','5%','6%','7%','8%','9%','10%','20%','25%','30%','33%','40%','50%','60%','66%','70%','75%','80%','90%','91%','92%','93%','94%','95%','96%','97%','98%','99%','100%']

    #return the final dataframe containing stats for continuous variables
    return df_variable_stats

# var_cont = train.select_dtypes(include=['int64', 'float64']).columns.values
# df_stats_1 = univariate_stats_continuous(df_raw_data=train, var_cont=var_cont)
# display(df_stats_1)

####################################################

#define a function to return all the stats required for univariate analysis of continuous variables
def univariate_stats_categorical(df_raw_data, var_catg):

    #get the unique values of the variables
    df_catg_nunique = df_raw_data[var_catg].nunique().reset_index()
    df_catg_nunique.columns = ['Variable', 'unique_values']
    
    #get the population for different observations of each variable
    df_catg_population = pd.DataFrame(columns = ['Variable', 'Level', 'Population'])
    
    for i in df_raw_data[var_catg].columns:
        df_temp = pd.DataFrame(df_raw_data[i].value_counts()).reset_index()
        df_temp['Variable'] = i
        df_temp = df_temp[['Variable', 'index', i]]
        df_temp.columns = ['Variable', 'Level', 'Population']
        df_catg_population = df_catg_population.append(df_temp)
    
    #merge the population and unique counts
    df_catg_stats = pd.merge(df_catg_population, df_catg_nunique, on = 'Variable', how = 'left')

    df_catg_stats['Population %'] = df_catg_stats.groupby(['Variable'])['Population'].apply(lambda x: 100 * x / float(x.sum()))

    return df_catg_stats

# var_cat = train.select_dtypes(include=['object']).columns.values
# df_stats_2 = univariate_stats_categorical(df_raw_data=train, var_catg=var_cat)
# display(df_stats_2)

####################################################

#create a function to give average value of dependent variable for every observation of categorical variables
def bivariate_stats_categorical(df_raw_data, var_catg, var_dependent):
    global iter
    iter = 0
    all_cols = pd.DataFrame(columns = ['col', 'level', 'mean', 'std'])
    for i in tqdm(var_catg):
        grouped_df = pd.DataFrame(df_raw_data.groupby([i])[var_dependent].agg(['mean', 'std'])).reset_index()
        grouped_df.columns = ['level', 'mean', 'std']
        grouped_df['col'] = str(i)
        iter += 1
        
        all_cols = all_cols.append(grouped_df, ignore_index=True)
    return all_cols

# var_cat = list(train.select_dtypes(include=['object']).columns)
# df_stats_3 = bivariate_stats_categorical(train, var_cat, 'label')
# display(df_stats_3)

####################################################
###########################################################################################################################################

# function to return interpretation across methods (specify which or All)

class model_interpret():
    
    def __init__():
        """ this module takes as input the model and train/test datasets to generate interpretations of the
        predictions generated by the model
        LIME and SHAP methods have been added as a provision currently, treeinterpreter will be added later
        """
    
    def lime_interpreter(feat_names, classnames, categindices, categnames, 
                         kw, num_feature, train, test, n):
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data = train.values,
                                                   feature_names = list(feat_names),
                                                   class_names = classnames,
                                                   categorical_features=categindices, 
                                                   categorical_names=categnames, kernel_width = kw)
        xtest = test.values
        exp = explainer.explain_instance(xtest[n], model.predict_proba, num_features = num_feature)
        return exp.show_in_notebook()
    
    def shap_interpreter(model, train, test, n, method = 'tree'):
        """ specify n as the prediction/observation you want the interpretation to be returned for """
        
        if method == 'tree':
            # create our SHAP explainer
            shap_explainer = shap.TreeExplainer(model)
            # calculate the shapley values for our test set
            shap_values = shap_explainer.shap_values(test.values)
        elif method == 'kernel':
            # create our SHAP explainer
            shap_explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(train[:100], 5))
            shap_values = shap_explainer.shap_values(test.values)
            
        # load JS in order to use some of the plotting functions from the shap package in the notebook
        shap.initjs()
        
        # plot the explanation for a single prediction
        return shap.force_plot(shap_values[n, :], test.iloc[n, :])
    
    def model_interpreter(interpreter_algo, train, test, shap_method = 'tree', kw = 3, n = 0, model = None,
                          feat_names = None, classnames = None,
                          categindices = None, categnames = None, num_feature = None):
        if interpreter_algo == 'lime':
            return model_interpret.lime_interpreter(feat_names, classnames, categindices, categnames,
                                                    kw, num_feature, train, test, n)
        elif interpreter_algo == 'shap':
            return model_interpret.shap_interpreter(model, train, test, n, method = shap_method)
        
model = model_selection_with_tuning.model_selection(modelling_algo='xgb', X_train=X_train, X_test=X_test, y_train=y_train)

model_interpret.model_interpreter(interpreter_algo='shap', model = model, train = X_train, test = X_test, feat_names = feature_names,
                                  classnames = ['not delayed', 'delayed'], categindices = categ_idx, categnames = categ_names,
                                 num_feature = num_feature, n=0)

###########################################################################################################################################

class feat_eng():
    
    def __init__():
        """ this module contains several functions for creating new features. find below a brief description of each """
    
    def scalers(train, valid, which_method):
        if which_method == 'ss':
            sc = StandardScaler()
            sc.fit(train)
            train = pd.DataFrame(sc.transform(train), columns=train.columns.values)
            valid = pd.DataFrame(sc.transform(valid), columns=valid.columns.values)
            return train, valid # scale all variables to zero mean and unit variance, required for PCA and related
        if which_method == 'mm':
            mm = MinMaxScaler()
            mm.fit(train)
            train = pd.DataFrame(mm.transform(train), columns=train.columns.values)
            valid = pd.DataFrame(mm.transform(valid), columns=train.columns.values)
            return train, valid # use this method to iterate
    
    def decomp_various():
        return None
    
    def pca_feats(train, valid, n = .95):
            train, valid = feat_eng.scalers(train, valid, which_method='ss')
            pca_fit = decomposition.PCA(n_components=n)
            pca_fit.fit(train)
            pca_train = pd.DataFrame(pca_fit.transform(train))
            pca_valid = pd.DataFrame(pca_fit.transform(valid))
            pca_cols = list(set(list(pca_train)))
            pca_cols = ['pca_' + str(s) for s in pca_cols]
            pca_train.columns = pca_cols
            pca_valid.columns = pca_cols
            return pca_train, pca_valid
        
    def ica_feats(train, valid, n = 5):
            train, valid = feat_eng.scalers(train, valid, which_method='ss')
            ica_fit = decomposition.FastICA(n_components=n)
            ica_fit.fit(train)
            ica_train = pd.DataFrame(ica_fit.transform(train))
            ica_valid = pd.DataFrame(ica_fit.transform(valid))
            ica_cols = list(set(list(ica_train)))
            ica_cols = ['ica_' + str(s) for s in ica_cols]
            ica_train.columns = ica_cols
            ica_valid.columns = ica_cols
            return ica_train, ica_valid
        
    def tsvd_feats(train, valid, n = 5):
            train, valid = feat_eng.scalers(train, valid, which_method='ss')
            tsvd_fit = decomposition.TruncatedSVD(n_components=n)
            tsvd_fit.fit(train)
            tsvd_train = pd.DataFrame(tsvd_fit.transform(train))
            tsvd_valid = pd.DataFrame(tsvd_fit.transform(valid))
            tsvd_cols = list(set(list(tsvd_train)))
            tsvd_cols = ['tsvd_' + str(s) for s in tsvd_cols]
            tsvd_train.columns = tsvd_cols
            tsvd_valid.columns = tsvd_cols
            return tsvd_train, tsvd_valid
        
    def grp_feats(train, valid, n = 5):
            train, valid = feat_eng.scalers(train, valid, which_method='ss')
            grp_fit = GaussianRandomProjection(n_components=n, eps=0.3)
            grp_fit.fit(train)
            grp_train = pd.DataFrame(grp_fit.transform(train))
            grp_valid = pd.DataFrame(grp_fit.transform(valid))
            grp_cols = list(set(list(grp_train)))
            grp_cols = ['grp_' + str(s) for s in grp_cols]
            grp_train.columns = grp_cols
            grp_valid.columns = grp_cols
            return grp_train, grp_valid
    
    def srp_feats(train, valid, n = 5):
            train, valid = feat_eng.scalers(train, valid, which_method='ss')
            srp_fit = SparseRandomProjection(n_components=n, dense_output=True, eps=0.3)
            srp_fit.fit(train)
            srp_train = pd.DataFrame(srp_fit.transform(train))
            srp_valid = pd.DataFrame(srp_fit.transform(valid))
            srp_cols = list(set(list(srp_train)))
            srp_cols = ['srp_' + str(s) for s in srp_cols]
            srp_train.columns = srp_cols
            srp_valid.columns = srp_cols
            return srp_train, srp_valid
        
    def return_combined(train, valid, list_objects = ['pca', 'ica', 'tsvd', 'grp', 'srp']):
        if 'pca' in list_objects:
            train = pd.concat([train.reset_index(drop=True), pca_train], axis=1)
            valid = pd.concat([valid.reset_index(drop=True), pca_valid], axis=1)
        if 'ica' in list_objects:
            train = pd.concat([train.reset_index(drop=True), ica_train], axis=1)
            valid = pd.concat([valid.reset_index(drop=True), ica_valid], axis=1)
        if 'tsvd' in list_objects:
            train = pd.concat([train.reset_index(drop=True), tsvd_train], axis=1)
            valid = pd.concat([valid.reset_index(drop=True), tsvd_valid], axis=1)
        if 'grp' in list_objects:
            train = pd.concat([train.reset_index(drop=True), grp_train], axis=1)
            valid = pd.concat([valid.reset_index(drop=True), grp_valid], axis=1)
        if 'srp' in list_objects:
            train = pd.concat([train.reset_index(drop=True), srp_train], axis=1)
            valid = pd.concat([valid.reset_index(drop=True), srp_valid], axis=1)
        return train, valid

###########################################################################################################################################

## k-means clustering features

from sklearn.cluster import KMeans

class kmeans_feats():
    def __init__():
        """ module for adding features based on kmeans clusters generated """
    
    def clusterer(train_df, valid_df, n):
        clusterer = KMeans(n, random_state=1, init='k-means++')
        
        # fit the clusterer
        clusterer.fit(train_df)
        
        train_clusters = clusterer.predict(train_df)
        valid_clusters = clusterer.predict(valid_df)
        
        return train_clusters, valid_clusters
    
    def combine(train_df, valid_df, m=5):
        for i in range(2, m):
            t, v = kmeans_feats.clusterer(train_df, valid_df, n=i)
            col_name = str('kmeans_'+ str(i))
            t = pd.DataFrame({col_name: t})
            v = pd.DataFrame({col_name: v})
            
            train_df = pd.concat([train_df.reset_index(drop=True), t], axis=1)
            valid_df = pd.concat([valid_df.reset_index(drop=True), v], axis=1)
            
        return train_df, valid_df

###########################################################################################################################################
p, r, thresholds = metrics.precision_recall_curve(y_true=y_valid, probas_pred=xgb_pred)

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
###########################################################################################################################################
import os

def Is64Windows():
    return 'PROGRAMFILES(X86)' in os.environ

def GetProgramFiles32():
    if Is64Windows():
        return os.environ['PROGRAMFILES(X86)']
    else:
        return os.environ['PROGRAMFILES']

def GetProgramFiles64():
    if Is64Windows():
        return os.environ['PROGRAMW6432']
    else:
        return None
###########################################################################################################################################
## instructions to setup a virtualenv using pipenv and open a jupyter notebook with the created kernel

1. Install pipenv
	- python -m pip install pipenv
2. Activate the virtualenv
	- pipenv install
	- this will setup the pipfile and activate the virtualenv
	- execute "pipenv shell" to open up a sub shell of the virtualenv created
3. Install packages using --skip-lock argument since piplock takes too much time
	- pipenv install --skip-lock jupyter
	- then execute > ipython kernel install --name=project1 (project1 = name of project/kernel you want to use)
	- then execute > jupyter notebook (it starts with default python kernel from the virtual env)
	- dynamic code for the same is "python -m ipykernel install --user --name=`basename $VIRTUAL_ENV`"
4. For enabling variable inspector
	- jupyter nbextension enable varInspector/main
5. For enabling jupyter themes
	- jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T
	- jt -t grade3 -T -N (this is for toggling toolbar)
    
## instructions to setup and use Docker with the jupyter/pyspark-notebook image

1. Download and install the Docker community edition version
	- https://store.docker.com/editions/community/docker-ce-desktop-windows
2. Enable virtualization in Windows
	- https://support.lenovo.com/in/en/solutions/ht500216
	- follow steps above to enable it the recommended way
3. Open Docker and restart to enable the changes
4. Open a windows terminal (admin mode) and type the below command
	- docker pull jupyter/pyspark-notebook
5. It will take a while to download and set it up (around 6gb)
6. When done, close and open a new terminal and type below command
	- docker run -p 8888:8888 jupyter/pyspark-notebook
	- the above command means
		- [docker (keyword to specify it is a docker command)] [run (run command)] [-p (publish command)] [8888 (host post) : 8888 (container port)] [jupyter/pyspark-notebook (image name)]
7. This will create a container from the specified image, and start a session on localhost:8888 that can communicate with the 8888 port of the container
	- open your browser and go to localhost:8888/?token="key"
	- the link is automatically displayed in the terminal once the container is running
8. Create a new jupyter notebook and execute the below statements
	- import pyspark
	- sc = pyspark.SparkContext('local[*]')
	- do something to prove it works
        - rdd = sc.parallelize(range(1000))
        - rdd.takeSample(False, 5)
9. If it returns an output, everything is working fine
10. Keep in mind this is for setting up a temporary container in memory and not for production. You will need to create a folder elsewhere in your drive and add it to the container repository to retain work done during the session (refer to below link for instructions on that)
	- https://www.dataquest.io/blog/docker-data-science/
11. To mount a local folder to the docker and run execute the below
	- docker run -p 8888:8888 -v d:/python:/home/jovyan/work jupyter/pyspark-notebook


### Notes:
1. You need to stop/remove containers when shutting down to avoid sharing memory when not using it
	- manual method is execute the below steps:
		- docker ps (will return the running/stopped containers)
		- get the container id of the one you want to stop
		- docker stop [container id]
		- docker rm [container id]
2. To close All the running/stopped containers, create a "docker remove all.bat" textfile with the below contents
	@ECHO OFF
	FOR /f "tokens=*" %%i IN ('docker ps -aq') DO docker stop %%i
3. This will remove all existing containers. You can also simply "stop" a container using its id if you want to retain it
	- docker stop [container id]
	- container id can be acquired by typing "docker ps" in the terminal
4. You can display the downloaded images by executing "docker images"
###########################################################################################################################################

###########################################################################################################################################