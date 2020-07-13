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

# for i in nlp_cols:
#     df[i] = df[i].str.replace(',', ' ')
#     df[i] = df[i].str.replace(r's |s$', ' ', regex=True).str.strip()
#     df[i] = df[i].str.replace('(?=[A-Z])', ' ', regex=True).str.strip()
#     df[i] = (df[i].str.split()
#                               .apply(lambda x: OrderedDict.fromkeys(x).keys())
#                               .str.join(' '))
# df['string_all'] = df.apply(lambda x: ' '.join(x.dropna()), axis=1)
# df = df[['string_all']]

# vectorizer = TfidfVectorizer(max_features=10, ngram_range=(1,3), stop_words='english',
#                                strip_accents='unicode', analyzer='word')
# df = pd.DataFrame(vectorizer.fit_transform(df.string_all).todense())
# df.columns = vectorizer.get_feature_names()

# ads_allclasses = pd.concat([ads_allclasses.reset_index(drop=True), df], axis=1)


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

