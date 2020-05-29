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