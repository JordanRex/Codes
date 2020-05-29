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