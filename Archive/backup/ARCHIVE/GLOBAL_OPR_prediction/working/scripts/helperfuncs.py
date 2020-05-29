## helperfuncs.py

import pandas as pd, numpy as np

# HELPER FUNCTIONS CLASS #
class helper_funcs():

    def __init__(self):
        """ list down the various functions defined here """
    
    def csv_read(self, file_path, cols_to_keep=None, dtype=None, drop_dup=None):
        self.cols_to_keep = cols_to_keep
        if dtype is None:
            x=pd.read_csv(file_path, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], encoding='latin-1', low_memory=False)
        else:
            x=pd.read_csv(file_path, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], encoding='latin-1', low_memory=False, dtype=dtype)
        chars_to_remove = [' ', '\.', '\(', '\)', '\-', '\/', '\'', '\:', '\%']
        for i in chars_to_remove: x.columns = x.columns.str.strip().str.lower().str.replace(i, '_').str.replace('\_+', '_')
        if cols_to_keep is not None: x = x[cols_to_keep]
        if drop_dup is not None: x.drop_duplicates(inplace=True)
        print(x.shape)
        return x
    
    def txt_read(self, file_path, cols_to_keep=None, sep='|', skiprows=1, dtype=None, drop_dup=None):
        # currently only supports salary files with the default values (need to implement dynamic programming for any generic txt)
        self.cols_to_keep = cols_to_keep
        if dtype is None:
            x=pd.read_table(file_path, sep=sep, skiprows=skiprows, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'])
        else:
            x=pd.read_table(file_path, sep=sep, skiprows=skiprows, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], dtype=dtype)
        chars_to_remove = [' ', '\.', '\(', '\)', '\-', '\/', '\'', '\:', '\%']
        for i in chars_to_remove: x.columns = x.columns.str.strip().str.lower().str.replace(i, '_').str.replace('\_+', '_')
        if cols_to_keep is not None: x = x[cols_to_keep]
        if drop_dup is not None: x.drop_duplicates(inplace=True)
        print(x.shape)
        return x

    def xlsx_read(self, file_path, cols_to_keep=None, sheet_name=0, dtype=None, drop_dup=None):
        self.cols_to_keep = cols_to_keep
        if dtype is None:
          x=pd.read_excel(file_path, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], sheet_name=sheet_name)
        else:
          x=pd.read_excel(file_path, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], sheet_name=sheet_name, dtype=dtype)
        chars_to_remove = [' ', '\.', '\(', '\)', '\-', '\/', '\'', '\:', '\%']
        for i in chars_to_remove: x.columns = x.columns.str.strip().str.lower().str.replace(i, '_').str.replace('\_+', '_')
        if cols_to_keep is not None: x = x[cols_to_keep]
        if drop_dup is not None: x.drop_duplicates(inplace=True)
        print(x.shape)
        return x
    
    def process_columns(self, df, cols=None):
        if cols is None:
            df = df.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.strip() if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.replace('\s+|\s', '_', regex=True) if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.replace('[^\w+\s+]', '_', regex=True) if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.replace('\_+', '_', regex=True) if (x.dtype == 'object') else x)
        else:
            df[cols] = df[cols].apply(lambda x: x.str.lower())
            df[cols] = df[cols].apply(lambda x: x.str.strip())
            df[cols] = df[cols].apply(lambda x: x.str.replace('\s+|\s', '_', regex=True))
            df[cols] = df[cols].apply(lambda x: x.str.replace('[^\w\s]+', '_', regex=True))
            df[cols] = df[cols].apply(lambda x: x.str.replace('\_+', '_', regex=True))
        return df
  
    def nlp_process_columns(self, df, nlp_cols):
        df[nlp_cols] = df[nlp_cols].apply(lambda x: x.str.replace('_', ' '))
        df[nlp_cols] = df[nlp_cols].apply(lambda x: x.str.replace('\s+', ' ', regex=True))
        df[nlp_cols] = df[nlp_cols].apply(lambda x: x.str.replace('crft', 'craft'))
        return df
    
    def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]
            
    def getduplicates(df, idcol):
        return pd.concat(g for _, g in df.groupby(idcol) if len(g) > 1)

    def group_and_get_missingcount(df, grp_cols, missingcount_col):
        return df.groupby(grp_cols)[missingcount_col].apply(lambda x: x.isna().sum()/len(x)*100)
