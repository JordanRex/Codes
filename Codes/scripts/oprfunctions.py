## oprfunctions.py

import pandas as pd, numpy as np

## demographics function
def demo_fn(df, time, year, idcol='global_id', mngr_variant=None):
    df2 = df.groupby([idcol, 'date_of_birth', 'original_hire_date']).size().to_frame(name='count').reset_index()
    df2.sort_values([idcol, 'count'], ascending=[True, False], inplace=True)
    df2.drop_duplicates(subset=[idcol], inplace=True)
    df2.drop(['count'], axis=1, inplace=True)
    df2.columns = [idcol, 'DOB', 'OHD']
    df2['year_current'] = pd.to_datetime(time, format='%Y%m')
    df2['age'] = (df2.year_current - df2.DOB).astype('timedelta64[D]')
    df2['tenure'] = (df2.year_current - df2.OHD).astype('timedelta64[D]')
    df2['year'] = year
    df2 = df2[[idcol, 'age', 'tenure', 'year']]
    if mngr_variant==None:
        df2.columns = ['global_id', 'emp_age_asof_current', 'emp_tenure_asof_current', 'year']
    elif mngr_variant=='yes':
        df2.columns = ['direct_manager_emp_id', 'mngr_age_asof_current', 'mngr_tenure_asof_current', 'year']
    return df2

## salary processing function
def salary_process(df, col, curr_dict):
    df[col].fillna('usd', inplace=True)
    df['rate'] = df['currency_key'].map(curr_dict)
    df['mod_salary'] = df[col]*df['rate']
    df.drop(['rate', col, 'currency_key'], inplace=True, axis=1)
    return df
