## miscellaneous script
### 1. MRS
### 2. Salary
### 3. TeamSize

### MRS ###
mrsdf = bp_full[['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 
                 'employee_band', 'pay_scale_level', 'annual_salary', 'year']].copy()
mrscols = ['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 'employee_band', 'pay_scale_level']
mrsdf[mrscols] = mrsdf[mrscols].fillna(value='none')
bp_full[mrscols] = bp_full[mrscols].fillna(value='none')
mrsdf2 = mrsdf.groupby(['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 
                 'employee_band', 'pay_scale_level', 'year']).agg({'annual_salary':['min', 'max']}).reset_index()
mrsdf2.columns = mrsdf2.columns.droplevel(1)
mrsdf2.columns = ['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 
                 'employee_band', 'pay_scale_level', 'year', 'mingrade_new', 'maxgrade_new']
# mrs_data[mrs_data.duplicated(subset=['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 
#                  'employee_band', 'pay_scale_level', 'startdate'], keep=False)]

with adlsfsc.open(path + '/2019/Data/Raw_Data/sharp/mrs/salary_mrs_data.xlsx') as f:
    mrs_data = helpers.xlsx_read(f, 
                                 dtype='object',
                                 cols_to_keep=['cgrpg', 'pgt', 'pga', 'grade', 'lvl', 
                                               'start_date', 'end_date', 'min_grade_level', 'max_grade_level', 'reference_salary'])
mrs_data.columns = ['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 
                 'employee_band', 'pay_scale_level', 'startdate', 'enddate', 'mingrade', 'maxgrade', 'mrs']
mrs_data.fillna(value='none', inplace=True)
mrs_data['startdate'] = pd.to_datetime(mrs_data['startdate'], format='%d-%m-%Y')
mrs_data['year'] = mrs_data['startdate'].dt.year
mrs_data.drop(columns=['enddate', 'startdate'], inplace=True)
mrs_data = mrs_data[mrs_data['year'].isin([2016, 2017, 2018])]
mrs_data.drop_duplicates(subset=['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 
                 'employee_band', 'pay_scale_level', 'year'], inplace=True)
mrs_data.reset_index(drop=True, inplace=True)

mrs_full = mrsdf2.merge(mrs_data.reset_index(drop=True), how='left', 
                          on=['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 
                 'employee_band', 'pay_scale_level', 'year'])

mrs_full.fillna(value=0, inplace=True)
mrs_full['mingrade'] = np.where(mrs_full['mingrade']<1, mrs_full['mingrade_new'], mrs_full['mingrade'])
mrs_full['maxgrade'] = np.where(mrs_full['maxgrade']<1, mrs_full['maxgrade_new'], mrs_full['maxgrade'])
mrs_full['mrs'] = np.where(mrs_full['mrs']<1, (mrs_full['mingrade']+mrs_full['maxgrade'])/2, mrs_full['mrs'])

mrs_full = mrs_full[['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 
                 'employee_band', 'pay_scale_level', 'year', 'mrs']]
mrs_full.reset_index(drop=True, inplace=True)

bp_full = bp_full.merge(mrs_full, how='left', on=['country_grouping_p_', 'pay_scale_type', 'pay_scale_area_code', 
                 'employee_band', 'pay_scale_level', 'year'])
bp_full['mrs'] = np.where(bp_full['mrs']<1, 1., bp_full['mrs'])
bp_full['compare_ratio'] = bp_full['annual_salary'].astype(float)/bp_full['mrs'].astype(float)
bp_full['compare_ratio'] = np.where(bp_full['compare_ratio']>100, 100., bp_full['compare_ratio'])



### SALARY ###
# need to confirm if salary has already been normalized. seems so for some currency types. still implementing the module to handle it
bp_full = helpers.process_columns(bp_full.copy(), cols=['currency_key'])
bp_full = salary_process(bp_full.copy(), 'annual_salary', curr_dict=curr_dict)



### TEAMSIZE and TEAMSIZE_DIFF ###
def team_func(df):
    def teamsize_func(df):
        temp = df[['direct_manager_emp_id', 'global_id', 'year']]
        temp = cust_funcs.force_numeric(temp)
        temp.drop_duplicates(inplace=True)
        temp.dropna(inplace=True)
        return temp
    df = teamsize_func(df)
    
    dfgrp = df.groupby(['direct_manager_emp_id', 'year']).size().to_frame(name='teamsize').reset_index()
    dfgrp.sort_values(by=['direct_manager_emp_id', 'year'], ascending=True, inplace=True)
    dfgrp.reset_index(drop=True, inplace=True)
    dfgrp['teamsize_delta'] = dfgrp.groupby(['direct_manager_emp_id'])['teamsize'].diff().fillna(0)
    return dfgrp[['direct_manager_emp_id', 'year', 'teamsize', 'teamsize_delta']]

teamsize_and_diff_df = team_func(df=bp_full)
