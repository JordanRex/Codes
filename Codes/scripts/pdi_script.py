## pdi script

### PDI components
pdi_cols = ['global_id', 'pdi_score', 'pdi_score_category']
pdi_train = helpers.csv_read(file_path='../input/PDI/pdi_2017.csv', cols_to_keep=pdi_cols)
pdi_train['year'] = 2017
pdi_valid = helpers.csv_read(file_path='../input/PDI/pdi_2018.csv', cols_to_keep=pdi_cols)
pdi_valid['year'] = 2018
