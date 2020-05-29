## main ads python script

import sys, pandas as pd, numpy as np, inspect, re as re, functools as functools, pickle, glob, warnings, os
from tqdm import tqdm
# sklearn packages
from sklearn.feature_selection import VarianceThreshold
import sklearn.metrics as skm
# some options/variables
randomseed = 1 # the value for the random state used at various points in the pipeline
# append the scripts path to pythonpath
sys.path.append('./scripts/')
# ignore warnings (only if you are the kind that would code when the world is burning)
warnings.filterwarnings('ignore')
# # import the various ml modules
import xgboost as xgb

############################################## import the custom modules ################################
import helperfuncs as helper
import feateng as fte
import misc as miscfun
import oprfunctions as oprfun
from misc import ce_encodings, DataFrameImputer, scalers
from oprfunctions import demo_fn, salary_process

# instantiate the classes
helpers = helper.helper_funcs()
cust_funcs = fte.custom_funcs()
feat_sel = miscfun.feat_selection()

#############################################################################################################
# global function to flatten columns after a grouped operation and aggregation
# outside all classes since it is added as an attribute to pandas DataFrames
def __my_flatten_cols(self, how="_".join, reset_index=True):
    how = (lambda iter: list(iter)[-1]) if how == "last" else how
    self.columns = [how(filter(None, map(str, levels))) for levels in self.columns.values] \
    if isinstance(self.columns, pd.MultiIndex) else self.columns
    return self.reset_index(drop=True) if reset_index else self
pd.DataFrame.my_flatten_cols = __my_flatten_cols

#############################################################################################################

exec(open("./scripts/dicts_cols.py").read())

#############################################################################################################

from azure.datalake.store import core, lib, multithread
tenant = 'cef04b19-7776-4a94-b89b-375c77a8f936'
resource = 'https://datalake.azure.net/'
client_id = 'e9aaf06a-9856-42a8-ab3c-c8b0d3a9b110'
client_secret = 'DlbuV60szYT2U0CQNjzwRA55EsH42oX92AB7vbD2clk='
adlcreds = lib.auth(tenant_id = tenant,
                   client_secret = client_secret,
                   client_id = client_id,
                   resource = resource)
subs_id = '73f88e6b-3a35-4612-b550-555157e7059f'
adls = 'edhadlsanasagbdev'
adlsfsc = core.AzureDLFileSystem(adlcreds, store_name=adls)
path = '/root/anasandbox/people/opr10x/'

#############################################################################################################
## OPR SCRIPT ##

# exec(open("./scripts/opr_script.py").read())
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Miscellaneous/opr_backup_17to18.pickle', 'rb') as f:
#     opr_reshaped = pickle.load(f)
#     f.close()
    
#############################################################################################################
## BLUEPRINT SCRIPT ##

# exec(open("./scripts/blueprint_script.py").read())
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Blueprint/bp_backup_16to19_processed.pickle', 'rb') as f:
#     bp_full = pickle.load(f)
#     f.close()
    
#############################################################################################################
## MISCELLANEOUS SCRIPT ##

# exec(open("./scripts/misc_script.py").read())

#############################################################################################################
## COMPETENCY SCRIPT ##

# exec(open("./scripts/competency_script_new.py").read())
# compfull = open('E:/ADLS/pickles/competency_16to18_raw.pickle', 'rb')
# comp_full = pickle.load(compfull)
# compfull.close()

# exec(open("./scripts/comp_2019.py").read())
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/competency_full_2019.pickle', 'rb') as f:
#     comp2019 = pickle.load(f)
#     f.close()

#############################################################################################################
## TARGET SCRIPT ##

# exec(open("./scripts/target_script.py").read())
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Miscellaneous/target_backup.pickle', 'rb') as f:
#     tar_reshaped = pickle.load(f)
#     f.close()

#############################################################################################################
## MOVEMENTS SCRIPT ##

# exec(open("./scripts/movements_script.py").read())
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Movements/career_velocity.pkl', 'rb') as f:
#     cv_full = pickle.load(f)
#     pv_full = pickle.load(f)
#     tib_full = pickle.load(f)
#     f.close()

#############################################################################################################
## NAVIGATE SCRIPT ##

# exec(open("./scripts/navigate_script.py").read())
# with adlsfsc.open(path + '/2019/Data/Raw_Data/pickle_files/Navigate/navigate.pkl', 'rb') as f:
#     belts_grp = pickle.load(f)
#     tp_full = pickle.load(f)
#     engfull = pickle.load(f)
#     pdp_full = pickle.load(f)
#     f.close()

#############################################################################################################
## ORG CHART FEATURES ##

# exec(open("./scripts/org_chart_features.py").read())

#############################################################################################################
## ADS PREPARATION ##
# exec(open("./scripts/ads_prepare_new.py").read())
# with adlsfsc.open(path + '/2019/Data/Output_Data/ads/final_ads.pickle', 'wb') as f:
#     pickle.dump(ads, f)
#     f.close()
    
with adlsfsc.open(path + '/2019/Data/Output_Data/ads/final_ads.pickle', 'rb') as f:
    ads = pickle.load(f)
    f.close()

with adlsfsc.open(path + '/2019/Data/Output_Data/ads/final_ads.csv', 'wb') as f:
    ads_str = ads.to_csv()
    f.write(str.encode(ads_str))
    f.close()

print('ADS preparation completed')
print(ads.head(2))
