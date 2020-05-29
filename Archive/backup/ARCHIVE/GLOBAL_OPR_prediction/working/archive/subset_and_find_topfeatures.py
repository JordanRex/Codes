## to subset features created from a concept and check for top n important features

train_count_dummy = train_dummy.filter(regex = 'rankindex')
valid_count_dummy = valid_dummy.filter(regex = 'rankindex')

weights_dict = {0:1.2, 1:1.1, 2:1, 3:1, 4:1.1, 5:1.5}
ytrain_weights = np.copy(ytrain)
yvalid_weights = np.copy(yvalid)
for k, v in weights_dict.items(): 
    ytrain_weights[ytrain==k]=v
    yvalid_weights[yvalid==k]=v
np.array(np.unique(ytrain_weights, return_counts=True)).T

xg_train = xgb.DMatrix(train_count_dummy, label=ytrain, weight=ytrain_weights)
xg_test = xgb.DMatrix(valid_count_dummy, label=yvalid, weight=yvalid_weights)

# setup parameters for xgboost
param = {'objective':'multi:softmax', 'max_depth':4, 'silent':1, 'nthread':-1, 'num_class':6, 'subsample':0.8, 
         'colsample_bytree':0.8, 'learning_rate':0.1, 'eval_metric':['merror', 'mlogloss'], 'max_bin':300,
        'colsample_bylevel':0.7, 'tree_method':'hist', 'seed':1, 'grow_policy':'lossguide', }

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 100
model = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=40, verbose_eval=10)
# get prediction
pred = model.predict(xg_test)
error_rate = np.sum(pred != yvalid) / yvalid.shape[0]
print('Test error using softmax = {}'.format(error_rate))

print(skm.accuracy_score(y_pred=pred, y_true=yvalid))
print(skm.confusion_matrix(y_pred=pred, y_true=yvalid))

def get_xgb_imp(xgb):
    imp_vals = xgb.get_fscore()
    feats_imp = pd.DataFrame(imp_vals,index=np.arange(2)).T
    feats_imp.iloc[:,0]= feats_imp.index    
    feats_imp.columns=['feature','importance']
    feats_imp.sort_values('importance',inplace=True,ascending=False)
    feats_imp.reset_index(drop=True,inplace=True)
    return feats_imp

feature_importance_df1 = get_xgb_imp(model)
#feature_importance_df1.to_csv('dummy.csv', index=False)

print(feature_importance_df1.feature[0:10].values)
