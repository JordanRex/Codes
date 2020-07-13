import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing imort StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf


class sklearn_elasticnet:
    
    def __init__(self, df, grpvar, depvar, disc_cols, other_covar):
        segment_groups = df.groupby([grpvar])

        params = {}
        metrics = {}

        discounts_formula = '+'.join(disc_cols)

        other_covariates = other_covar
        other_covariates = ['C('+x+')' for x in other_covariates]
        covariates_formula = '+'.join(other_covariates)

        for x in segment_groups.groups:
            data = segment_groups.get_group(x)
            data = sm.add_constant(data)
            
            y, X = dmatrices(f'{depvar} ~' + discounts_formula + '+' + covariates_formula + '+Qty', data=data, return_type="dataframe")
            scaler = StandardScaler()
            scaler.fit(X)
            X = pd.DataFrame(data=scaler.transform(X), columns=X.columns)
            
            model = ElasticNet(fit_intercept=False).fit(X, y)
            
            params[x] = dict(zip(X.columns, model.coef_))
            metrics[x] = {'R2':model.score(X, y),
                          'Number of Customers':int(data['SoldToNumber'].nunique()),
                            'Mean TheoreticalMax':data['Theoretical_Max__Final_Adj_'].mean(),
                            'Mean RealizedMargin':data['RealizedMargin'].mean()}
            for discount in disc_cols:
                metrics[x][discount] = np.mean(data[discount]==0)*100

        ###### OVERALL #######
        x = 'overall'
        data = df
        data = sm.add_constant(data)
        
        y, X = dmatrices(f'{depvar} ~' + discounts_formula + '+' + covariates_formula + '+Qty', data=data, return_type="dataframe")
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(data=scaler.transform(X), columns=X.columns)
        
        model = ElasticNet(fit_intercept=False).fit(X, y)
        self.model = model
        self.feature_names = X.columns

        params[x] = dict(zip(X.columns, model.coef_))
        metrics[x] = {'R2':model.score(X, y),
                      'Number of Customers':int(data['SoldToNumber'].nunique()),
                        'Mean TheoreticalMax':data['Theoretical_Max__Final_Adj_'].mean(),
                        'Mean RealizedMargin':data['RealizedMargin'].mean()}
        for discount in disc_cols:
            metrics[x][discount] = np.mean(data[discount]==0)*100
            
        self.params = params
        self.metrics = metrics
        
        
class ols_model:
    
    def __init__(self, df, grpvar, depvar, disc_cols, other_covar, model_name):
        segment_groups = df.groupby([grpvar])

        params = {}
        metrics = {}

        discounts_formula = '+'.join(disc_cols)

        other_covariates = other_covar
        other_covariates = ['C('+x+')' for x in other_covariates]
        covariates_formula = '+'.join(other_covariates)

        for x in segment_groups.groups:
            data = segment_groups.get_group(x)
            data = sm.add_constant(data)
            ## need to incorporate standard scaler here
            model = smf.ols(f'{depvar} ~' + discounts_formula + '+' + covariates_formula + '+Qty', data = data).fit()
            params[x] = dict(zip(model.params.index,model.params.values))
            metrics[x] = {'R2':model.rsquared,
                              'Number of Customers':int(data['SoldToNumber'].nunique()),
                              'Number of Products':int(data['Material'].nunique()),
                                'Mean TheoreticalMax':data['Theoretical_Max__Final_Adj_'].mean(),
                                'Mean RealizedMargin':data['RealizedMargin'].mean(),
                                'Mean RealizedMarginPerc':data['RealizedMarginPerc'].mean(),
                             'F_pval':model.f_pvalue,
                             'AIC':model.aic,
                             'BIC':model.bic}
            for discount in disc_cols:
                metrics[x][discount] = np.mean(data[discount]==0)*100
            model.save(f'../../../data/{model_name}_{x}.pkl', remove_data=True)

        x = 'overall'
        data = df
        data = sm.add_constant(data)
        model = smf.ols(f'{depvar} ~' + discounts_formula + '+' + covariates_formula + '+Qty', data = data).fit()
        params[x] = dict(zip(model.params.index,model.params.values))
        metrics[x] = {'R2':model.rsquared,
                      'Number of Customers':int(data['SoldToNumber'].nunique()),
                      'Number of Products':int(data['Material'].nunique()),
                        'Mean TheoreticalMax':data['Theoretical_Max__Final_Adj_'].mean(),
                        'Mean RealizedMargin':data['RealizedMargin'].mean(),
                        'Mean RealizedMarginPerc':data['RealizedMarginPerc'].mean(),
                     'F_pval':model.f_pvalue,
                     'AIC':model.aic,
                     'BIC':model.bic}
        for discount in disc_cols:
            metrics[x][discount] = np.mean(data[discount]==0)*100
        model.save(f'../../../data/{model_name}.pkl', remove_data=True)
        self.params = params
        self.metrics = metrics