
import pandas as pd
import numpy as np
from copy import deepcopy

class risk_models:
    def __init__(self):
        self.factor_returns = {}
        self.return_residuals = pd.DataFrame()
            
    def store_factor_returns(self, models):
        """
        Arguments:
            model: Dictionary with timestamps as keys and values an object with .coef_ attribute 
                or object indexable by 'model' with .coef_ attribute (e.g. sklearn pipeline object)
        """
        for date in models:
            model = models[date]
            
            try:
                coefs = model.coef_.flatten()
            except:
                coefs = model['model'].coef_.flatten()
                
            self.factor_returns[date] = coefs
    
    def store_residuals(self, models, features, returns):
        """        
        Arguments:
            model: Object with .predict() method
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing model predictors
            returns: Pandas series with multi-index of (TimeStamp, AssetID) containing asset returns
        """
        residuals = pd.DataFrame([np.nan] * len(returns.index), index = returns.index, columns = ['resid'])

        for key, feats in features.groupby('TimeStamp'):
            model = models[key]
            rtns = returns.loc[pd.IndexSlice[key,:]]
            residuals.loc[pd.IndexSlice[key, :]] = rtns.values - model.predict(feats).flatten()
    
        self.return_residuals = pd.concat([self.return_residuals, residuals])
    
    def weighted_cov(self, returns, weights):
        """
        Computes covariance using a weighted average of the deviations.
        
        Arguments:
            returns: Array containing return series
            weights: Array containing weights
        """
        n = len(returns)
        cov = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                
                dev_x = (returns[:, i] - returns[:, i].mean())
                dev_y = (returns[:, i] - returns[:, i].mean())
                cov[i, j] = cov[j, i] =  (dev_x * dev_y * weights).sum()
        
        return cov
    
    def factor_cov(self, look_back = 'all', weights = None, semi_cov = False):
        """
        Computes the factor covariance matrix from stored factor returns.
        
        Arguments:
            look_back: Integer representing number of historical periods to include
            weights: Array containing weight associated with each time period
            semi_cov: Logical indicating whether to consider downside deviations only
        """
        
        return_df = pd.DataFrame.from_dict(self.factor_returns, orient = 'index')
        
        if look_back != 'all':
            return_df = return_df.iloc[-look_back:,:]
        
        if semi_cov:
            return_df = return_df.applymap(lambda x: max(x, 0))
            
        else:
            if not weights:
                return return_df.cov().values
            
            if weights:
                return self.weighted_cov(return_df.values, weights)
            
    def vol_fit_predict(self, model, universe, model_params = None, min_obs = 1):
        """
        Fits a model to the time series of each stock's residuals and forecasts next periods volatility.
        
        Arguments:
            model: Object with .fit() and .predict() methods
            residuals: Pandas series with index (TimeStamp, AssetID) containing the de-meaned asset returns
            min_obs: Int specifing the minimum number of residuals to observe per asset before fitting a volatility model
        """
        
        trailing_vol = self.return_residuals.loc[pd.IndexSlice[:, self.return_residuals.index.isin(universe, level = 1)],:]
        trailing_vol = trailing_vol ** 2
                
        vol_forecasts = pd.Series(index = universe)
        
        #fitting model to avg of residuals over time for those series with insufficient data
        avg_residuals = self.return_residuals.groupby('TimeStamp').mean()
        copied_model = deepcopy(model)
        
        if model_params == None:    
            copied_model = copied_model(avg_residuals, **model_params).fit()
        else:
            copied_model = copied_model(avg_residuals).fit()
        
        avg_fcast = copied_model.forecast(reindex=False).variance.values.flatten()[0]
        
        for key, df in trailing_vol.groupby('AssetID'):
            
            if len(df) > min_obs:
                copied_model = deepcopy(model)
                
                if model_params == None:
                    vol_model = copied_model(df).fit()
                else:
                    vol_model = copied_model(df, **model_params).fit()
                
                vol_forecasts[key] = vol_model.forecast(reindex=False).variance.values.flatten()[0]
            
            else:
                vol_forecasts[key] = avg_fcast
        
        vol_forecasts.fillna(avg_fcast, inplace = True)
        
        return vol_forecasts

    def risk_model(self, vol_model, features, vol_model_params = None,  factor_cov_params = None, min_obs = None):
        """
        Computes stock covariance from an estimate of factor covariance and forecasts of idiosyncratic volatility.
        
        Arguments:
            vol_model: Object with .fit() and .predict() methods
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing factor scores
            vol_model_params: Dict containing kwargs for vol_model
            factor_cov_params: Dict containing kwargs for factor_cov function
            min_obs: Int specifing the minimum number of residuals to observe per asset before fitting a volatility model
        """
        
        univ = features.index.get_level_values(1).unique()
        idiosyncratic_vol = np.diag(self.vol_fit_predict(model = vol_model, universe = univ, model_params = vol_model_params).values.flatten())
        
        if factor_cov_params == None:
            stock_cov = self.factor_cov()
        else:
            stock_cov = self.factor_cov(**factor_cov_params)
            
        stock_cov = np.matmul(np.matmul(features.values, stock_cov), features.transpose().values)
        stock_cov += idiosyncratic_vol
        
        return stock_cov
    
    