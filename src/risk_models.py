
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class risk_models:
    def __init__(self, factor_risk_model = True, cov_calculator = None, cov_lookback = None):
        """
        Initialises storage of risk model data
        
        Arguments:
            factor_risk_model: Logical indicating intention to use factor risk model
            cov_calculator: A function which given a pandas series of asset returns produces a covariance matrix
            cov_lookback: Int number of historical periods to use in covariance calculation
        """
        if (not factor_risk_model) and (not cov_calculator or not cov_lookback):
            raise Exception("Need to use a factor model or specify a direct covariance estimation process")
            
        self.use_factor_model = factor_risk_model
        
        if factor_risk_model:    
            self.factor_returns = {}
            self.cov_lookback = None
        else:
            self.cov_lookback = cov_lookback
            self.cov_calc = cov_calculator
            
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
                return weighted_cov(return_df.values, weights)
                
    def vol_fit_predict(self, model, universe, model_params = None, min_obs = 12):
        """
        Fits a model to the time series of each asset's residuals and forecasts next periods volatility
        
        Arguments:
            model: Object with .fit() and .predict() methods
            residuals: Pandas series with index (TimeStamp, AssetID) containing the de-meaned asset returns
            min_obs: Int specifing the minimum number of residuals to observe per asset before fitting a volatility model
        """
        trailing_vol = self.return_residuals.loc[pd.IndexSlice[:, self.return_residuals.index.isin(universe, level = 1)],:]
        trailing_vol = trailing_vol ** 2
                
        vol_forecasts = pd.Series(index = universe)
        
        #fitting model to avg of squared residuals over time for those series with insufficient data
        avg_sq_residuals = (self.return_residuals **2).groupby('TimeStamp').mean()
        copied_model = model(avg_sq_residuals, model_params)
        copied_model.fit()
        avg_fcast = copied_model.predict()
        
        for key, df in trailing_vol.groupby('AssetID'):
            
            if len(df) > min_obs:
                copied_model = model(df, model_params)
                copied_model.fit()                
                vol_forecasts[key] = copied_model.predict()
            
            else:
                vol_forecasts[key] = avg_fcast
        
        vol_forecasts.fillna(avg_fcast, inplace = True)
        
        return vol_forecasts

    def risk_model(self, returns = None, vol_model = None, vol_model_params = None, vol_min_obs = 12, features = None, factor_cov_params = None):
        """
        Computes stock covariance either directly with an estimator such as Ledoit-Wolf or via a factor model
        
        Arguments:
            returns: Pandas series with multi-index of (TimeStamp, AssetID)
            vol_model: Object with .fit() and .predict() methods
            vol_model_params: Dict containing kwargs for vol_model
            vol_min_obs: Int specifing the minimum number of residuals to observe per asset before fitting a volatility model
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing factor scores
            factor_cov_params: Dict containing kwargs for factor_cov function
        """
        
        if type(returns) == type(None) and not self.use_factor_model:
            raise Exception("Need to specify either returns for direct covariance estimation or features for a factor model")
        
        elif type(returns) != type(None) and self.use_factor_model:
            raise Exception("Overspecified - both factor models and direct estimation are specified")
        
        elif type(returns) != type(None) and not self.use_factor_model:
            #direct estimation of covariance
            stock_cov = self.cov_calc(returns)
            
            #scales covariance by future forecasts of volatility
            if vol_model:
                univ = returns.index.get_level_values(1).unique()
                vol_forecasts = self.vol_fit_predict(model = vol_model, universe = univ, model_params = vol_model_params, min_obs = vol_min_obs).values.flatten()
                current_vol = np.diag(stock_cov)
                stock_cov = stock_cov / np.outer(current_vol ** 0.5, current_vol ** 0.5)
                stock_cov = stock_cov * np.outer(vol_forecasts ** 0.5, vol_forecasts ** 0.5)

        elif self.use_factor_model and not vol_model:
            raise Exception("Need to specify an estimator of idiosyncratic volatility to use a factor model")
                                
        else:
            #factor model covariance estimation
            univ = features.index.get_level_values(1).unique()
            idiosyncratic_vol = np.diag(self.vol_fit_predict(model = vol_model, universe = univ, model_params = vol_model_params, min_obs = vol_min_obs).values.flatten())
            
            if factor_cov_params == None:
                stock_cov = self.factor_cov()
            else:
                stock_cov = self.factor_cov(**factor_cov_params)
                
            stock_cov = np.matmul(np.matmul(features.values, stock_cov), features.transpose().values)
            stock_cov += idiosyncratic_vol
            
        return stock_cov


def pca_cov(returns, n_comps = 0.8):
    """
    PCA based covariance estimation
    
    Arguments:
        returns: Pandas series with multi-index of (TimeStamp, AssetID) containing asset returns
        n_comps: Either an int indicating the number of components or a float target fraction of variance explained
    """
    returns = returns.unstack().dropna(axis = 1)
    pca_obj = PCA(n_components = n_comps).fit(returns)
    cov_est = pca_obj.get_covariance()- pca_obj.noise_variance_ * np.eye(returns.shape[1])
    
    cov_est = cov_est / np.outer(np.diag(cov_est) ** 0.5, np.diag(cov_est) ** 0.5)
    cov_est = cov_est * np.outer(np.diag(returns.cov()) ** 0.5, np.diag(returns.cov()) ** 0.5)
    
    return cov_est    

def weighted_cov(returns, weights):
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