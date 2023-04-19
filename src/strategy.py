
from src.expected_returns import expected_returns
from src.risk_models import risk_models
import pandas as pd

class strategy:
    def __init__(self, return_model, optim_func, vol_model = None, ret_lookback = 1, vol_lookback = 12, vol_model_params = None, 
                       use_factor_risk = True, cov_calc = None, cov_lookback = None, return_transformation = None):
        """
        Collates the key aspects of the strategy: models for return/risk and an optimiser
        
        Arguments:
            return_model: Object with .fit() and .predict() methods
            optim_func: Function taking arguments of the expected return vector and covariance matrix and outputs portfolio weights
            vol_model: Object with .fit() and .predict() methods
            ret_lookback: Int indicating lookback window used to train expected return models
            vol_lookback: Int indicating minimum number of observations when training vol models
            vol_model_params: Dict containing kwargs for vol_model
            use_factor_risk: Logical indicating intention 
            cov_calculator: A function which given a pandas series of asset returns produces a covariance matrix
            cov_lookback: Int number of historical periods to use in covariance calculation 
            return_transformation: Function to transform the series of returns fed into the model if desired
        """
        self.return_model = return_model
        self.vol_model = vol_model
        self.vol_model_params = vol_model_params
        self.optim_func = optim_func
        self.expected_returns = expected_returns(ret_lookback)
        self.return_transform = return_transformation
        self.vol_lookback = vol_lookback
        self.risk_model = risk_models(factor_risk_model = use_factor_risk, cov_calculator = cov_calc, cov_lookback = cov_lookback)
            
    def process_data(self, features, returns):
        """
        Updates strategy with information as it becomes available
        
        Arguments:
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing factor scores
            returns: Pandas series with multi-index of (TimeStamp, AssetID) containing returns
        """
        self.expected_returns.train(self.return_model, features, returns)
        
        new_data = features.index.get_level_values(0).unique()
        sub_models = {key : self.expected_returns.models[key] for key in new_data}
        
        if self.risk_model.use_factor_model:
            self.risk_model.store_factor_returns(sub_models)
            
        self.risk_model.store_residuals(sub_models, features, returns)
        
    def compute_weights(self, features, returns):
        """
        Computes portfolio weights based on values of current signals and historical information
        
        Arguments:
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing factor scores
            returns: Pandas series with multi-index of (TimeStamp, AssetID) containing returns
        """
        forecast_returns = self.expected_returns.predict(features)
        
        if self.risk_model.use_factor_model:
            forecast_covariance = self.risk_model.risk_model(vol_model = self.vol_model, features = features, vol_model_params = self.vol_model_params, vol_min_obs = self.vol_lookback)

        else:
            forecast_covariance = self.risk_model.risk_model(returns = returns, vol_model = self.vol_model, vol_model_params = self.vol_model_params, vol_min_obs = self.vol_lookback)

        weights = self.optim_func(expected_returns = forecast_returns, expected_covariance = forecast_covariance)
        weights = pd.Series(weights.values(), index = features.index.get_level_values(1))
        
        return weights
    