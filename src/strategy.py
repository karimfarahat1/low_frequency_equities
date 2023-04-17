
from src.expected_returns import expected_returns
from src.risk_models import risk_models
import pandas as pd

class strategy:
    def __init__(self, return_model, vol_model, optim_func, lookback_window = 1, vol_model_params = None):
        """
        Collates the key aspects of the strategy: models for return/risk and an optimiser
        
        Arguments:
            return_model: Object with .fit() and .predict() methods
            vol_model: Object with .fit() and .predict() methods
            optim_func: Function taking arguments of the expected return vector and covariance matrix and outputs portfolio weights
            lookback_window: Int indicating lookback window used to train expected return models
            vol_model_params: Dict containing kwargs for vol_model
        """
        self.return_model = return_model
        self.vol_model = vol_model
        self.vol_model_params = vol_model_params
        self.optim_func = optim_func
        self.expected_returns = expected_returns(lookback_window)
        self.risk_model = risk_models()
            
    def process_data(self, features, returns):
        """
        Updates strategy with information as it becomes available
        
        Arguments:
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing factor scores
            returns: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing returns
        """
        self.expected_returns.train(self.return_model, features, returns)
        
        new_data = features.index.get_level_values(0).unique()
        sub_models = {key : self.expected_returns.models[key] for key in new_data}
        
        self.risk_model.store_factor_returns(sub_models)
        self.risk_model.store_residuals(sub_models, features, returns)
        
    def compute_weights(self, features):
        """
        Computes portfolio weights based on values of current signals and historical information
        
        Arguments:
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing factor scores
        """
        forecast_returns = self.expected_returns.forecast(features)
        
        if self.vol_model_params:
            forecast_covariance = self.risk_model.risk_model(vol_model = self.vol_model, features = features, vol_model_params = self.vol_model_params)
        else:
            forecast_covariance = self.risk_model.risk_model(vol_model = self.vol_model, features = features)
        
        weights = self.optim_func(expected_returns = forecast_returns, expected_covariance = forecast_covariance)
        
        weights = pd.Series(weights.values(), index = features.index.get_level_values(1))
        
        return weights
    