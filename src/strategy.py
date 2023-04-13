
from src.expected_returns import expected_returns
from src.risk_models import risk_models

class strategy:
    def __init__(self, return_model, vol_model, optim_func, features, returns, vol_model_params = None):
        """
        Collates the key aspects of the strategy: models for return/risk and an optimiser
        
        Arguments:
            return_model: Object with .fit() and .predict() methods
            vol_model: Object with .fit() and .predict() methods
            optim_func: Function taking arguments of the expected return vector and covariance matrix and outputs portfolio weights
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing factor scores
            returns: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing returns
            vol_model_params: Dict containing kwargs for vol_model
        """
        self.return_model = return_model
        self.vol_model = vol_model
        self.vol_model_params = vol_model_params
        self.optim_func = optim_func
        self.expected_returns = expected_returns()
        self.expected_returns.train(return_model, features, returns)
        self.risk_model = risk_models(factor_risk_model = True, models = self.expected_returns.models, features = features, returns = returns)
            
    def process_data(self, features, returns):
        """
        Updates strategy with information as it becomes available
        
        Arguments:
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing factor scores
            returns: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing returns
        """
        self.expected_returns.train(self.return_model, features, returns)
        date = next(reversed(self.expected_returns.models))
        model = self.expected_returns.models[date]
        self.risk_model.store_factor_returns(model, date)
        self.risk_model.store_residuals(model, features, returns)
        
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
        
        return weights
    