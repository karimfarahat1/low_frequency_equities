import numpy as np
import pandas as pd
from copy import deepcopy

class expected_returns:
    def __init__(self, lookback_window = 1):
        """
        Initialisation. It is assumed that models are fit to the panel, or have a consistent lookback window throughout.
        
        Arguments:
            lookback_window: either positive integer or the string 'all'. 
        """
        
        self.models = {}
        self.window = lookback_window
        
    def train(self, model, features, returns, burn_in = None):
        """
        Stores a model trained at each timestamp in the data provided. Assumes features are lagged by one timestamp relative to returns.
        
        Arguments:
            model: Object with .fit() and .predict() methods
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID)
            returns: Pandas series with multi-index of (TimeStamp, AssetID)
            burn_in: Relevant only when fitting models to the panel
        """
        
        timestamps = features.index.get_level_values(0).unique()
        
        if self.window == 'all':
            for time in timestamps[burn_in:]:
                cloned_model = deepcopy(model)
                self.models[time] = cloned_model.fit(features.loc[pd.IndexSlice[:time,:]], returns.loc[pd.IndexSlice[:time, :]])
        
        else:
            for idx, time in enumerate(timestamps[self.window-1:], self.window):
                training_window = timestamps[idx - self.window : idx]
                cloned_model = deepcopy(model)
                self.models[time] = cloned_model.fit(features.loc[pd.IndexSlice[training_window,:]], returns.loc[pd.IndexSlice[training_window, :]])
        
    def predict(self, features, aggregate = True, weights = None):
        """
        Supports prediction via panel regression or taking linear combinations of predictions from a series of models fit to the cross-section.
        
        Arguments:
            features: model inputs
            aggregate: to indicate whether models are fit cross-sectionally and hence require aggregation
            weights: weights used when averaging over each model output  
        """
        
        if not aggregate:
            current_model = next(reversed(self.models.keys()))
            
            return self.models[current_model].predict(features)
            
        else:
            forecast_returns = []
            
            for model_key in self.models: 
                model = self.models[model_key]
                forecast_returns.append(model.predict(features).flatten())
    
            if weights == None:
                return np.array(forecast_returns).mean(axis = 0)
            
            else:
                return (np.array(forecast_returns) * weights).mean(axis=0)
        
        
        