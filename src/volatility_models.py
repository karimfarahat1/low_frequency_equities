from arch import arch_model

class mean_vol:
    def __init__(self, sq_residuals, halflife = None):
        """
        Forecasts volatility via taking a (weighted) average of past observed residuals
        
        Arguments:
            sq_residuals: Pandas dataframe with multi-index of (TimeStamp, AssetID) containing squared residuals
            halflife: Int representing the halflife parameter if an EWMA is desired
        """
        self.data = sq_residuals
        self.halflife = halflife
        self.pred_date = sq_residuals.index.get_level_values(1).unique.sort_values()[-1]
        
    def fit(self):
        if self.halflife:
            self.predict = self.data.groupby('AssetID').ewm(halflife = self.halflife).mean().xs(self.date, level = 1) 
        else:
            self.predict = self.data.groupby('AssetID').mean().xs(self.date, level = 0)
    
    def predict(self):
        return self.predict

class arch_wrapper:
    def __init__(self, sq_residuals, model_params = None):
        """
        Acts as a wrapper to the ARCH library available from pip, putting it in an appropriate form for our library
        
        Arguments:
            sq_residuals:
            model_params:
        """
        self.model_params = model_params
        self.data = sq_residuals
    
    def fit(self):
        if self.model_params == None:    
            self.model = arch_model(self.data).fit()
        else:
            self.model = arch_model(self.data, **self.model_params).fit()

    def predict(self):
        return self.model.forecast(reindex=False).variance.values.flatten()[0]
