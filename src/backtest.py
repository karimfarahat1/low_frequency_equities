
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class backtest:
    def __init__(self, strategies):
        """
        Initialises containers for strategies and summary statistics
        
        Arguments:
            strategies: Dict with keys names of strategies and values strategy objects
        """
        self.strategies = strategies
        self.statistics = {}
        
    def compute_returns(self, features, returns, burn_in):
        """
        A function to walk through the data and compute the returns for each strategy. Assumes features are lagged by 1 relative to returns. 
         
        Arguments: 
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID)
            returns: Pandas series with multi-index of (TimeStamp, AssetID)
            burn_in: Int denoting the number of time periods used to initialise each strategy
        """
        strat_returns_dict = {'Equal Weight' : returns.groupby('TimeStamp').mean()}
                
        dates = features.index.get_level_values(0).unique()
        init_features = features.loc[pd.IndexSlice[:dates[burn_in-1],:],:]
        
        for strat in self.strategies:

            strat_obj = self.strategies[strat]            
            
            #trains models on transformed returns if desired by strategy
            if strat_obj.return_transform:
                transformed_returns = strat_obj.return_transform(returns)
            else:
                transformed_returns = returns
            
            init_returns = transformed_returns.loc[pd.IndexSlice[:dates[burn_in-1],:]]
            strat_returns = []
            strat_obj.process_data(init_features, init_returns)
            
            for key, feats in features.loc[pd.IndexSlice[dates[burn_in]:,:],:].groupby('TimeStamp'):
                
                #enforces minimum lookback
                if strat_obj.risk_model.cov_lookback:
                    rtns = returns.loc[pd.IndexSlice[:key,:]]
                    feats, rtns = self.min_lookback(feats, rtns, strat_obj.risk_model.cov_lookback)
                    trans_rtns = transformed_returns.loc[pd.IndexSlice[:key,:]]
                    _, trans_rtns = self.min_lookback(feats, trans_rtns, strat_obj.risk_model.cov_lookback)
                else:
                    trans_rtns = transformed_returns.loc[pd.IndexSlice[:key,:]]

                #computes weights, returns, and updates strategy with new info
                rtns = returns.xs(key, drop_level = False)
                portfolio_weights = strat_obj.compute_weights(feats, trans_rtns)
                portfolio_returns = (portfolio_weights * rtns).sum()
                strat_returns.append(portfolio_returns)
                strat_obj.process_data(feats, trans_rtns.xs(key, drop_level = False))
                
            strat_returns_dict[strat] = pd.Series(strat_returns, index = dates[burn_in:])
        
        self.statistics['returns'] = strat_returns_dict
    
    def min_lookback(self, features, returns, min_obs):
        """
        Reshapes the investment universe as dictated by the minimum number of observations required in the risk model
        
        Arguments: 
            features: Pandas dataframe with multi-index of (TimeStamp, AssetID)
            returns: Pandas series with multi-index of (TimeStamp, AssetID)
            min_obs: Int denoting the minimum number of obser
        """
        returns = returns.unstack().iloc[-min_obs:].dropna(axis=1)
        univ = returns.columns
        returns = returns.stack()
        features = features.loc[pd.IndexSlice[:, univ],:]
        
        return features, returns
    
    def compute_stat(self, stat_calcs, stat_names):
        """
        Computes a given statistic for each strategy stores the results in the self.statistics attribute
        
        Arguments:
            stat_calc: A list of functions to compute a performance summary statistic given the return series
            stat_name: A list of keys used to store the statistic in self.statistics 
        """     
        for calc, name in zip(stat_calcs, stat_names):
            self.statistics[name] = {}
            
            for strat, returns in self.statistics['returns'].items():
                self.statistics[name][strat] = calc(returns)
            
    def plot_stat(self, stat_name):
        """
        Creates a summary plot for a statistic
        
        Arguments:
            stat_name: key stored in self.statistics
        """        
        stat = self.statistics[stat_name]        
        plt.bar(stat.keys(), stat.values())
        plt.ylabel(stat_name)
                     
    def sharpe_ratio(self, returns):
        """
        Observed returns over volatility annualised
        """
        return (np.mean(returns) / np.std(returns)) * np.sqrt(12)
    
    def sortino_ratio(self, returns):
        """
        Observed mean returns over downside volatility annualised
        """
        return_downside = np.minimum(0, returns)**2
        expected_downside = np.mean(return_downside)
        downside_risk = np.sqrt(expected_downside)    
        return (np.mean(returns) / downside_risk) * np.sqrt(12)

    def max_drawdown(self, returns):
        """
        Largest observed loss annualised
        """
        return np.min(returns) * 12
