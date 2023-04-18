
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
        
        strat_returns_dict = {}
        
        dates = features.index.get_level_values(0).unique()
        init_features = features.loc[pd.IndexSlice[:dates[burn_in-1],:],:]
        init_returns = returns.loc[pd.IndexSlice[:dates[burn_in-1],:]]
                     
        for strat in self.strategies:

            strat_returns = []
            strat_obj = self.strategies[strat]
            strat_obj.process_data(init_features, init_returns)
            
            for key, df in features.loc[pd.IndexSlice[dates[burn_in]:,:],:].groupby('TimeStamp'):
                
                rtns = returns.xs(key, drop_level = False)
                portfolio_weights = strat_obj.compute_weights(df)
                portfolio_returns = (portfolio_weights * rtns).sum()
                strat_returns.append(portfolio_returns)
                strat_obj.process_data(df, rtns)
                
            strat_returns_dict[strat] = pd.Series(strat_returns, index = dates)
        
        self.statistics['returns'] = strat_returns_dict
    
    def compute_stat(self, stat_calc, stat_name):
        """
        Computes a given statistic for each strategy stores the results in the self.statistics attribute
        
        Arguments:
            stat_calc: A function to compute a performance summary statistic given the return series
            stat_name: Key used to store the statistic in self.statistics 
        """     
        self.statistics[stat_name] = {}
        
        for strat, returns in self.statistics['returns'].items():
            self.statistics[stat_name][strat] = stat_calc(returns)
            
    def plot_stat(self, stat_name):
        """
        Creates a summary plot for a statistic
        
        Arguments:
            stat_name: key stored in self.statistics
        """        
        stat = self.statistics[stat_name]        
        plt.bar(stat.keys(), stat.values())
        plt.xlabel(stat_name)
                     
    def sharpe_ratio(self, returns):
        """
        Observed returns over volatility
        """
        return np.mean(returns) / np.std(returns)
    
    def sortino_ratio(self, returns):
        """
        Observed mean returns over downside volatility
        """
        return_downside = np.minimum(0, returns)**2
        expected_downside = np.mean(return_downside)
        downside_risk = np.sqrt(expected_downside)    
        return np.mean(returns) / downside_risk

    def max_drawdown(self, returns):
        """
        Largest observed loss
        """
        return np.min(returns)
    