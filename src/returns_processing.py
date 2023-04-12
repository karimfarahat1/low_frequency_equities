
from sklearn.linear_model import LinearRegression
import numpy as np

def return_to_alpha(stock_returns, index_returns):
    """
    Computes a stock's CAPM alpha.
    
    Arguments:
        stock_returns: Pandas series with multi-index of (TimeStamp, AssetID)
        index_returns: Pandas series with index of TimeStamp
    """
    
    index_returns = index_returns.loc[stock_returns.index.get_level_values(0)]
    
    CAPM = LinearRegression().fit(index_returns.values.reshape(-1,1), stock_returns.values.reshape(-1,1))
    
    return CAPM.intercept_[0]

def return_to_vol(stock_returns, index_returns):
    """
    Computes a stock's CAPM implied idiosyncratic volatility.
    
    Arguments:
        stock_returns: Pandas series with multi-index of (TimeStamp, AssetID)
        index_returns: Pandas series with index of TimeStamp
    """
    
    index_returns = index_returns.loc[stock_returns.index.get_level_values(0)]
    
    CAPM = LinearRegression().fit(index_returns.values.reshape(-1,1), stock_returns.values.reshape(-1,1))
    
    resid = (index_returns.values - CAPM.predict(index_returns.values.reshape(-1,1)).flatten())
    
    return np.sum(resid ** 2)

