### Low Frequency Equities

This repo exists to allow for rapid (and simplified) backtesting of typical low frequency cross-sectional trading strategies.

The idea is that the user need only specify a few core components of their strategy (e.g. a regression object and optimiser) to compute performance statistics.

For example, data loading/imports aside, a strategy which uses a linear expected return model, PCA to estimate covariance, and a mean-variance optimiser could be run with the simple snippet below.

```python
optim = lambda returns, covariance : EfficientFrontier(returns, covariance).efficient_risk(target_volatility = 0.1)
cov_est = lambda x: pca_cov(x, n_comps = 0.9)

factor_model = strategy(return_model = LinearRegression(), optim_func = optim, use_factor_risk = False, 
                            cov_calc = cov_est, cov_lookback = 24)

test = backtest({‘factor model’ : factor_model}
test.compute_returns(features, returns, 24)
```

Full working examples are available in the demo folder.