
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from src.expected_returns import expected_returns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

"""
Loading data
"""

data_dir = "../data/factors"

data = pd.read_csv(data_dir + "\\" + str(1) + ".csv")

for i in range(2, 11):            
    data = pd.concat([data, pd.read_csv(data_dir + "\\" + str(i) + ".csv")], axis = 0)

data = data.fillna(0)
data.set_index(['TimeStamp', 'StockID'], inplace = True)
data.sort_index(inplace = True)

"""
Fitting models
"""

returns = data.loc[:, 'Return']
features = data.iloc[:, 1:]

pipeline = Pipeline([('scaler', StandardScaler()), ('model', RidgeCV(alphas = np.linspace(0.001,100,10)))])

return_models = expected_returns()
return_models.train(pipeline, features, returns)

models = {}

for key, df in features.groupby('TimeStamp'):
    
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', RidgeCV(alphas = np.linspace(0.001,100,10)))])
    pipeline.fit(df, returns.loc[pd.IndexSlice[key,:]])
    models[key] = pipeline

"""
Checking coefficients are equal
"""

coefs_loop = []
coefs_class =[]

for key in models:
    coefs_loop.append(models[key]['model'].coef_)
    coefs_class.append(return_models.models[key]['model'].coef_)

coefs_loop = np.array(coefs_loop)
coefs_class = np.array(coefs_class)

print('Model coefficients equal:', (coefs_loop == coefs_class).all())
    
"""
Checking predictions are equal
"""

fcast = return_models.forecast(df)

preds = []
for mod in models:
    mod1 = models[mod]
    preds.append(mod1.predict(df).flatten())

preds = np.array(preds).mean(axis=0)

print('Predictions equal:', (preds == fcast).all())







    
    
    


