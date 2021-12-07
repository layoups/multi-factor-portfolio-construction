import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.regression.rolling import RollingOLS
import random as rd
from datetime import datetime
from dateutil.relativedelta import relativedelta

import gurobipy as gp
from gurobipy import GRB

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor 
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor

from sklearn import metrics 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials, space_eval
from sklearn.datasets import fetch_openml

######################################## DATA ########################################
def get_stock_factors_data(sedols=None):
    df = pd.read_csv(
        './data/rus1000_stocks_factors.csv', 
        on_bad_lines='skip', 
        header = 2, 
        # nrows = 10000, 
        low_memory=False, 
        converters={
            'SEDOL': lambda x: x[:6], 
            'DATE': lambda x: pd.to_datetime(x) + pd.offsets.MonthBegin(1)
        },
        # parse_dates=['DATE'], 
        index_col=[2, 3]
    ).groupby(
        ['DATE', 'SEDOL']
    ).fillna(
        method='ffill'
    ).sort_index()

    return df if sedols is None else df.loc[(sedols,), :]

def get_stock_return_data(sedols=None):
    df = pd.read_csv(
        "./data/cleaned_return_data_sc.csv", 
        # nrows=3, 
        parse_dates=['DATE'], 
        index_col=[0]
    ).fillna(method='ffill').fillna(0)

    return df if sedols is None else df.loc[(sedols,), :]

def get_benchmark_return_data():
    df = pd.read_csv(
        './data/Benchmark Returns.csv', 
        on_bad_lines='skip', 
        # nrows = 100, 
        low_memory=False, 
        # parse_dates=['Date'],
        converters={'Date': lambda x: pd.to_datetime(x) + pd.offsets.MonthBegin(1)}, 
        index_col=[0]
    )
    df.index.name = "DATE"

    return df

###################################### PERFORMANCE MEASURES ######################################
def information_ratio(returns, index_returns):
    return_diff = returns - index_returns
    final_return_diff = return_diff.iloc[-1]
    return_diff_std = return_diff.std()

    return final_return_diff / return_diff_std * (12.0 / 191)

def max_drawdown(returns):
    i = np.argmax(np.maximum.accumulate(returns) - returns) # end of the period
    j = np.argmax(returns[:i])

    return (returns[j] - returns[i]) / returns[j]

def group_by_decile(date, factors):
    return factors.loc[
            date
        ].RETURN.groupby(
            pd.qcut(factors.loc[date].RETURN.values, 10)
        ).count()

def scale_predicted_returns(y_pred):
    return y_pred.rank(pct=True)

def information_coefficient_t_statistic(X, y):
    ic, _, _, p_value, _ = st.linregress(X, y)
    return ic, p_value

######################################## RETURNS ########################################
def get_portfolio_returns(weights, date, delta, returns):
    relevant_returns = returns.loc[
        date: date + relativedelta(months=delta)
    ].add(1).cumprod()

    portfolio_returns = relevant_returns.multiply(weights).sum()
    return portfolio_returns

def get_rus1000_returns(date, delta, returns):
    index_returns = returns.loc[
        date: date + relativedelta(months=delta)
    ].add(1).cumprod()

    return index_returns

######################################## ML PIPELINE ########################################
class NumericalFeatureCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._scalar = StandardScaler()
        return None

    def fit(self, X, y=None):
        X = self._scalar.fit(X)
        return self

    def transform(self, X, y = None):
        X = pd.DataFrame(
            self._scalar.transform(X), 
            columns=X.columns, 
            index=X.index
        )
        return X

numerical_columns = []

transformer = ColumnTransformer(
    transformers=[
        ("numerical_transformer", NumericalFeatureCleaner(), numerical_columns),
    ], 
    remainder='drop'
)

models = {
    'LinearRegression': ElasticNet(
        max_iter=1000, 
        # tol=1e-3, 
    ),
    'AdaBoost': AdaBoostRegressor(), 
    'KNN': KNeighborsRegressor(), 
}

space = {
    'LinearRegression': {
        'alpha': hp.uniform('alpha', 0.005, 10),
        'l1_ratio': hp.uniform('l1', 0, 1.0)
    },
    'AdaBoost': {
        "n_estimators": hp.randint("n_estimators", 100, 600)
    },
    'KNN': {
        "n_neighbors": hp.randint("n_neighbors", 3, 23)
    }, 
}

@ignore_warnings(category=ConvergenceWarning)
def tune_train_test(X_train, X_test, y_train, y_test, model, params, algo, date):
    trials = Trials()
    thePredictionDict = {}
    thePredictionEvalDict = {}
    
    def objective(params):
        model.set_params(**params)
        
        score = cross_val_score(
            model, X_train, y_train, cv=3, n_jobs=-1, error_score=0.99
        )
        return {'loss': 1 - np.mean(score), 'status': STATUS_OK}

    best_classifier = fmin(objective, params, algo=tpe.suggest, max_evals=3, trials=trials)
    best_params = space_eval(params, best_classifier)

    opti = model
    opti.set_params(**best_params)

    opti_model = opti.fit(
        X_train,
        y_train
    )
    y_pred = opti_model.predict(X_test)
    
    thePredictionEvalDict["Model"] = algo
    thePredictionEvalDict["Date"] = date
    thePredictionEvalDict["IC"], thePredictionEvalDict["T"] =\
        information_coefficient_t_statistic(y_test, y_pred)
    
    thePredictionDict
    return thePredictionDict, thePredictionEvalDict



if __name__ == "__main__":
    benchmark_returns = get_benchmark_return_data()

    stock_returns = pd.read_csv(
        "data/cleaned_return_data.csv", 
        parse_dates=["DATE"],
        index_col=[0]
    )

    factors = pd.read_csv(
        "data/rus1000_stocks_factors_subset.csv",
        converters={"DATE": lambda x: pd.to_datetime(x) + pd.offsets.MonthBegin(1)},
        # parse_dates=["DATE"],
        index_col=[1, 0]
    )

    date = stock_returns.index[100]

    print(stock_returns.loc[date: date + relativedelta(months=3)])
    print(get_rus1000_returns(date, 1000, benchmark_returns))

    X = np.random.random(100)
    y = np.random.random(100)

    print(information_coefficient_t_statistic(X, y))

    print(factors.head())

    print(
        group_by_decile(date, factors)
    )

    print(
        information_ratio(
            get_rus1000_returns(date, 1000, benchmark_returns)["Russell 1000 Bench Return"], get_rus1000_returns(date, 1000, benchmark_returns)["Russell 1000 Bench Return"]
        )
    )

    print(
        max_drawdown(
            get_rus1000_returns(date, 1000, benchmark_returns)["Russell 1000 Bench Return"]
        )
    )

    # print(scale_predicted_returns(factors))


    