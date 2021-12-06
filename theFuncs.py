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

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier 
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier

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
        index_col=[3, 2]
    ).groupby(
        ['Symbol', 'DATE']
    ).fillna(
        method='ffill'
    ).sort_index()

    return df if sedols is None else df.loc[(sedols,), :]

def get_stock_return_data(sedols=None):
    df = pd.read_csv(
        "./data/cleaned_return_data_sc.csv", 
        # nrows=3, 
        parse_dates=['DATE'], 
        index_col=0
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
def information_ratio():
    None

def max_drawdown():
    None

def group_by_decile(date, factors):
    return factors.loc[
        (slice(None), date), :
    ].RETURN.groupby(
        pd.qcut(factors.loc[(slice(None), date), :].RETURN.values, 10)
    ).count()
    

def scale_predicted_returns(y_pred):
    None

def information_coefficient_t_statistic(X, y):
    ic, _, _, p_value, _ = st.linregress(X, y)
    return ic, p_value

######################################## RETURNS ########################################
def get_portfolio_returns(weights, date, delta, returns):
    relevant_returns = returns.loc[
        date: date + relativedelta(months=delta)
    ].add(1).cumprod().iloc[-1]

    portfolio_returns = relevant_returns.multiply(weights).sum()
    return portfolio_returns

def get_rus1000_returns(date, delta, returns):
    index_returns = returns.loc[
        date: date + relativedelta(months=delta)
    ].add(1).cumprod().iloc[-1]

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
        index_col=[0, 1]
    )

    date = stock_returns.index[100]

    print(stock_returns.loc[date: date + relativedelta(months=3)])
    print(get_rus1000_returns(date, 10, benchmark_returns))

    X = np.random.random(100)
    y = np.random.random(100)

    print(information_coefficient_t_statistic(X, y))

    print(factors.head())

    print(
        group_by_decile(date, factors)
    )
    