import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.regression.rolling import RollingOLS
import random as rd
from datetime import datetime

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

def get_stock_factors_data(sedols=None):
    df = pd.read_csv(
        './data/rus1000_stocks_factors.csv', 
        on_bad_lines='skip', 
        header = 2, 
        # nrows = 10000, 
        low_memory=False, 
        converters={'SEDOL': (lambda x: x[:6])},
        parse_dates=['DATE'], 
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
        parse_dates=['Date'], 
        index_col=[0]
    )
    df.index.name = "DATE"

    return df



if __name__ == "__main__":
    benchmark_returns = get_benchmark_return_data()

    stock_returns = pd.read_csv(
        "data/cleaned_return_data.csv", 
        parse_dates=["DATE"],
        index_col=[0]
    )

    factors = pd.read_csv(
        "data/rus1000_stocks_factors_subset.csv",
        parse_dates=["DATE"],
        index_col=[0, 1]
    )

    print(factors.head())
    