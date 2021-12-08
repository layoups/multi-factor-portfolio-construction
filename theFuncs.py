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
from sklearn.tree import DecisionTreeRegressor

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
    sedols = pd.read_csv("data/sedols.csv", index_col=False)
    df = pd.read_csv(
        './data/rus1000_stocks_factors.csv', 
        on_bad_lines='skip', 
        header = 2, 
        # nrows = 100000, 
        low_memory=False, 
        converters={
            'SEDOL': lambda x: x[:6], 
            'DATE': lambda x: pd.to_datetime(x) + pd.offsets.MonthBegin(1)
        },
        # parse_dates=['DATE'], 
        index_col=[2, 3]
    ).sort_index()

    df = df[df.index.get_level_values("SEDOL").isin(sedols.SEDOLS)]
    df = df[~df.index.duplicated(keep='first')]

    df["TARGET"] = df.groupby(level="SEDOL").RETURN.shift(-1)
    df = df.groupby(level=1).fillna(method='ffill').fillna(0)
    df["MODEL"] = "CTEF"
    df.sort_index(inplace=True)

    return df

def get_stock_return_data(sedols=None):
    df = pd.read_csv(
        # "./data/cleaned_return_data_sc.csv", 
        "./data/cleaned_return_data.csv", 
        # nrows=3, 
        parse_dates=['DATE'], 
        index_col=[0]
    ).fillna(method='ffill').fillna(0).sort_index()

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
    ).sort_index()
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
    ][weights.index.get_level_values(1)].add(1).cumprod().copy()

    for i in weights.index:
        relevant_returns[[i]] = relevant_returns[[i]].multiply(weights.loc[i])

    # portfolio_returns = relevant_returns.multiply(weights).sum()
    # portfolio_returns = relevant_returns.multiply(weights, axis='index').sum()
    return relevant_returns.sum(axis=1)

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

numerical_columns = ["RCP", "RBP", "RSP", "REP", "RDP", "RPM71", "RSTDEV", "ROA1", "9MFR", "8MFR"]
# RCP,RBP,RSP,REP,RDP,RPM71,RSTDEV,ROE1,ROE3,ROE5,ROA1,ROA3,ROIC,BR1,BR2,EP1,EP2,RV1,RV2,CTEF,9MFR,8MFR,LIT

transformer = ColumnTransformer(
    transformers=[
        ("numerical_transformer", NumericalFeatureCleaner(), numerical_columns),
    ], 
    remainder='drop'
)

def custom_scoring(y_true, y_pred):
    y_pred = scale_predicted_returns(
        pd.Series(y_pred, index=pd.RangeIndex(len(y_true)))
    ).multiply(100)
    return metrics.mean_absolute_error(y_true, y_pred)

models = {
    'LinearRegression': ElasticNet(
        max_iter=1000, 
        tol=1e-3, 
    ),
    'AdaBoost': AdaBoostRegressor(), 
    'DecisionTree': DecisionTreeRegressor(), 
}

space = {
    'LinearRegression': {
        'alpha': hp.uniform('alpha', 0.5, 2),
        'l1_ratio': hp.uniform('l1', 1e-2, 1)
    },
    'AdaBoost': {
        "n_estimators": hp.randint("n_estimators", 300, 500),
    },
    'DecisionTree': {
        'max_depth': hp.randint('max_depth', 1, 7),
        'min_samples_leaf': hp.randint('min_samples_leaf', 1, 20),
        'min_samples_split': hp.randint('min_samples_split', 2, 40),
    }, 
}

@ignore_warnings(category=ConvergenceWarning)
def tune_train_test(
    X_train, X_test, y_train, y_test, model, params, algo, date, index, columns=numerical_columns
):
    trials = Trials()
    thePredictionDict = []
    thePredictionEvalDict = {}
    theFeatureImportanceDict = {}
    
    def objective(params):
        model.set_params(**params)
        
        score = cross_val_score(
            model, X_train, y_train, cv=3, n_jobs=-1, error_score=0.99, 
            scoring=metrics.make_scorer(custom_scoring, greater_is_better=False)
            # scoring='neg_mean_absolute_error'
        )
        return {'loss':  -np.mean(score), 'status': STATUS_OK}

    best_classifier = fmin(
        objective, params, algo=tpe.suggest, max_evals=10, trials=trials, show_progressbar=False
    )
    best_params = space_eval(params, best_classifier)

    opti = model
    opti.set_params(**best_params)

    opti_model = opti.fit(
        X_train,
        y_train
    )
    y_pred = opti_model.predict(X_test)
    
    new_y_pred = scale_predicted_returns(pd.Series(y_pred, index=index))

    thePredictionEvalDict["MODEL"] = algo
    thePredictionEvalDict["DATE"] = date
    
    thePredictionEvalDict["IC"], thePredictionEvalDict["T"] =\
        information_coefficient_t_statistic(y_test.div(100), new_y_pred)

    for i in index:
        thePredictionDict += [
            {"MODEL": algo, "DATE": date, "SEDOL": i, "RETURN": new_y_pred.loc[i]}
        ]

    if algo == "LinearRegression":
        coef_sig = opti_model.coef_
        theFeatureImportanceDict["DATE"] = date
        theFeatureImportanceDict["MODEL"] = algo
        for i in range(len(numerical_columns)):
            theFeatureImportanceDict[numerical_columns[i]] = coef_sig[i]
    if algo == "AdaBoost":
        coef_sig = opti_model.feature_importances_
        theFeatureImportanceDict["DATE"] = date
        theFeatureImportanceDict["MODEL"] = algo
        for i in range(len(numerical_columns)):
            theFeatureImportanceDict[numerical_columns[i]] = coef_sig[i]

    return thePredictionDict, thePredictionEvalDict, theFeatureImportanceDict

def return_prediction_evaluation_pipeline(
    start="2004-11-01", end="2018-11-01", 
    eval_path="output/IC_T_Final.csv", predictions_path="output/predictions_Final.csv", 
    feature_path="output/feature_importance_Final.csv",
    output=True
):
    eval_df = []
    return_df = []
    feature_importance_df = []

    for date in pd.date_range(start, end, freq="MS"):
        print(date)
        
        # date = datetime.strptime("2004-11-01", "%Y-%m-%d")
        start_date = date - relativedelta(months=12)

        X_train = factors.loc[start_date: date + relativedelta(months=-1)]
        y_train = X_train.TARGET

        X_test = factors.loc[date]
        y_test = X_test.TARGET

        X_train_tr, X_test_tr = transformer.fit_transform(X_train), transformer.transform(X_test)
        for algo in models:
            return_instance, eval_instance, feature_instance = tune_train_test(
                X_train_tr, 
                X_test_tr,
                y_train,
                y_test,
                models[algo],
                space[algo],
                algo,
                date,
                X_test.index.get_level_values("SEDOL").unique()
            )

            eval_df += [eval_instance]
            return_df += return_instance
            feature_importance_df += [feature_instance]

        ctef = factors.loc[date].CTEF.rank(pct=True)
        ctefPredictionEvalDict = {}
        ctefPredictionEvalDict["MODEL"] = "CTEF"
        ctefPredictionEvalDict["DATE"] = date
        ctefPredictionEvalDict["IC"], ctefPredictionEvalDict["T"] =\
            information_coefficient_t_statistic(y_test.div(100), ctef)
        eval_df += [ctefPredictionEvalDict]

    eval_df = pd.DataFrame(eval_df).set_index(["DATE", "MODEL"]).sort_index()
    return_df = pd.DataFrame(return_df).set_index(["DATE", "MODEL", "SEDOL"])
    return_df = pd.concat(
        [
            return_df, 
            factors[["MODEL", "CTEF"]].loc[start: end].set_index(
                "MODEL", append=True
            ).reorder_levels(
                ["DATE", "MODEL", "SEDOL"]
            ).groupby(level=0).rank(pct=True).rename(columns={"CTEF": "RETURN"})
        ]
    ).sort_index()
    feature_importance_df = pd.DataFrame(
        feature_importance_df
    ).set_index(["DATE", "MODEL"]).sort_index().dropna()

    if output:
        eval_df.to_csv(eval_path)
        return_df.to_csv(predictions_path)
        feature_importance_df.to_csv(feature_path)

    return eval_df.groupby(level=1).describe()["T"], eval_df.groupby(level=1).describe()["IC"]

######################################## PORTFOLIO ########################################
def get_prediction_thresholds(predictions, H=0.7):
    predictions["MASK"] = (predictions >= H).rename(columns={"RETURN": "MASK"})
    thresholds = (predictions).join(
        (predictions.drop(columns=["RETURN"])).groupby(level=[0, 1]).sum().rename(columns={"MASK": "SUM"})
    )
    thresholds["WEIGHT"] = thresholds.MASK / thresholds.SUM
    return thresholds[thresholds.MASK == True].drop(columns=["SUM", "MASK"])

def rebalance_portfolio(date, algo, return_thresholds, portfolio_weights, K=4):
    A = return_thresholds

    B = portfolio_weights.loc[
        (date + relativedelta(months=-1), algo,)
    ]

    B_not_A_K = B.loc[
        np.setdiff1d(B.index, A.index)
    ].sort_values(by="RETURN").iloc[:K]
    A_not_B_K = A.loc[
        np.setdiff1d(A.index, B.index)
    ].sort_values(by="RETURN", ascending=False).iloc[:K]

    B_star = pd.concat(
        [
            A_not_B_K.copy(),
            B.drop(index=B_not_A_K.index),
        ]
    )
    B_star["WEIGHT"] = 1.0 / len(B_star)
    B_star["DATE"] = date
    B_star["MODEL"] = algo
    
    return B_star.set_index(
                ["DATE", "MODEL"], append=True
            ).reorder_levels(
                    ["DATE", "MODEL", "SEDOL"]
            )

def portfolio_pipeline(
    predictions, H=0.7, K=4, start="2005-01-01", end="2018-11-01",
    path="output/portfolio_weights.csv", output=True
):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    return_thresholds = get_prediction_thresholds(predictions, H)
    portfolio_weights = return_thresholds.loc[:start_date + relativedelta(months=-1)]
    algos = ["LinearRegression", "CTEF", "AdaBoost", "DecisionTree"]

    for date in pd.date_range(start, end, freq="MS"):
        print(date)
        curr_return_thresholds = return_thresholds.loc[date]
        curr_algos = curr_return_thresholds.index.get_level_values("MODEL").unique()
        for algo in curr_algos:
            new_weights = rebalance_portfolio(
                date, algo, curr_return_thresholds.loc[algo], portfolio_weights, K
            )
            new_weights = new_weights[~new_weights.index.duplicated(keep='first')].dropna()

            portfolio_weights = pd.concat(
                [
                    portfolio_weights,
                    new_weights
                ]
            ).sort_index()

        for algo in np.setdiff1d(algos, curr_algos):
            new_weights = portfolio_weights.loc[(date + relativedelta(months=-1), algo)]
            new_weights["DATE"] = date
            new_weights["MODEL"] = algo
            portfolio_weights = pd.concat(
                [
                    portfolio_weights,
                    new_weights.set_index(
                        ["DATE", "MODEL"], append=True
                    ).reorder_levels(
                            ["DATE", "MODEL", "SEDOL"]
                    )
                ]
            ).sort_index()
    portfolio_weights.drop(columns=["RETURN"], inplace=True)
    if output:
        portfolio_weights.to_csv(path)

    return portfolio_weights


if __name__ == "__main__":
    # benchmark_returns = get_benchmark_return_data()
    # factors = get_stock_factors_data()
    stock_returns = get_stock_return_data()

    start = datetime.now()

    # t, ic = return_prediction_evaluation_pipeline()
    # print(ic)
    # print(t)

    # t, ic = return_prediction_evaluation_pipeline(
    #     eval_path="output/IC_T_CS.csv", predictions_path="output/predictions_CS.csv", feature_path="output/feature_importance_CS.csv"
    # )
    # print(ic)
    # print(t)

    feature_importance = pd.read_csv(
        "output/feature_importance_Final.csv", 
        index_col=[0, 1], 
        parse_dates=["DATE"]
    )
    IC_T = pd.read_csv(
        "output/IC_T_Final.csv", 
        index_col=[0, 1], 
        parse_dates=["DATE"]
    )
    predictions = pd.read_csv(
        "output/predictions_Final.csv", 
        index_col=[0, 1, 2], 
        parse_dates=["DATE"]
    )

    # print(predictions.loc[("2011-03-01", "AdaBoost",), :].describe())

    # get_prediction_thresholds(predictions).to_csv("output/return_thresholds.csv")

    # for date in thresholds.index.get_level_values(0).unique():
    #     for algo in ["LinearRegression", "CTEF", "AdaBoost"]:
    #         if algo in thresholds.loc[date].index.get_level_values(0).unique():
    #             continue
    #         else:
    #             print(date, algo)

    # portfolio_weights = portfolio_pipeline(predictions)    

    portfolio_weights = pd.read_csv(
        "output/portfolio_weights.csv",
        index_col=[0, 1, 2],
        parse_dates=["DATE"]
    )

    # print(portfolio_weights.loc["2005-01-01"])

    agg_weights = portfolio_weights.groupby(level=[0, 1]).sum()
    print(agg_weights.describe())

    # filter1 = agg_weights  1

    # print(
    #     filter1[filter1.WEIGHT == True]
    # )

    # print(
    #     get_portfolio_returns(
    #         portfolio_weights.loc[("2005-01-01", "AdaBoost"): ("2005-03-01", "AdaBoost")].WEIGHT, 
    #         portfolio_weights.index.get_level_values(0).unique()[2], 
    #         2, 
    #         stock_returns
    #     )
    # )


    print(datetime.now() - start)




    