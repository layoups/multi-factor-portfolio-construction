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
def get_stock_factors_data(path='./data/rus1000_stocks_factors.csv'):
    sedols = pd.read_csv("data/sedols.csv", index_col=False)
    df = pd.read_csv(
        path, 
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

def get_stock_return_data():
    sedols = pd.read_csv("data/sedols.csv", index_col=False)
    df = pd.read_csv(
        # "./data/cleaned_return_data_sc.csv", 
        "./data/cleaned_return_data.csv", 
        # nrows=3, 
        parse_dates=['DATE'], 
        index_col=[0]
    ).fillna(method='ffill').fillna(0).div(100).sort_index()

    return df[sedols.SEDOLS]

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
def information_ratio(returns):
    return returns.mean() / returns.std() * np.sqrt(12)

def max_drawdown(returns):
    i = np.argmax(np.maximum.accumulate(returns) - returns)
    j = np.argmax(returns[:i])

    return (returns[j] - returns[i]) / returns[j]

def group_by_decile(date, factors):
    factors.loc[date].RETURN.plot.hist()
    plt.show()
    return factors.loc[
            date
        ].RETURN.groupby(
            pd.qcut(factors.loc[date].RETURN.values, 10, duplicates='drop')
        ).count()

def group_all_by_decile(path_to_factors):
    factors = get_stock_factors_data(path_to_factors)
    factors.RETURN.plot.hist()
    plt.show()
    return factors.RETURN.groupby(
            pd.qcut(factors.RETURN.values, 10, duplicates='drop')
        ).count()


def scale_predicted_returns(y_pred):
    return y_pred.rank(pct=True)

def information_coefficient_t_statistic(X, y):
    ic, _, _, p_value, _ = st.linregress(X, y)
    return ic, p_value

######################################## RETURNS ########################################
def get_portfolio_weights_for_model(model, portfolio_weights):
    return portfolio_weights[
            portfolio_weights.index.get_level_values("MODEL") == model
        ].droplevel(level=1).pivot_table(
            values="WEIGHT", 
            columns="SEDOL",
            index="DATE"
        ).fillna(0)

def get_portfolio_weights_with_model(model, portfolio_weights):
    return portfolio_weights[
            portfolio_weights.index.get_level_values("MODEL") == model
        ].pivot_table(
            values="WEIGHT", 
            columns="SEDOL",
            index=["DATE", "MODEL"]
        ).fillna(0)

def get_portfolio_weights_for_all_models(models, portfolio_weights):
    return pd.concat(
        [
            get_portfolio_weights_with_model(model, portfolio_weights) 
            for model in models
        ], 
        axis=0
    ).fillna(0)

def get_monthly_portfolio_returns(weights, returns):
    return weights.multiply(
        returns[weights.columns].loc[weights.index]
    ).sum(axis=1)

def get_cumulative_portfolio_returns(weights, returns):
    return get_monthly_portfolio_returns(weights, returns).add(1).cumprod()

def get_rus1000_monthly_returns(returns):
    return returns["Russell 1000 Bench Return"]

def get_rus1000_cumulative_returns(returns):
    return get_rus1000_monthly_returns(returns).add(1).cumprod()


def get_portfolio_benchmark_returns(
    portfolio_return_metric, portfolio_weights, stock_returns, 
    benchmark_return_metric, benchmark_returns
):
    algos = portfolio_weights.index.get_level_values("MODEL").unique()
    to_concat = [
        portfolio_return_metric(
                get_portfolio_weights_for_model(model, portfolio_weights), 
                stock_returns
            )
        for model in algos
    ]
    to_concat += [benchmark_return_metric(benchmark_returns)]
    return pd.concat(
        to_concat,
        axis=1
    ).dropna().rename(
        columns={i: algos[i] for i in range(len(algos))}
    )

def evaluate_all_returns(all_returns):
    return (
        all_returns.apply(lambda x: x.add(1).cumprod().iloc[-1]), 
        all_returns.apply(lambda x: max_drawdown(x.add(1).cumprod())), 
        all_returns.apply(lambda x: information_ratio(x))
    )

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
    # 'DecisionTree': DecisionTreeRegressor(), 
    'KNN': KNeighborsRegressor()
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
    'KNN': {
        'n_neighbors': hp.uniformint('n_neighbors', 3, 23)
    }
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

    return eval_df.groupby(level=1).describe()["T"], eval_df.groupby(level=1).describe()["IC"], return_df, feature_importance_df

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
    predictions, H=0.7, K=4, start="2004-12-01", end="2018-11-01",
    path="output/portfolio_weights.csv", output=True, 
    algos=["LinearRegression", "CTEF", "AdaBoost", "KNN"]
):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    return_thresholds = get_prediction_thresholds(predictions, H)
    portfolio_weights = return_thresholds.loc[:start_date + relativedelta(months=-1)]
    for date in pd.date_range(start, end, freq="MS"):
        print(date)
        curr_return_thresholds = return_thresholds.loc[date]
        curr_algos = curr_return_thresholds.index.get_level_values("MODEL").unique()
        for algo in np.intersect1d(curr_algos, algos):
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
    # print(group_all_by_decile("data/rus1000_stocks_factors.csv"))

    start = datetime.now()

    # t, ic = return_prediction_evaluation_pipeline()
    # print(ic)
    # print(t)

    # factors = get_stock_factors_data()

    # t, ic, predictions, feature_importance = return_prediction_evaluation_pipeline(
    #     eval_path="./output/IC_T_mae.csv", predictions_path="./output/predictions_mae.csv", feature_path="./output/feature_importance_mae.csv"
    # )

    # predictions = pd.read_csv(
    #     "output/predictions_prime.csv", 
    #     index_col=[0, 1, 2], 
    #     parse_dates=["DATE"]
    # )


    # portfolio_weights = portfolio_pipeline(predictions, H=0.7, K=4, path='./output/portfolio_weights_mae.csv') 

    benchmark_returns = get_benchmark_return_data()
    stock_returns = get_stock_return_data()

    # feature_importance = pd.read_csv(
    #     "output/feature_importance_prime.csv", 
    #     index_col=[0, 1], 
    #     parse_dates=["DATE"]
    # )
    IC_T = pd.read_csv(
        "output/IC_T.csv", 
        index_col=[0, 1], 
        parse_dates=["DATE"]
    )
    # predictions = pd.read_csv(
    #     "output/predictions_prime.csv", 
    #     index_col=[0, 1, 2], 
    #     parse_dates=["DATE"]
    # )
    portfolio_weights = pd.read_csv(
        "output/portfolio_weights.csv",
        index_col=[0, 1, 2],
        parse_dates=["DATE"]
    )
    # all_returns = pd.read_csv(
    #     "output/summaries/all_returns_1.csv",
    #     index_col=[0],
    #     parse_dates=["DATE"]
    # )

    # print(IC_T.groupby(level=1).describe().IC)
    # print(IC_T.groupby(level=1).describe()["T"])
    # get_prediction_thresholds(predictions).to_csv("output/return_thresholds_KNN.csv")

    all_returns_crash = get_portfolio_benchmark_returns(
            get_monthly_portfolio_returns,
            portfolio_weights,
            stock_returns,
            get_rus1000_monthly_returns,
            benchmark_returns
        )["2008-01-01": "2012-03-01"]

    all_returns = get_portfolio_benchmark_returns(
            get_monthly_portfolio_returns,
            portfolio_weights,
            stock_returns,
            get_rus1000_monthly_returns,
            benchmark_returns
        )

    print(all_returns.corr())

    # all_returns.to_csv("output/summaries/all_returns_13.csv")
    # get_portfolio_weights_for_all_models(["LinearRegression", "CTEF", "AdaBoost", "KNN"], portfolio_weights).to_csv("output/summaries/weights_13.csv")
    
    # print(ic, '\n')
    # print(t, '\n')

    cum_returns, mdd, ir = evaluate_all_returns(all_returns)
    cum_returns_crash, mdd_crash, ir_crash = evaluate_all_returns(all_returns_crash)

    print(cum_returns, '\n')
    print(mdd, '\n')
    print(ir, '\n')
    print(cum_returns_crash, '\n')
    print(mdd_crash, '\n')
    print(ir_crash, '\n')

    all_returns.drop(columns=["Russell 1000 Bench Return"]).add(1).cumprod().plot()
    all_returns.add(1).cumprod().plot()
    plt.show()
    all_returns_crash.drop(columns=["Russell 1000 Bench Return"]).add(1).cumprod().plot()
    all_returns_crash.add(1).cumprod().plot()
    plt.show()

    print(datetime.now() - start)




    