from theFuncs import *

benchmark_returns = get_benchmark_return_data()
stock_returns = get_stock_return_data()

###################### Standardized Return Distribution ######################
# print(group_all_by_decile("data/rus1000_stocks_factors.csv"))

start = datetime.now()

###################### Run All Pipelines ######################

factors = get_stock_factors_data()

t, ic, predictions, feature_importance = return_prediction_evaluation_pipeline(
    factors, 
    output=False
)

portfolio_weights = portfolio_pipeline(predictions, output=False)

###################### Start Given Predictions ######################

# feature_importance = pd.read_csv(
#     "output/feature_importance.csv", 
#     index_col=[0, 1], 
#     parse_dates=["DATE"]
# )

# IC_T = pd.read_csv(
#     "output/IC_T.csv", 
#     index_col=[0, 1], 
#     parse_dates=["DATE"]
# )

# predictions = pd.read_csv(
#     "output/predictions.csv", 
#     index_col=[0, 1, 2], 
#     parse_dates=["DATE"]
# )

# portfolio_weights = portfolio_pipeline(predictions, output=False)

# t, ic = IC_T.groupby(level=1).describe()["T"], IC_T.groupby(level=1).describe().IC

###################### Performance Evaluation Given All Data ######################

# feature_importance = pd.read_csv(
#     "output/feature_importance.csv", 
#     index_col=[0, 1], 
#     parse_dates=["DATE"]
# )

# IC_T = pd.read_csv(
#     "output/IC_T.csv", 
#     index_col=[0, 1], 
#     parse_dates=["DATE"]
# )

# predictions = pd.read_csv(
#     "output/predictions.csv", 
#     index_col=[0, 1, 2], 
#     parse_dates=["DATE"]
# )

# portfolio_weights = pd.read_csv(
#     "output/portfolio_weights.csv",
#     index_col=[0, 1, 2],
#     parse_dates=["DATE"]
# )

# t, ic = IC_T.groupby(level=1).describe()["T"], IC_T.groupby(level=1).describe().IC

###################### Performance Evaluation ######################

########## IC and t-statistic
print(ic, '\n')
print(t, '\n')

########## Performance during Great Recession

all_returns_crash = get_portfolio_benchmark_returns(
    get_monthly_portfolio_returns,
    portfolio_weights,
    stock_returns,
    get_rus1000_monthly_returns,
    benchmark_returns
)["2008-01-01": "2012-03-01"]

cum_returns_crash, mdd_crash, ir_crash = evaluate_all_returns(all_returns_crash)

print(cum_returns_crash, '\n')
print(mdd_crash, '\n')
print(ir_crash, '\n')

all_returns_crash.drop(columns=["Russell 1000 Bench Return"]).add(1).cumprod().plot()
all_returns_crash.add(1).cumprod().plot()
plt.show()

########## Overall Performance 

all_returns = get_portfolio_benchmark_returns(
    get_monthly_portfolio_returns,
    portfolio_weights,
    stock_returns,
    get_rus1000_monthly_returns,
    benchmark_returns
)

print(all_returns.corr())

cum_returns, mdd, ir = evaluate_all_returns(all_returns)

print(cum_returns, '\n')
print(mdd, '\n')
print(ir, '\n')

all_returns.drop(columns=["Russell 1000 Bench Return"]).add(1).cumprod().plot()
all_returns.add(1).cumprod().plot()
plt.show()

print(datetime.now() - start)