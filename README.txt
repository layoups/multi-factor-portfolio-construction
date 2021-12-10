Files
- theFuncs.py, contains all classes, functions, and project execution
- data/, the folder contains all data sources necessary to run the project
- output/, contains output required to run model performance evaluation without having to run all the project pipelines, in addition to the required project output in the output/required/ folder

Dependencies
- Python 3.7+, developed with Python 3.8.8
    - numpy
    - pandas
    - matplotlib.pyplot
    - scipy.stats 
    - 
from statsmodels.regression.rolling  RollingOLS
 random rd
from datetime  datetime
from dateutil.relativedelta  relativedelta

 gurobipy gp
from gurobipy  GRB

from sklearn.ensemble  RandomForestRegressor, AdaBoostRegressor 
from sklearn.linear_model  LinearRegression, Lasso, ElasticNet
from sklearn.neighbors  KNeighborsRegressor
from sklearn.tree  DecisionTreeRegressor

from sklearn  metrics 
from sklearn.model_selection  cross_val_score
from sklearn.preprocessing  StandardScaler 
from sklearn.base  BaseEstimator, TransformerMixin
from sklearn.model_selection  train_test_split
from sklearn.compose  ColumnTransformer
from sklearn.pipeline  Pipeline

from sklearn.utils._testing  ignore_warnings
from sklearn.exceptions  ConvergenceWarning

from hyperopt  tpe, hp, fmin, STATUS_OK,Trials, space_eval
from sklearn.datasets  fetch_openml