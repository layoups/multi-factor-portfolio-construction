Files
    - theFuncs.py, contains all classes and functions
    - main.py, the main execution code
    - data/, the folder contains all data sources necessary to run the project
    - output/, contains output required to run model performance evaluation without having to run all the project pipelines, in addition to the required project output in the output/required/ folder

Dependencies
    - Python 3.7+, developed with Python 3.8.8
        - numpy
        - pandas
        - matplotlib.pyplot
        - scipy.stats 
        - datetime
        - dateutil.relativedelta
        - sklearn
        - hyperopt

Instructions to Run the Main Code
    - Command: path/to/python path/to/main.py
    - main.py is organized in sections
        - Standardized Return Distribution = Task 2
        - Run All Pipelines = run complete project from scratch
        - Start Given Predictions = build portfolios given saved model predictions
        - Performance Evaluation Given All Data = simply evaluate portfolios given model predictions and portfolio weights
        - Performance Evaluation = self-evident