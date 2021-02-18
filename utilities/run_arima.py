from utilities.stata_models import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_arima(timeseries, time_col, date_min=None, date_max=None):
    date_min = pd.to_datetime('2020-07-01')
    df = timeseries[timeseries[time_col] > date_min].copy()

    df = df.sort_values(by=time_col).set_index(time_col)
    df = df[['c']]

    test_vals = df['c'].values
    model = sarima(test_vals, (2,1,2))

    yhat = model.predict(1,len(df))

    df['arima'] = yhat

    # this is a canned example, the input and output need tuning

    plt.figure()
    sns.lineplot(data=df)
    plt.show()











