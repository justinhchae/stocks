from utilities.stata_models import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_arima(timeseries, time_col, date_min=None, date_max=None, run_model=True):
    date_min = pd.to_datetime('2020-07-01')
    df = timeseries[timeseries[time_col] > date_min].copy()

    df = df.sort_values(by=time_col).set_index(time_col)
    df = df[['c']]

    test_vals = df['c'].values

    if run_model:
        model = sarima(test_vals, (2,1,2))

        yhat = model.predict(1,len(df))

        df['arima'] = yhat

        plt.figure()
        sns.lineplot(data=df)
        plt.show()
    else:
        print('skipped run model')











