from fbprophet import Prophet
from utilities.prep_stock_data import split_stock_data
# https://facebook.github.io/prophet/docs/quick_start.html#python-api
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import math
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

# https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python

import os

class suppress_stdout_stderr(object):
    '''
    # https://github.com/facebook/prophet/issues/223
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def train_prophet(df, time_col, data_col, run_model=False, window_size=15):
    ds_col = 'ds'
    y_col = 'y'
    key_map = {time_col: ds_col, data_col: y_col}

    df['week'] = df['t'].dt.isocalendar().week
    df['day'] = df['t'].dt.isocalendar().day

    # produce a unique tuple (per year) of a week and day number
    df['day'] = list(zip(df['week'], df['day']))

    df.drop(columns=['week'], inplace=True)

    train, valid, test = split_stock_data(df=df, time_col=time_col)

    # convert col names per facebook api needs
    train = train.rename(columns=key_map)

    # length of dataset for graphing later
    n = 'n=' + str(len(train))

    # group df by day, a week-day tuple
    df = train.groupby('day')

    def index_marks(nrows, chunk_size):
        # https://yaoyao.codes/pandas/2018/01/23/pandas-split-a-dataframe-into-chunks
        return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)

    def split(dfm, chunk_size):
        indices = index_marks(dfm.shape[0], chunk_size)
        return np.split(dfm, indices)

    counter = 0

    predictions = pd.DataFrame()
    targets = pd.DataFrame()

    if run_model:

        print('Running Facebook Prophet')

        if window_size:
            # estimated iterations
            est_iters = len(df)
            pbar = tqdm(total=est_iters)
            pbar.update(0)
            # loop through a grouped dataframe
            for group_name, group_frame in df:
                chunks = split(group_frame, window_size)
                idx = 0

                incre_update = 1 / len(chunks)

                while 1:
                    # fractional increment a progress bar
                    pbar.update(incre_update)
                    # instantiate a new model for each chunk
                    m = Prophet(yearly_seasonality=False
                                ,weekly_seasonality=False
                                ,daily_seasonality=False)
                    # a chunk is a period of time, default of 15 minutes
                    chunk = chunks[idx]
                    next_idx = idx + 1

                    # train with a sequence and predict the start of the next
                    if next_idx > len(chunks) - 1:
                        break
                    else:
                        target = chunks[next_idx].iloc[0]

                    # suppress pystan outputs for each model fit
                    with suppress_stdout_stderr():
                        m.fit(chunk)
                    # extends the current sequence of time by one
                    future = m.make_future_dataframe(periods=1, freq="1min")
                    # the 'yhat' column is created by the prophet API
                    forecast = m.predict(future)[[ds_col, 'yhat']].iloc[-1]

                    idx += 1
                    if idx > 30:
                        print("something is wrong")
                    targets = targets.append(target)
                    predictions = predictions.append(forecast)

            results = pd.merge(targets, predictions, left_on=ds_col, right_on=ds_col)
            results.drop(columns=['day'], inplace=True)

            results.to_csv('data/facebook_prophet_results.csv')

            print()
            error = mean_absolute_error(results['y'].values, results['yhat'].values) * 100

            print('The MAPE for facebook prophet is', error)

            fig, ax, = plt.subplots()
            ax.plot(results['ds'], results['y'], color='red', label='Actual Price')
            ax.plot(results['ds'], results['yhat']
                    , color='blue'
                    , marker='o'
                    , markersize=3
                    , linestyle='dashed'
                    , linewidth=1
                    , label='Predicted Price'
                    )
            ax.set_title('Amazon Stock Price Prediction (Dev Data)\nWith Facebook Prophet ' + n)
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            fig.autofmt_xdate()
            plt.show()

        else:
            # else, run training on a large range of data and predict the next
            m = Prophet(changepoint_prior_scale=0.005, yearly_seasonality=False)
            m.fit(df[:15])  # df is a pandas.DataFrame with 'y' and 'ds' columns

            future = m.make_future_dataframe(periods = 1, freq = "1min")
            future = future[(future['ds'].dt.hour >= 9) | (future['ds'].dt.hour  <= 16)]
            future = future[(future['ds'].dt.weekday < 5)]

            forecast = m.predict(future)

            fcst = forecast[[ds_col, 'yhat']]

            merged = pd.merge(train, fcst, left_on=['t'], right_on='ds')

            preds = merged['yhat'].values
            targets = merged['c'].values

            # error = mean_squared_error(targets, preds)
            error = mean_absolute_error(targets, preds) * 100
            print('The MAPE for facebook prophet is', error)

            fig, ax, = plt.subplots()
            ax.plot(merged['t'], merged['c'], color='red', label='Actual Price')
            ax.plot(merged['t'], merged['yhat']
                    , color='blue'
                    , marker='o'
                    , markersize=3
                    , linestyle='dashed'
                    , linewidth=1
                    , label='Predicted Price'
                    )
            ax.set_title('Amazon Stock Price Prediction (Dev Data)\nWith Facebook Prophet ' + n)
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            fig.autofmt_xdate()
            plt.show()

            plt.plot()
            fig1 = m.plot(forecast)
            plt.title('Amazon Stock Price Prediction (Dev Data)\nWith Facebook Prophet ' + n)
            plt.tight_layout()
            fig1.show()

            #fixme subplots and title
            plt.plot()
            fig2 = m.plot_components(forecast)
            plt.title('Amazon Stock Price Components (Dev Data)\nWith Facebook Prophet')
            plt.tight_layout()
            fig2.show()
            print('done')

    else:
        print("Set up Facebook Prophet, did not run model")


