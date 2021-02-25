from fbprophet import Prophet
from utilities.prep_stock_data import split_stock_data
# https://facebook.github.io/prophet/docs/quick_start.html#python-api
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import math
import numpy as np
from multiprocessing import Pool, cpu_count


from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
# https://medium.com/spikelab/forecasting-multiples-time-series-using-prophet-in-parallel-2515abd1a245

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

def index_marks(nrows, chunk_size):
    """
    a helper function for split()
    return an index of chunk size
    https://yaoyao.codes/pandas/2018/01/23/pandas-split-a-dataframe-into-chunks
    """
    return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)

def split(dfm, chunk_size):
    """
    a helper function to split and chunk a dataframe by row
    :params: dfm -> a dataframe
    :params: chunk_size -> an inteter
    :returns: a list of chunked dataframes of size chunk_size
    """
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

def setup_prophet(df, time_col, data_col
                  , seasonal_unit='day'
                  , window_size=15
                  , split_tts=False):
    """
    :params:
        df -> a dataframe
        time_col -> the col name having timestamps, usually 't'
        data_col -> the col name having data, usually 'c'
        run_model -> default to False, True if run the model
        window_size -> default to 15, determines sequence length for train and prediction
        split_tts -> if true, split and train on train data only, else train on entire dataset
        IF window_size is not None, training will take place in a training window.
        IF window_size is None, training will take place on entire dataset and default to predicting the next n units in validation

    :returns: null
    :action: run the facebook prophet to train on window_size minutes of data and predict the next minute on each day

    """
    # these column names are required by the facebook api
    ds_col = 'ds'
    y_col = 'y'
    key_map = {time_col: ds_col, data_col: y_col}

    # extract the week number and day number for each timestamp for sorting
    df['week'] = df['t'].dt.isocalendar().week
    df['day'] = df['t'].dt.isocalendar().day

    # produce a unique tuple (per year) of a week and day number
    df['day'] = list(zip(df['week'], df['day']))
    df.drop(columns=['week'], inplace=True)

    # initialize valid test data
    valid = None
    # split data
    if split_tts:
        train, valid, test = split_stock_data(df=df, time_col=time_col)
    else:
        train = df

    # convert col names per facebook api needs
    train = train.rename(columns=key_map)

    # group df by day, a week-day tuple
    df = train.groupby(seasonal_unit)

    prophet_data = []
    for group_name, group_frame in df:
        chunk_data = []

        # in each seasonal_unit, chunk data into window_size chunks
        chunks = split(group_frame, window_size)

        # initialize an index to return each chunk in sequence
        idx = 0

        while 1:
            # return a data chunk of window_size on index idx
            chunk = chunks[idx]

            # set up index of next chunk in sequence
            next_idx = idx + 1

            # at then end of a seasonal_unit, break if index out of range
            if next_idx > len(chunks) - 1:
                break
            else:
                # otherwise, return the first value of next sequence as y target
                target = chunks[next_idx].head(1)

            # increment the chunk
            idx += 1
            x_i = chunk.reset_index(drop=True)
            y_i = target.reset_index(drop=True)
            # save targets y and forecast predictions yhat
            chunk_data.append((x_i, y_i))

        prophet_data.append(chunk_data)

    return prophet_data

def run_prophet(chunked_data, window_size=15, n_prediction_units=1, prediction_frequency='1min'):
    ds_col = 'ds'
    results = []

    for x, y in chunked_data:
        m = Prophet(yearly_seasonality=False
                                    ,weekly_seasonality=False
                                    ,daily_seasonality=False
                                    ,uncertainty_samples=False)
        with suppress_stdout_stderr():
            m.fit(x)

        future = m.make_future_dataframe(periods=n_prediction_units
                                         , freq=prediction_frequency
                                         , include_history=False)

        yhat = m.predict(future)[[ds_col, 'yhat']]

        result = pd.merge(y, yhat, left_on=ds_col, right_on=ds_col).drop(columns='day')

        results.append(result)

    return results

#TODO: finish refactoring prophet to predict on default long sequence

#
# if run_model:
#
#     if window_size:
#         # estimated iterations
#         est_iters = len(df)
#
#         # create a progress bar for feedback during long computations
#         pbar = tqdm(total=est_iters)
#         # init progress bar with 0
#         pbar.update(0)
#         # loop through a grouped dataframe by seasonal_unit
#
#         # emtpy dataframes to hold model results
#         predictions = pd.DataFrame()
#         targets = pd.DataFrame()
#
#         accu = 0
#
#         for group_name, group_frame in df:
#
#             # in each seasonal_unit, chunk data into window_size chunks
#             chunks = split(group_frame, window_size)
#
#             # initialize an index to return each chunk in sequence
#             idx = 0
#
#             # create a fractional increment based on chunks
#             incre_update = 1 / len(chunks)
#
#             while 1:
#                 # increment a progress bar
#                 accu += incre_update
#
#                 rounder_check = est_iters - accu
#
#                 if rounder_check < .005:
#                     pbar.close()
#
#                 pbar.update(incre_update)
#
#                 # instantiate a new model for each chunk
#                 m = Prophet(yearly_seasonality=False
#                             ,weekly_seasonality=False
#                             ,daily_seasonality=False
#                             ,uncertainty_samples=False)
#
#                 # return a data chunk of window_size on index idx
#                 chunk = chunks[idx]
#
#                 # set up index of next chunk in sequence
#                 next_idx = idx + 1
#
#                 # at then end of a seasonal_unit, break if index out of range
#                 if next_idx > len(chunks) - 1:
#                     break
#                 else:
#                     # otherwise, return the first value of next sequence as y target
#                     target = chunks[next_idx].iloc[0]
#
#                 # suppress pystan outputs for each model fit
#                 with suppress_stdout_stderr():
#                     m.fit(chunk)
#                 # extends the current sequence of time by n_prediction_units as future
#                 future = m.make_future_dataframe(periods=n_prediction_units
#                                                  , freq=prediction_frequency
#                                                  , include_history=False)
#                 # exclude history to avoid having to negative index
#                 # the 'yhat' column is created by the prophet API
#                 # return the last n results from future as yhat
#                 # n_forcast_units = -1 * n_prediction_units
#                 forecast = m.predict(future)[[ds_col, 'yhat']]#.iloc[n_forcast_units:]
#
#                 # increment the chunk
#                 idx += 1
#
#                 # save targets y and forecast predictions yhat
#                 targets = targets.append(target)
#                 predictions = predictions.append(forecast)
#
#         # merge all results as a single dataframe
#         results = pd.merge(targets, predictions, left_on=ds_col, right_on=ds_col)
#
#         # drop day marker, not needed further
#         results.drop(columns=['day'], inplace=True)
#
#         return results

def assess_prophet_results(prophet_results):

    df = pd.concat([pd.concat(i) for i in prophet_results])
    df = df.reset_index(drop=True)

    ds_col = 'ds'
    # length of dataset for graphing later
    n = 'n=' + str(len(df))
    # export results to csv
    # df.to_csv('data/facebook_prophet_results.csv')

    # blank line for padding
    print()

    # compute the MAPE over the entire prediction set
    error = mean_absolute_percentage_error(df['y'].values, df['yhat'].values) * 100
    print('The MAPE for facebook prophet is', error)

    # plot the data
    fig, ax, = plt.subplots()
    ax.plot(df[ds_col], df['y'], color='red', label='Actual Price')
    ax.plot(df[ds_col], df['yhat']
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

#TODO: Finish refactoring default state for prediction

#     else:
#         if valid is not None:
#             n_prediction_units = len(valid)
#         # else, run training on a large range of data and predict the next
#         m = Prophet(changepoint_prior_scale=0.005, yearly_seasonality=False)
#         m.fit(train)  # df is a pandas.DataFrame with 'y' and 'ds' columns
#
#         # set future index period manually (hide for now)
#         future = m.make_future_dataframe(periods=n_prediction_units, freq=prediction_frequency)
#
#         # trading day parameters
#         trade_day_start = 9
#         trade_day_end = 16
#         weekend = 5
#
#         # apply forecast filter based on trade day parameters
#         # future = future[(future[ds_col].dt.hour >= trade_day_start) | (future[ds_col].dt.hour  <= trade_day_end)]
#         # future = future[(future[ds_col].dt.weekday < weekend)]
#
#         # set future dataframe forecast based on validation time index
#         future = valid[['t']].reset_index(drop=True)
#         future = future.rename(columns={'t':ds_col})
#
#         # return just the last n results which are the forecasts
#         n_forcast_units = -1 * n_prediction_units
#         forecast = m.predict(future)[[ds_col, 'yhat']].iloc[n_forcast_units:]
#
#         # return results in a single dataframe
#         results = pd.merge(valid, forecast, left_on=time_col, right_on=ds_col)
#
#         preds = results['yhat'].values
#         targets = results[data_col].values
#
#         # compute MAPE error for comparison to models
#         error = mean_absolute_error(targets, preds) * 100
#
#         print('The MAPE for facebook prophet is', error)
#
#         # plot data
#         fig, ax, = plt.subplots()
#         ax.plot(results[time_col], results[data_col], color='red', label='Actual Price')
#         ax.plot(results[time_col], results['yhat']
#                 , color='blue'
#                 , marker='o'
#                 , markersize=3
#                 , linestyle='dashed'
#                 , linewidth=1
#                 , label='Predicted Price'
#                 )
#         ax.set_title('Amazon Stock Price Prediction (Dev Data)\nWith Facebook Prophet ' + n)
#         plt.xlabel('Time')
#         plt.ylabel('Stock Price')
#         plt.legend()
#         fig.autofmt_xdate()
#         plt.show()
#
#         # plt.plot()
#         # fig1 = m.plot(forecast)
#         # plt.title('Amazon Stock Price Prediction (Dev Data)\nWith Facebook Prophet ' + n)
#         # plt.tight_layout()
#         # fig1.show()
#
#         #fixme subplots and title
#         # plt.plot()
#         # plt.title('Amazon Stock Price Components (Dev Data)\nWith Facebook Prophet')
#         # fig2 = m.plot_components(forecast)
#         # plt.tight_layout()
#         # fig2.show()
#         # print('done')
#
# else:
#     print("Set up Facebook Prophet, did not run model")


