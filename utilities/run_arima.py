from utilities.stata_models import *
from utilities.prep_stock_data import split_stock_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import numpy as np

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


# https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-an-arima-model-2e3b3080bd70



def train_arima(timeseries
                , time_col
                , date_min=None
                , date_max=None
                , run_model=False
                , window_size=15
                ):
    if date_min:
        date_min = pd.to_datetime(date_min)
        df = timeseries[timeseries[time_col] > date_min].copy()
    else:
        df = timeseries.copy()

    if run_model:

        plt.figure()
        lag_plot(df['c'])
        plt.title('Amazon Stock (Dev Data) - Autocorrelation Plot')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(df["t"], df["c"])
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        ax.set_title('Amazon Stock (Dev Data) - Minute-by-Minute Closing Prices')
        fig.autofmt_xdate()
        plt.show()

    train, valid, test = split_stock_data(df=df, time_col='t')
    train_data = train['c'].values
    train_time = train['t'].values

    valid_data = valid['c'].values
    valid_time = valid['c'].values
    test_data = test['c'].values

    history = [x for x in train_data]
    model_predictions = []
    N_train_observations = len(train_data)
    N_valid_observations = len(valid_data)
    N_test_observations = len(test_data)

    if run_model:

        # for every 15 minutes, predict the 16th minute (next closing price)
        if window_size:
            #TODO add validatoin and test
            print("Run ARIMA on Window Size", window_size)

            x_i = [train_data[i:i + window_size] for i in range(0, N_train_observations, window_size)]
            x_t = [train_time[i:i + window_size] for i in range(0, N_train_observations, window_size)]

            v_i = [valid_data[i:i + window_size] for i in range(0, N_valid_observations, window_size)]
            v_t = [valid_time[i:i + window_size] for i in range(0, N_valid_observations, window_size)]

            predictions = []
            targets = []
            pred_times = []
            losses = []

            for idx in tqdm(range(len(x_i))):

                model_data = x_i[idx][:-1]
                model = ARIMA(model_data, order=(0, 1, 0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                target = x_i[idx][-1]
                targets.append(target)
                pred_time = x_t[idx][-1]
                pred_times.append(pred_time)
                loss = mean_squared_error([target], [yhat])
                losses.append(loss)

            n = 'n=' + str(N_train_observations)

            # aggregate_mse_error = mean_squared_error(targets, predictions)
            mape = mean_absolute_error(targets, predictions) * 100
            print('The MAPE error is', mape)
            fig, ax = plt.subplots()
            ax.plot(pred_times, targets
                    , color='red'
                    , label='Actual Price')
            ax.plot(pred_times, predictions
                    , color='blue'
                    , marker='o'
                    , markersize=3
                    , linestyle='dashed'
                    , linewidth=1
                    , label='Predicted Price'
                    )
            ax.set_title('Amazon Stock Price Prediction (Dev Data)\nWith STATA ARIMA Model ' + n)
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            fig.autofmt_xdate()
            plt.show()

            fig, ax = plt.subplots()
            ax.plot(pred_times, losses, color='orange', label='MSE Loss')
            ax.set_title('Amazon Stock Price Prediction Losses (Dev Data)\nWith STATA ARIMA Model')
            plt.xlabel('Time')
            plt.ylabel('Loss')
            plt.legend()
            fig.autofmt_xdate()
            plt.show()

        else:
            print("Run ARIMA on Entire Training Set - This will Take A While")

            for time_point in tqdm(range(N_train_observations)):
                model = ARIMA(history, order=(4,1,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                model_predictions.append(yhat)
                true_test_value = test_data[time_point]
                history.append(true_test_value)

            MSE_error = mean_squared_error(test_data[:len(model_predictions)], model_predictions)
            print('Testing Mean Squared Error is {}'.format(MSE_error))

            fig, ax = plt.subplots()
            ax.plot(test['t'], test['c'], color='red', label='Actual Price')
            ax.plot(test['t'][:len(model_predictions)], model_predictions, color='blue', marker='o', linestyle='dashed', label='Predicted Price')
            ax.set_title('Amazon Stock Price Prediction (Dev Data)\nWith STATA ARIMA Model')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            fig.autofmt_xdate()
            plt.show()

            results_dict = {}
            results_dict['predictions'] = model_predictions
            results_dict['targets'] = test_data[:len(model_predictions)]
            df = pd.DataFrame.from_dict(results_dict)
            df.to_csv('data/arima_results.csv')

    else:
        print('Ran ARIMA module, did not train')









