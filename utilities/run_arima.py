# from utilities.stata_models import *
from utilities.prep_stock_data import split_stock_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm
import numpy as np

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning



# https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-an-arima-model-2e3b3080bd70



def setup_arima(train_data
                , valid_data
                , test_data
                , price_col
                , time_col
                , run_model=True
                , window_size=15
                ):

    if run_model:

        plt.figure()
        lag_plot(train_data[price_col])
        plt.title('Amazon Stock (Dev Data) - Autocorrelation Plot')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(train_data[time_col], train_data[price_col])
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        ax.set_title('Amazon Stock (Dev Data) - Minute-by-Minute Closing Prices')
        fig.autofmt_xdate()
        plt.show()

    # train, valid, test = split_stock_data(df=df, time_col='t')
    x_i = train_data[price_col].values
    x_t = train_data[time_col].values

    v_i = valid_data[price_col].values
    v_t = valid_data[time_col].values

    t_i = test_data[price_col].values
    t_t = test_data[time_col].values

    N = len(x_i)

    x_i = [x_i[i:i + window_size] for i in range(0, N, window_size)]
    x_t = [x_t[i:i + window_size] for i in range(0, N, window_size)]

    arima_data = list(zip(x_i, x_t))

    return arima_data

def run_arima(data):
    warnings.simplefilter('ignore', ConvergenceWarning)

    x_i = data[0]
    x_t = data[1]

    # for every t-m:t-1 minutes, predict the t-th minute (next closing price)
    model_data = x_i[:-1]
    model = ARIMA(model_data, order=(0, 1, 0))

    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    target = x_i[-1]
    pred_time = x_t[-1]

    results = {'yhat': yhat,
               'y' : target,
               't' : pred_time}

    return results

def assess_arima(data, time_col='t', pred_col='yhat', tgt_col='y'):

    df = pd.DataFrame(data)
    df = df.sort_values(time_col).reset_index()
    N = 'n=' + str(len(df))

    y_true = df[tgt_col].values
    y_pred = df[pred_col].values
    error = mean_absolute_percentage_error(y_true, y_pred) * 100
    print('The MAPE error is', error)

    fig, ax = plt.subplots()
    ax.plot(df[time_col], df[pred_col]
            , color='red'
            , label='Actual Price')
    ax.plot(df[time_col], df[tgt_col]
            , color='blue'
            , marker='o'
            , markersize=3
            , linestyle='dashed'
            , linewidth=1
            , label='Predicted Price'
            )
    ax.set_title('Amazon Stock Price Prediction (Dev Data)\nWith STATA ARIMA Model ' + N)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    fig.autofmt_xdate()
    plt.show()


#TODO: Clean this UP!!!
#     else:
#         print("Run ARIMA on Entire Training Set - This will Take A While")
#
#         for time_point in tqdm(range(N_train_observations)):
#             model = ARIMA(history, order=(4,1,0))
#             model_fit = model.fit()
#             output = model_fit.forecast()
#             yhat = output[0]
#             model_predictions.append(yhat)
#             true_test_value = valid_data[time_point]
#             history.append(true_test_value)
#
#         error = mean_absolute_percentage_error(valid_data[:len(model_predictions)], model_predictions)
#         print('Testing Mean Squared Error is {}'.format(error))
#
#         fig, ax = plt.subplots()
#         ax.plot(valid_data['t'], valid_data['c'], color='red', label='Actual Price')
#         ax.plot(valid_data['t'][:len(model_predictions)], model_predictions, color='blue', marker='o', linestyle='dashed', label='Predicted Price')
#         ax.set_title('Amazon Stock Price Prediction (Dev Data)\nWith STATA ARIMA Model')
#         plt.xlabel('Time')
#         plt.ylabel('Stock Price')
#         plt.legend()
#         fig.autofmt_xdate()
#         plt.show()
#
#         results_dict = {}
#         results_dict['predictions'] = model_predictions
#         results_dict['targets'] = valid_data[:len(model_predictions)]
#         # df = pd.DataFrame.from_dict(results_dict)
#         # df.to_csv('data/arima_results.csv')
#
# else:
#     print('Ran ARIMA module, did not train')









