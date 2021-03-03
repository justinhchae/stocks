import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-an-arima-model-2e3b3080bd70

def setup_arima(train_data
                , price_col
                , time_col
                , **kwargs
                ):

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

def run_arima(chunked_data, seasonal_unit, price_col='y', n_prediction_units=1, prediction_frequency='1min'):
    # supress trivial warnings from ARIMA
    warnings.simplefilter('ignore', ConvergenceWarning)

    # initialize a list to hold results (a list of dataframes)
    results = []

    if seasonal_unit == 'day':
        # numerate through a list of chunked tuples, each having a pair of dataframes
        for idx, (x_i, y_i) in enumerate(chunked_data):
            # create ARIMA model based on x_i values
            m = ARIMA(x_i[price_col].values, order=(0, 1, 0))
            # fit the model
            m_fit = m.fit()
            # forecast for n_prediction_units
            yhat = m_fit.forecast(steps=n_prediction_units)

            # return a dataframe of targets and predictions of len targets
            y_i['yhat'] = yhat[:len(y_i)]

            # save results to a list and then return the list
            results.append(y_i)

        # return a list of dataframes
        return results

    elif seasonal_unit == 'week':
        # each element is a target predict pair
        x_i = chunked_data[0]
        y_i = chunked_data[1]
        # create ARIMA model based on x_i values
        m = ARIMA(x_i[price_col].values, order=(0, 1, 0))
        # fit the model
        m_fit = m.fit()
        # forecast for n_prediction_units
        yhat = m_fit.forecast(steps=n_prediction_units)
        # return a dataframe of targets and predictions of len targets
        y_i['yhat'] = yhat[:len(y_i)]
        # add the result to a list (should be a list of one, but putting in a list for consistency)

        # return a list of dataframes
        return y_i













