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

def run_arima(chunked_data, price_col='y', n_prediction_units=1):
    warnings.simplefilter('ignore', ConvergenceWarning)

    results = []

    for x_i, y_i in chunked_data:
        m = ARIMA(x_i[price_col].values, order=(0, 1, 0))
        m_fit = m.fit()
        yhat = m_fit.forecast(steps=n_prediction_units)
        y_i['yhat'] = yhat
        results.append(y_i)

    return results










