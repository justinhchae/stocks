from fbprophet import Prophet
from utilities.prep_stock_data import split_stock_data
# https://facebook.github.io/prophet/docs/quick_start.html#python-api
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_prophet(df, time_col, data_col, run_model=True):

    train, valid, test = split_stock_data(df=df, time_col=time_col)

    ds_col = 'ds'
    y_col = 'y'

    key_map = {time_col: ds_col, data_col: y_col}

    df = train.rename(columns=key_map)

    n = 'n=' + str(len(df))

    m = Prophet(changepoint_prior_scale=0.005, yearly_seasonality=False)

    if run_model:
        m.fit(df)  # df is a pandas.DataFrame with 'y' and 'ds' columns

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
