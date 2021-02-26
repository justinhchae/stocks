import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

def assess_model(data, model_type, stock_name, time_col='t', pred_col='yhat', tgt_col='y'):

    df = pd.concat([pd.concat(i) for i in data])
    df = df.reset_index(drop=True)

    ds_col = 'ds'
    # length of dataset for graphing later
    n = 'n=' + str(len(df))
    # export results to csv
    # df.to_csv('data/facebook_prophet_results.csv')

    # blank line for padding
    print()

    # compute the MAPE over the entire prediction set
    # error = mean_absolute_percentage_error(df['y'].values, df['yhat'].values) * 100
    error = mean_absolute_percentage_error(df[tgt_col].values, df[pred_col].values) * 100
    print(f'MAPE score for {stock_name} on {model_type} model: {error}')

    # plot the data
    fig, ax, = plt.subplots()
    ax.plot(df[ds_col], df[tgt_col], color='red', label='Actual Price')
    ax.plot(df[ds_col], df[pred_col]
            , color='blue'
            , marker='o'
            , markersize=3
            , linestyle='dashed'
            , linewidth=1
            , label='Predicted Price'
            )
    ax.set_title(f'{stock_name} Stock Price Prediction\nWith {model_type} ' + n)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    fig.autofmt_xdate()
    plt.show()