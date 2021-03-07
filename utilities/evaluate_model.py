import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tqdm import tqdm

def assess_model(data, model_type, stock_name, seasonal_unit, time_col='t', pred_col='yhat', tgt_col='y'):
    # initialize empty object for df
    df = None
    # set a ds col
    ds_col = 'ds'

    if seasonal_unit == 'day':
        df = pd.concat([pd.concat(i) for i in data])
        df = df.reset_index(drop=True)

    elif seasonal_unit == 'week' or seasonal_unit == 'sliding_sequence':
        df = pd.concat(data)
        df = df.reset_index(drop=True)

    # length of dataset for graphing later
    n = len(df)

    # compute the MAPE over the entire prediction set
    # error = mean_absolute_percentage_error(df['y'].values, df['yhat'].values) * 100
    error = mean_absolute_percentage_error(df[tgt_col].values, df[pred_col].values) * 100
    # tqdm.write(f'MAPE score for {stock_name} on {model_type} model: {error}\n')

    # plot the data
    fig, ax, = plt.subplots()
    ax.plot(df[ds_col], df[tgt_col], color='red', label='Target Price')
    ax.plot(df[ds_col], df[pred_col]
            , color='blue'
            , marker='o'
            , markersize=3
            , linestyle='dashed'
            , linewidth=1
            , label='Predicted Price'
            )
    ax.set_title('{} Stock Price Prediction\nWith {}, n={}, MAPE: {:.4f}'.format(stock_name,model_type,n,error))
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend(loc='upper left')
    fig.autofmt_xdate()
    fig.savefig(f'figures/{stock_name}_{model_type}_results.png')
    # plt.show()

    results = {'ticker':stock_name
            , 'N':n
            , 'MAPE': error
            , 'date_start':min(df[ds_col])
            , 'date_end':max(df[ds_col])
            , 'model_type':model_type
            , 'notes': ' '
            , 'n_epochs': None
               }
    df['model_type'] = model_type
    df.to_csv(f'data/model_results/{stock_name}_{model_type}_results.csv', index=False)

    return results