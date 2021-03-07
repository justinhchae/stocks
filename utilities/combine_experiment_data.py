import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np

def combine_news_stock(stock_df
                       , news_df
                       , ticker
                       , time_col='t'
                       , data_col='c'
                       , sentiment_col='compound'
                       , frequency='D'
                       , n_window_units=90
                       ):

    # aggregate news and stock data to synch for forecasting
    stock_aggregator = 'last'
    news_aggregator = 'mean'

    stock_df = stock_df.groupby(
        [pd.Grouper(key=time_col, freq=frequency)]).agg(stock_aggregator).dropna().reset_index()

    # resample data to average sentiment score per day (per paper specifications)
    news_df = news_df.groupby(
        [pd.Grouper(key=time_col, freq=frequency)]).agg(news_aggregator).dropna().reset_index()

    # merge on common frequency time period, left on stocks
    df = pd.merge(stock_df, news_df, how='left', left_on=time_col, right_on=time_col)

    # get variance of sentiment data to understand more
    # sentiment_variance = np.var(df[sentiment_col].values)


    # resample to fill in missing values with spline
    df['resampled_compound'] = df[sentiment_col].interpolate(method='spline', order=4)
    # fill na values with resampled points
    df[sentiment_col] = df[sentiment_col].fillna(df['resampled_compound'])

    # drop records not having enough data to resample and fill by year
    time_filter = df.groupby([pd.Grouper(key=time_col, freq='Y')]).agg('count').reset_index()
    time_filter = time_filter[(time_filter[sentiment_col] == 0)]
    min_date = max(time_filter[time_col])

    # filter dataframe having both sentiemnet scores and prices
    df = df[df[time_col] >= min_date].copy()
    df = df.reset_index(drop=True)

    # set index to datetime for resampling
    # df = df.set_index(time_col)
    # interpolate method to spline to fill in reasonable values for missing
    # df['resampled_compound'] = df[sentiment_col].interpolate(method='spline', order=4)
    #TODO: conduct testing to determine why spline or alt methods
    # df = df.reset_index()

    # scale stock prices
    scaler = StandardScaler()
    df[[data_col]] = scaler.fit_transform(df[[data_col]])

    df = df.dropna()
    df = df.drop(columns=['resampled_compound'])
    df = df.reset_index(drop=True)

    # get n data points for context and graph
    N = len(df)

    # plot the data
    fig, ax, = plt.subplots()
    ax.plot(df[time_col], df[data_col], color='red', label='Scaled Price')
    ax.plot(df[time_col], df[sentiment_col]
            , color='blue'
            , marker='o'
            , markersize=3
            , linestyle='dashed'
            , linewidth=1
            , label='Sentiment Score'
            , alpha=.2
            )

    ax.set_title(f'{ticker} Data Prep, n={N}\nScaled Prices and Resampled Sentiment Scores')
    plt.xlabel('Time')
    plt.legend()
    fig.autofmt_xdate()
    fig.savefig(f'figures/{ticker}_data_prep.png')
    # plt.show()

    return df#, sentiment_variance