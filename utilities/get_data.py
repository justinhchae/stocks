import json
import pandas as pd
from utilities.clean_data import cleaner
from utilities.sentiment_data import score_sentiment

def get_news_dummies(date_col='pub_time', date_conversion='US/Eastern'):
    """
    :return: pandas df from dict of dummy news data
    """

    with open('data/dummies/dummy_news.json') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df[date_col] = pd.to_datetime(df[date_col])

    if date_conversion:
        date_est =  date_col + '_est'
        df[date_est] = df[date_col].dt.tz_convert(None)
        df.drop(columns=date_col, inplace=True)

    df = cleaner(df, 'text')

    df = score_sentiment(df, 'text', 'pub_time_est')

    return df

def get_stock_dummies(date_col='t', data_col='c'):
    """
    :return: pandas df from dict of dummy news data
    """
    filename = 'data/dummies/price_minute.csv'
    df = pd.read_csv(filename, index_col=0)
    df[date_col] = pd.to_datetime(df[date_col])

    # hard coded to min date from news data
    date_min = pd.to_datetime('2020-10-06')
    df = df[df[date_col] > date_min].copy()
    df = df[[date_col, data_col]]

    df = df.set_index(date_col)

    df = df.resample('1min').fillna('nearest')
    df['c'] = df['c'].rolling(2880).mean()
    df.reset_index(inplace=True)

    # print(df.tail())
    # df = df.resample('1D').last().dropna().reset_index()

    date_min = pd.to_datetime('2020-10-11')
    df = df[df[date_col] > date_min].copy()
    df.reset_index(inplace=True, drop=True)

    return df


