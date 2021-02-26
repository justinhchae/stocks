import json
import pandas as pd
from utilities.clean_data import cleaner
from utilities.sentiment_data import score_sentiment



def get_news_dummies(filepath
                     , date_col='pub_time'
                     , date_conversion='US/Eastern'):
    """
    :return: pandas df from dict of dummy news data
    """

    if filepath == 'Amazon':
        filepath = 'data/dummies/dummy_news.json'
    else:
        filepath = 'data/dummies/dummy_news.json'

    pd.set_option('display.max_columns', None)

    with open(filepath) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df[date_col] = pd.to_datetime(df[date_col])

    if date_conversion:
        date_est =  date_col + '_est'
        df[date_est] = (df[date_col].dt.tz_convert(date_conversion))
        df[date_est] = df[date_est].dt.tz_localize(tz=None)
        df.drop(columns=date_col, inplace=True)

    df = cleaner(df, 'text')

    df = score_sentiment(df, 'text', 'pub_time_est')

    return df

def get_stock_dummies(filepath
                      , date_col='t'
                      , data_col='c'
                      , window_minutes=2880
                      ):
    """
    :return: pandas df from dict of dummy news data
    """
    if filepath == 'Amazon':
        filepath = 'data/dummies/price_minute.csv'
    else:
        filepath = 'data/dummies/price_minute.csv'

    pd.set_option('display.max_columns', None)
    df = pd.read_csv(filepath, index_col=0)
    df[date_col] = pd.to_datetime(df[date_col])

    # hard coded to min date from news data
    date_min = pd.to_datetime('2020-10-06')
    df = df[df[date_col] > date_min].copy()
    df = df[[date_col, data_col]]

    df = df.set_index(date_col)

    df = df.resample('1min').fillna('nearest')
    df['c'] = df['c'].rolling(window_minutes).mean()
    df.reset_index(inplace=True)

    # print(df.tail())
    # df = df.resample('1D').last().dropna().reset_index()
    # start data on a monday
    date_min = pd.to_datetime('2020-10-11')
    df = df[df[date_col] > date_min].copy()
    df.reset_index(inplace=True, drop=True)

    return df


