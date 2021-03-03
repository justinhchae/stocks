import json
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

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
                      , time_col='t'
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
    df[time_col] = pd.to_datetime(df[time_col])

    # hard coded to min date from news data
    date_min = pd.to_datetime('2020-10-06')
    df = df[df[time_col] > date_min].copy()
    df = df[[time_col, data_col]]

    df = df.set_index(time_col)

    df = df.resample('1min').fillna('nearest')
    df['c'] = df['c'].rolling(window_minutes).mean()
    df.reset_index(inplace=True)

    # print(df.tail())
    # df = df.resample('1D').last().dropna().reset_index()
    # start data on a monday
    date_min = pd.to_datetime('2020-10-11')
    df = df[df[time_col] > date_min].copy()
    df.reset_index(inplace=True, drop=True)

    return df


def get_stock_tickers(filepath='data/class_data/news.json'):
    """
    :return: a list of stock tickers to analyze
    """

    # read/open json data
    with open(filepath) as f:
        # news data returned as dict, keyed by stock ticker
        data = json.load(f)

    return list(data.keys())


def get_news_real(ticker
                , filepath='data/class_data/news.json'
                , date_col='pub_time'
                , date_conversion='US/Eastern'
                , time_col='t'):
    """
    :return: pandas df from dict of real news data for a single stock
    """

    # enable view all cols
    pd.set_option('display.max_columns', None)

    # read/open json data
    #TODO: target file reader to only open the ticker's data
    with open(filepath) as f:
        # news data returned as dict, keyed by stock ticker
        data = json.load(f)

    # select just the stock ticker to run through pipeline
    df = pd.DataFrame(data[ticker])

    df[date_col] = pd.to_datetime(df[date_col])

    if date_conversion:
        date_est =  date_col + '_est'
        df[date_est] = (df[date_col].dt.tz_convert(date_conversion))
        df[date_est] = df[date_est].dt.tz_localize(tz=None)
        df.drop(columns=date_col, inplace=True)

    # run text cleaner on article narrative
    df = cleaner(df, 'text')

    # score article narrative and return only time and scores
    df = score_sentiment(df, 'text', 'pub_time_est', is_resample=False, date_min=None)

    # this returns a dataframe of time as provided by data
    df = df.sort_values(by=[time_col]).reset_index(drop=True)

    return df


def get_stock_real(ticker
                 , time_col='t'
                 , data_col='c'):
    """
    :return: pandas df from dict of real news data
    help from: https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
    """

    # enable view all cols
    pd.set_option('display.max_columns', None)
    path = r'data/class_data/historical_price'
    target_files = glob.glob(path + f"/{ticker}*.csv")

    li = []

    for filename in target_files:
        # looping in case there are multiple data points
        # expecting just one though (refactor later)
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    bad_column = 'Unnamed: 0'
    if  bad_column in df.columns:
        df = df.drop(columns=[bad_column])

    df[time_col] = pd.to_datetime(df[time_col])

    # return sorted time_col and data_col
    df = df.sort_values(by=[time_col]).reset_index(drop=True)[[time_col, data_col]]

    return df

