import json
import pandas as pd


def get_news_dummies():
    """
    :return: pandas df from dict of dummy news data
    """
    with open('data/dummies/dummy_news.json') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['pub_time'] = pd.to_datetime(df['pub_time'])
    return df

def get_stock_dummies():
    """
    :return: pandas df from dict of dummy news data
    """
    filename = 'data/dummies/price_minute.csv'
    df = pd.read_csv(filename, index_col=0)
    df['t'] = pd.to_datetime(df['t'])

    date_min = pd.to_datetime('2020-07-01')
    df = df[df['t'] > date_min].copy()

    return df