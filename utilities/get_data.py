import json
import pandas as pd


def get_news_dummies(date_col='pub_time'):
    """
    :return: pandas df from dict of dummy news data
    """

    with open('data/dummies/dummy_news.json') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df[date_col] = pd.to_datetime(df[date_col])

    date_check = str(df[date_col].dtypes)

    if "UTC" in date_check:
        date_est =  date_col + '_est'
        df[date_est] = df[date_col].dt.tz_convert('US/Eastern')
        df.drop(columns=date_col, inplace=True)

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


