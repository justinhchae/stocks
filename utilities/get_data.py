import json
import pandas as pd


def get_news_dummies():
    """
    :return: pandas df from dict of dummy news data
    """
    with open('data/dummies/dummy_news.json') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    return df

def get_stock_dummies():
    """
    :return: pandas df from dict of dummy news data
    """
    filename = 'data/dummies/price_minute.csv'
    df = pd.read_csv(filename, index_col=0)


    return df