import json
import pandas as pd


def get_dummies():
    """
    :return: pandas df from dict of dummy news data
    """
    with open('data/dummies/dummy_news.json') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    return df