import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import matplotlib.pyplot as plt


def score_sentiment(df
                    , data_col
                    , date_col
                    , score_type='compound'
                    , window_minutes=2880
                    ):
    """
    :param df: a pandas dataframe
    :param col: a string (name of col to clean)
    :return: a dataframe with sentiment scores from vader

    fake data: resample to produce a score for each minute of the dataframe

    The compound score is the overall sentiment where 0 is neutral,
    negative is arbitrarily worse and positive is arbitrarily better
    """

    df[score_type] = [analyzer.polarity_scores(v)[score_type] for v in df[data_col]]
    # df['neg'] = [analyzer.polarity_scores(v)['neg'] for v in df[col]]
    # df['neu'] = [analyzer.polarity_scores(v)['neu'] for v in df[col]]
    # df['pos'] = [analyzer.polarity_scores(v)['pos'] for v in df[col]]

    df = df[[date_col, score_type]]
    df = df.set_index(date_col)
    df = df.resample('1min').fillna('nearest')
    df[score_type] = df[score_type].rolling(window_minutes).mean()

    df.reset_index(inplace=True)

    # this aggs score by day
    # df = df[[date_col, score_type]].groupby(
    #     [pd.Grouper(key=date_col, freq='D')]).agg('mean').dropna().reset_index()

    # start data on a monday
    date_min = pd.to_datetime('2020-10-11')
    df = df[df[date_col] > date_min].copy()
    df.reset_index(inplace=True, drop=True)

    df.rename(columns={date_col:'t'}, inplace=True)

    return df

