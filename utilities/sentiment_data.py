import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def score_sentiment(df
                    , data_col
                    , date_col
                    , display_all_cols=True
                    , score_type='compound'
                    , frequency='D'
                    ):
    """
    :param df: a pandas dataframe
    :param col: a string (name of col to clean)
    :return: a dataframe with sentiment scores from vader

    The compound score is the overall sentiment where 0 is neutral,
    negative is arbitrarily worse and positive is arbitrarily better
    """
    if display_all_cols:
        pd.set_option('display.max_columns', None)

    df[score_type] = [analyzer.polarity_scores(v)[score_type] for v in df[data_col]]
    # df['neg'] = [analyzer.polarity_scores(v)['neg'] for v in df[col]]
    # df['neu'] = [analyzer.polarity_scores(v)['neu'] for v in df[col]]
    # df['pos'] = [analyzer.polarity_scores(v)['pos'] for v in df[col]]

    df = df[[date_col, score_type]].groupby(
        [pd.Grouper(key=date_col, freq=frequency)]).agg('mean').dropna().reset_index()

    return df

