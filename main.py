### this section represents a pipeline to return news data as a df of sentiment scores
from utilities.get_data import *
from utilities.clean_data import *
from utilities.sentiment_data import *

df = get_dummies()
df = cleaner(df, 'title')
df = score_sentiment(df, 'title')

print(df)
###

#TODO get reddit and or twitter data for at least 1-2020 to 1-2021, more is better, if possible.

#TODO make cleaner module (for social media data)

#TODO make sentiment analysis module (for social media data)

#TODO prepare stock data for time series analysis

#TOD prepare nueral network for sentiment/stock price learning
