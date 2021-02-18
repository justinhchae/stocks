### this section represents a pipeline to return news data as a df of sentiment scores
from utilities.get_data import *
from utilities.clean_data import *
from utilities.sentiment_data import *
from utilities.run_arima import *
from utilities.prep_stock_data import *
from model.lstm_approach_1 import *

news_df = get_news_dummies()
news_df = cleaner(news_df, 'text')
news_df = score_sentiment(news_df, 'text', 'pub_time_est')

stock_df = get_stock_dummies()
# uncomment below to run arima model
# train_arima(timeseries=stock_df, time_col='t')

# split data into train, validation, and testing
train, valid, test = split_stock_data(df=stock_df, time_col='t')

# train on unscaled data
train_model_1(train, run_model=False)

# scale stock data
train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                   , valid=valid
                                                                   , test=test
                                                                   )
# train on scaled data
train_model_1(train_scaled, run_model=False)

###

# minute-by-minute sentiment score and stock price
df = trading_days(news_df, stock_df)
train, valid, test = split_stock_data(df=df, time_col='t')
train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                   , valid=valid
                                                                   , test=test
                                                                   )
#TODO: configure data loader and train model to handle sentiment features 'compound'
# train on combined sentiment and stock data
train_model_1(train_scaled, run_model=False)