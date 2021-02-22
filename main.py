### this section represents a pipeline to return news data as a df of sentiment scores
from utilities.get_data import get_news_dummies, get_stock_dummies
from utilities.clean_data import trading_days
from utilities.sentiment_data import *
from utilities.run_arima import train_arima
from utilities.run_prophet import train_prophet
from utilities.prep_stock_data import split_stock_data, scale_stock_data
from model.lstm_approach_1 import train_model_1

# run get pipelines for news and stock data
news_df = get_news_dummies()
stock_df = get_stock_dummies()

# transform to minute-by-minute sentiment score and stock price
df = trading_days(news_df, stock_df)

# train arima on stock data only
train_arima(timeseries=df[['t','c']], time_col='t', run_model=False, window_size=15)

# train prophet on stock data only
train_prophet(df[['t', 'c']], time_col='t', data_col='c')
# split data into train, validation, and testing
train, valid, test = split_stock_data(df=df[['t','c']], time_col='t')
# train lstm on unscaled data
# train_model_1(train, run_model=False, is_scaled=False)

# scale stock data
train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                   , valid=valid
                                                                   , test=test
                                                                   )
# train lstm on scaled data
# preds = train_model_1(train_scaled, run_model=False, is_scaled=True)

###

# split on data having closing price 'c' and sentiment score 'compound'
train, valid, test = split_stock_data(df=df, time_col='t')
# train_model_1(train, valid, run_model=True, is_scaled=False)

train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                   , valid=valid
                                                                   , test=test
                                                                   )
#TODO: configure data loader and train model to handle sentiment features 'compound'
# train on combined sentiment and stock data
train_model_1(train_scaled, valid_scaled, run_model=True, is_scaled=True)