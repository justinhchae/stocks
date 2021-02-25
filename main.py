### this section represents a pipeline to return news data as a df of sentiment scores
from utilities.get_data import get_news_dummies, get_stock_dummies
from utilities.clean_data import trading_days
from utilities.sentiment_data import score_sentiment
from utilities.run_arima import train_arima
from utilities.run_prophet import setup_prophet, run_prophet, assess_prophet_results
from utilities.prep_stock_data import split_stock_data, scale_stock_data
from model.lstm_approach_1 import train_model_1
from model.LSTM import Model
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# get count of max available CPUs, minus 2
CPUs = cpu_count() - 2


if __name__ == '__main__':
    # run get pipelines for news and stock data
    news_df = get_news_dummies()
    stock_df = get_stock_dummies()

    # transform to minute-by-minute sentiment score and stock price
    df = trading_days(news_df, stock_df)

    # split data into train, validation, and testing
    train, valid, test = split_stock_data(df=df[['t','c']], time_col='t')
    # train lstm on unscaled data
    # train_model_1(train, run_model=False, is_scaled=False)

    # scale stock data
    train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                       , valid=valid
                                                                       , test=test
                                                                       )
    # START HERE uncomment the line you want to run
    run_mode = 'arima'
    # run_mode = 'prophet'
    # run_mode = 'lstm1'
    # run_mode = 'lstm2'

    if run_mode == 'arima':
        print('Training Approach run_mode:', run_mode)
        # train arima on stock data only
        train_arima(timeseries=test_scaled
                    , validation_data=valid
                    , time_col='t'
                    , window_size=15
                    )

    elif run_mode == 'prophet':
        print('Training Approach run_mode:', run_mode)
        # train prophet on stock data only
        prophet_data = setup_prophet(test_scaled
                      , time_col='t'
                      , data_col='c'
                      )
        # pooling enabled.
        p = Pool(CPUs)
        prophet_results = list(tqdm(p.imap(run_prophet, prophet_data)))
        p.close()
        p.join()
        assess_prophet_results(prophet_results)

    elif run_mode == 'lstm1':
        print('Training Approach run_mode:', run_mode)
        # train lstm on stock data only
        model = Model(num_layers=1, input_dim=1, seq_length=14)
        preds = train_model_1(train_scaled, valid_scaled, test_scaled, model, epochs=2, run_model=True, is_scaled=True, sequence_length=14)

    elif run_mode == 'lstm2':
        print('Training Approach run_mode:', run_mode)
        # split on data having closing price 'c' and sentiment score 'compound'
        model = Model(num_layers=1, input_dim=2, seq_length=14)
        train, valid, test = split_stock_data(df=df, time_col='t')
        # train_model_1(train, valid, run_model=True, is_scaled=False)

        train_scaled, valid_scaled, test_scaled, scaler = scale_stock_data(train=train
                                                                           , valid=valid
                                                                           , test=test
                                                                           )
        preds = train_model_1(train_scaled, valid_scaled, test_scaled, model, epochs=20, run_model=True, is_scaled=True, sequence_length=14)
