from utilities.get_data import get_news_dummies, get_stock_dummies, get_news_real, get_stock_real, get_stock_tickers
from utilities.clean_data import trading_days
from utilities.run_arima import run_arima
from utilities.run_prophet import run_prophet
from utilities.data_chunker import chunk_data
from utilities.evaluate_model import assess_model
from utilities.prep_stock_data import split_stock_data, scale_stock_data
from model.lstm_approach_1 import train_model, test_model, plot_losses
from utilities.combine_experiment_data import combine_news_stock
from model.LSTM import Model
from tqdm import tqdm
import torch
# from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
from functools import partial
import pandas as pd

if __name__ == '__main__':
    ## make two options to run the program, one for experiment mode and one run modes

    # uncomment one of two types of exp modes
    # experiment_mode = 'class_data'
    experiment_mode = 'demo'

    # uncomment one of two types of run modes
    # if class_data, these values are overridden (fix this later)
    run_modes = ['arima', 'prophet']
    # run_modes = ['lstm1', 'lstm2']

    # initialize empty structs
    train_scaled = None
    valid_scaled = None
    test_scaled = None
    tickers_historical = None
    params = {}
    experiment_results = []
    trouble = []

    try:
        # tickers_historical = get_stock_tickers()
        tickers_historical = ['AVGO']
    except:
        pass

    tickers = ['Amazon'] if experiment_mode == 'demo' else tickers_historical

    # configure gpu if available
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    mp.set_start_method('spawn')

    # get cpu count for processes
    CPUs = mp.cpu_count()

    n_tickers = len(tickers)
    tqdm.write(f'Experimenting with {n_tickers} tickers:')
    #TODO: handle cases like "GOOGL" or "GOOG", ignore for now
    for ticker in tickers:
        tqdm.write(ticker)

    for ticker in tickers:

        if experiment_mode == 'demo':
            news_df = get_news_dummies(ticker)
            stock_df = get_stock_dummies(ticker)
            # transform to minute-by-minute sentiment score and stock price
            df = trading_days(news_df, stock_df)
            ## run data splits on just price data
            # split data into train, validation, and testing
            train, valid, test = split_stock_data(df=df[['t', 'c']], time_col='t')
            # scale stock data
            train_scaled_price, valid_scaled_price, test_scaled_price, scaler = scale_stock_data(train=train, valid=valid, test=test)
            # set parameters unique to demonstration/dummy data
            ## run data splits with both price and sentiment data
            train, valid, test = split_stock_data(df=df, time_col='t')
            # scale stock data
            train_scaled_sentiment, valid_scaled_sentiment, test_scaled_sentiment, scaler = scale_stock_data(train=train, valid=valid, test=test)
            # set parameters unique to demonstration/dummy data
            params = {'stock_name': ticker
                    , 'train_data': train_scaled_price
                    , 'valid_data': valid_scaled_price
                    , 'test_data': test_scaled_price
                    , 'train_data_sentiment': train_scaled_sentiment
                    , 'valid_data_sentiment': valid_scaled_sentiment
                    , 'test_data_sentiment': test_scaled_sentiment
                    , 'time_col': 't'
                    , 'price_col': 'c'
                    , 'run_model': True
                    , 'window_size': 15
                    , 'seasonal_unit': 'day'
                    , 'prediction_unit': '1min'
                    , 'epochs': 6
                    , 'n_layers': 1
                    , 'learning_rate': 0.001
                    , 'batch_size': 16
                    , 'hidden_dim': 50
                    , 'n_prediction_units': 15
                    , 'device': device
                    , 'max_processes': CPUs // 2
                    , 'pin_memory': False
                    , 'enable_mp': True
                      }

        elif experiment_mode == 'class_data':

            # override run_modes
            run_modes = ['arima', 'prophet', 'lstm1', 'lstm2']
            # tickers = get_stock_tickers()
            # later, cycle through tickers, for now, work with first ticker in index
            news_df = get_news_real(ticker=ticker)
            stock_df = get_stock_real(ticker=ticker)
            # consolidated data prep for training (scale, combine, filter)
            try:
                df = combine_news_stock(stock_df=stock_df, news_df=news_df, ticker=ticker)
            except:
                trouble.append(ticker)
                tqdm.write(f'Trouble with {ticker}, skipping to next.')
                continue
            # split data on data that is already scaled
            train_scaled_price, valid_scaled_price, test_scaled_price = split_stock_data(df=df[['t', 'c']], time_col='t')
            train_scaled_sentiment, valid_scaled_sentiment, test_scaled_sentiment = split_stock_data(df=df, time_col='t')

            params = {'stock_name': ticker
                    , 'train_data': train_scaled_price
                    , 'valid_data': valid_scaled_price
                    , 'test_data': test_scaled_price
                    , 'train_data_sentiment': train_scaled_sentiment
                    , 'valid_data_sentiment': valid_scaled_sentiment
                    , 'test_data_sentiment': test_scaled_sentiment
                    , 'time_col': 't'
                    , 'price_col': 'c'
                    , 'sentiment_col':'compound'
                    , 'run_model': True
                    , 'window_size': 4
                    , 'seasonal_unit': 'week'
                    , 'prediction_unit': 'D'
                    , 'epochs': 50
                    , 'n_layers': 1
                    , 'learning_rate': 0.001
                    , 'batch_size': 32
                    , 'hidden_dim': 128
                    , 'n_prediction_units': 1
                    , 'device': device
                    , 'max_processes': CPUs // 2
                    , 'pin_memory': False
                    , 'enable_mp': True
                      }

        for run_mode in run_modes:
            # configure parameters for forecasting here
            params.update({'run_mode':run_mode})

            if params['run_mode'] == 'arima' or params['run_mode'] == 'prophet':
                tqdm.write('\nForecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))

                # for baselines, run train-predict pattern on all available data
                df = pd.concat([params['train_data'], params['valid_data'], params['test_data']])

                # chunk data
                chunked_data = chunk_data(**params)

                # the model_ object is a temporary object to be updated during mp with partial()
                model = run_arima if run_mode == 'arima' else run_prophet if run_mode == 'prophet' else None

                if params['enable_mp']:
                    tqdm.write('Pooling {}x Processes with Multiprocessor'.format(params['max_processes']))
                    # pooling enabled
                    p = mp.Pool(params['max_processes'])
                    # pass function params with partial method, then call the partial during mp
                    model_ = partial(model
                                     , n_prediction_units=params['n_prediction_units']
                                     , seasonal_unit=params['seasonal_unit']
                                     , prediction_frequency=params['prediction_unit'])
                    # with pooling, iterate through model and data
                    results = list(tqdm(p.imap(model_, chunked_data)))
                    p.close()
                    p.join()

                else:
                    tqdm.write('Forecasting Without Pooling')
                    # list comprehension through the same model and data without pooling
                    results = [model(i
                                     , n_prediction_units=params['n_prediction_units']
                                     , seasonal_unit=params['seasonal_unit']
                                     , prediction_frequency=params['prediction_unit'] ) for i in tqdm(chunked_data)]

                # assess model results with MAPE and visualize predict V target
                result = assess_model(results, model_type=run_mode, stock_name=params['stock_name'], seasonal_unit=params['seasonal_unit'])

                experiment_results.append(result)

            elif params['run_mode'] == 'lstm1' or params['run_mode'] =='lstm2':

                tqdm.write('Forecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))
                # if a cuda is available
                if is_cuda:
                    params.update({'num_workers': 1
                                   ,'pin_memory':True})

                n_features = 1 if run_mode == 'lstm1' else 2 if run_mode == 'lstm2' else None

                # if run mode == lstm2, then refactor data to have both sentiment and price
                if run_mode == 'lstm2':
                    params.update({'train_data': params['train_data_sentiment']
                                  , 'valid_data': params['valid_data_sentiment']
                                  , 'test_data': params['test_data_sentiment']
                                   })

                params.update({'n_features':n_features})

                params.update({'input_dim':params['n_features']
                              , 'seq_length':params['window_size']
                              , 'sequence_length':params['window_size']
                               })

                model = Model(**params)

                # update the model in the dict (not sure if this is 100% necessary), check again later
                params.update({'model': model})

                # force no multiprocessing due to performance issues
                params.update({'enable_mp': False})

                if params['enable_mp']:
                    params['model'].share_memory()

                    processes = []
                    tqdm.write('Pooling {}x Processes with Multiprocessor'.format(params['max_processes']))

                    # assign processes
                    for rank in tqdm(range(params['max_processes'])):
                        # pool data for train_scaled to function train_model
                        p = mp.Process(target=train_model, kwargs=params)
                        p.start()
                        processes.append(p)
                    # run computation and then close processes
                    for p in tqdm(processes):
                        p.join()
                else:
                    tqdm.write('Forecasting Without Pooling')
                    # run train, validation, and test
                    result = train_model(**params)

                    experiment_results.append(result)


    df = pd.DataFrame(experiment_results)

    df.to_csv('data/results.csv')

    tqdm.write('Had trouble with the following and did not run, do some troubleshooting.')

    for i in trouble:
        tqdm.write(i)