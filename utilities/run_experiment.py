from utilities.get_data import get_news_dummies, get_stock_dummies, get_news_real, get_stock_real, get_stock_tickers
from utilities.clean_data import trading_days
from utilities.run_arima import run_arima
from utilities.run_prophet import run_prophet
from utilities.data_chunker import chunk_data
from utilities.evaluate_model import assess_model
from utilities.prep_stock_data import split_stock_data, scale_stock_data
from model.lstm_approach_1 import train_model
from utilities.combine_experiment_data import combine_news_stock
from model.LSTM import Model
from tqdm import tqdm
import torch.multiprocessing as mp
from functools import partial


def run_experiment(ticker, experiment_mode, device, CPUs, run_modes):
    #TODO: data struct error when running baseline exps in series from arima, prohpet to lstms
    # problem: cannot run all exps in a loop for demo mode

    params = {}
    trouble = []
    experiment_results = []
    df = None
    if run_modes is None:
        run_modes = ['arima', 'prophet']

    if experiment_mode == 'demo':
        news_df = get_news_dummies(ticker)
        stock_df = get_stock_dummies(ticker)
        # transform to minute-by-minute sentiment score and stock price
        df = trading_days(news_df, stock_df)
        ## run data splits on just price data
        # split data into train, validation, and testing
        train, valid, test = split_stock_data(df=df[['t', 'c']], time_col='t')
        # scale stock data
        train_scaled_price, valid_scaled_price, test_scaled_price, scaler = scale_stock_data(train=train, valid=valid,
                                                                                             test=test)
        # set parameters unique to demonstration/dummy data
        ## run data splits with both price and sentiment data
        train, valid, test = split_stock_data(df=df, time_col='t')
        # scale stock data
        train_scaled_sentiment, valid_scaled_sentiment, test_scaled_sentiment, scaler = scale_stock_data(train=train,
                                                                                                         valid=valid,
                                                                                                         test=test)
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
            , 'seasonal_unit': 'sliding_sequence'
            , 'prediction_unit': '1min'
            , 'epochs': 6
            , 'n_layers': 1
            , 'learning_rate': 0.001
            , 'batch_size': 16
            , 'hidden_dim': 64
            , 'n_prediction_units': 15
            , 'device': device
            , 'max_processes': CPUs // 2
            , 'pin_memory': False
            , 'enable_mp': True # only run mp for demo mode for child processes
                  }

    elif experiment_mode == 'class_data':

        # override run_modes
        run_modes = ['arima', 'prophet', 'lstm1', 'lstm2']

        # wrap functions in try to work out issues in bad data
        try:
            news_df = get_news_real(ticker=ticker)
        except:
            trouble.append((ticker, "news_df"))
            # tqdm.write(f'Trouble with {ticker}, skipping to next.')
            pass
        try:
            stock_df = get_stock_real(ticker=ticker)
        except:
            trouble.append((ticker, "stock_df"))
            # tqdm.write(f'Trouble with {ticker}, skipping to next.')
            pass

        # consolidated data prep for training (scale, combine, filter)
        try:
            df = combine_news_stock(stock_df=stock_df, news_df=news_df, ticker=ticker)
        except:
            trouble.append((ticker, "combine_news_stock"))
            # tqdm.write(f'Trouble with {ticker}, skipping to next.')
            pass

        if df is not None:
            # split data on data that is already scaled
            try:
                train_scaled_price, valid_scaled_price, test_scaled_price = split_stock_data(df=df[['t', 'c']], time_col='t')
                train_scaled_sentiment, valid_scaled_sentiment, test_scaled_sentiment = split_stock_data(df=df, time_col='t')
                # set parameters unique to class data
                params = {'stock_name': ticker
                    , 'train_data': train_scaled_price
                    , 'valid_data': valid_scaled_price
                    , 'test_data': test_scaled_price
                    , 'train_data_sentiment': train_scaled_sentiment
                    , 'valid_data_sentiment': valid_scaled_sentiment
                    , 'test_data_sentiment': test_scaled_sentiment
                    , 'time_col': 't'
                    , 'price_col': 'c'
                    , 'sentiment_col': 'compound'
                    , 'run_model': True
                    , 'window_size': 4
                    , 'seasonal_unit': 'sliding_sequence' #options: 'day', 'week', 'sliding_sequence'
                    , 'prediction_unit': 'D'
                    , 'epochs': 20
                    , 'n_layers': 1
                    , 'learning_rate': 0.001
                    , 'batch_size': 16
                    , 'hidden_dim': 64
                    , 'n_prediction_units': 1
                    , 'device': device
                    , 'max_processes': CPUs // 2
                    , 'pin_memory': False
                    , 'enable_mp': False # running exps in mp from the top, disable sub mps
                          }
            except:
                pass
        else:
            result = {'ticker': ticker
                     , 'N': ' '
                     , 'MAPE': ' '
                     , 'date_start': ' '
                     , 'date_end': ' '
                     , 'model_type': ' '
                     , 'notes': f'error during data_prep'
                       }

            experiment_results.append(result)
            return experiment_results

    for run_mode in run_modes:
        # configure parameters for forecasting here
        params.update({'run_mode': run_mode})

        if params['run_mode'] == 'arima' or params['run_mode'] == 'prophet':
            # tqdm.write('\nForecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))
            ## baselines can be trained on all data, but for comparison to lstm, train predict on test set
            # df = pd.concat([params['train_data'], params['valid_data'], params['test_data']])

            # chunk data
            chunked_data = chunk_data(**params)
            desc = '{}-{}'.format(params['stock_name'], params['run_mode'])

            chunked_data_pbar = tqdm(chunked_data, desc=desc, position=0, leave=True)

            # the model_ object is a temporary object to be updated during mp with partial()
            model = run_arima if run_mode == 'arima' else run_prophet if run_mode == 'prophet' else None

            if params['enable_mp']:
                # tqdm.write('Pooling {}x Processes with Multiprocessor'.format(params['max_processes']))
                # pooling enabled
                p = mp.Pool(params['max_processes'])
                # pass function params with partial method, then call the partial during mp
                model_ = partial(model
                                 , n_prediction_units=params['n_prediction_units']
                                 , seasonal_unit=params['seasonal_unit']
                                 , prediction_frequency=params['prediction_unit'])
                try:
                    # with pooling, iterate through model and data
                    results = list(p.imap(model_, chunked_data_pbar))
                    p.close()
                    p.join()
                except:
                    trouble.append((ticker, run_mode))
                    # tqdm.write(f'Trouble with {ticker}, skipping to next.')
                    continue

            else:
                # tqdm.write('Forecasting Without Pooling')
                # list comprehension through the same model and data without pooling
                results = None
                result = {'ticker': ticker
                        , 'N': ' '
                        , 'MAPE': ' '
                        , 'date_start': ' '
                        , 'date_end': ' '
                        , 'model_type': ' '
                        , 'notes': run_mode}
                try:
                    results = [model(i
                                     , n_prediction_units=params['n_prediction_units']
                                     , seasonal_unit=params['seasonal_unit']
                                     , prediction_frequency=params['prediction_unit']) for i in chunked_data_pbar]
                    # assess model results with MAPE and visualize predict V target
                except:
                    result = {'ticker': ticker
                             , 'N': ' '
                             , 'MAPE': ' '
                             , 'date_start': ' '
                             , 'date_end': ' '
                             , 'model_type': ' '
                             , 'notes': f'error during {run_mode}'
                              }
                if results:
                    result = assess_model(results
                                          , model_type=run_mode
                                          , stock_name=params['stock_name']
                                          , seasonal_unit=params['seasonal_unit']
                                          )

            experiment_results.append(result)

        elif params['run_mode'] == 'lstm1' or params['run_mode'] == 'lstm2':

            # tqdm.write('Forecasting for {} with Approach run_mode: {}'.format(params['stock_name'], run_mode))
            # if a cuda is available
            if device=='cpu':
                params.update({'num_workers': 1, 'pin_memory': True})

            # lstm1 has one input feature, lstm2 has two input features
            n_features = 1 if run_mode == 'lstm1' else 2 if run_mode == 'lstm2' else None

            # if run mode == lstm2, then refactor data to have both sentiment and price
            if run_mode == 'lstm2':
                # lstm2 experiment params update
                params.update({'train_data': params['train_data_sentiment']
                             , 'valid_data': params['valid_data_sentiment']
                             , 'test_data': params['test_data_sentiment']
                               })
            # update params dictionary
            params.update({'n_features': n_features})
            params.update({'input_dim': params['n_features']
                         , 'seq_length': params['window_size']
                         , 'sequence_length': params['window_size']
                           })
            # configure model with params
            model = Model(**params)
            # update the model in the dict (not sure if this is 100% necessary), check again later
            params.update({'model': model})
            # force no multiprocessing due to performance issues
            params.update({'enable_mp': False})

            if params['enable_mp']:
                params['model'].share_memory()

                processes = []
                # tqdm.write('Pooling {}x Processes with Multiprocessor'.format(params['max_processes']))
                # assign processes
                # pooling lstm on cpu is mothball for now due to performance issues, come back later to debug
                for rank in tqdm(range(params['max_processes'])):
                    # pool data for train_scaled to function train_model
                    p = mp.Process(target=train_model, kwargs=params)
                    p.start()
                    processes.append(p)
                # run computation and then close processes
                for p in tqdm(processes):
                    p.join()
            else:
                # tqdm.write('Forecasting Without Pooling')
                try:
                    # run train, validation, and test
                    result = train_model(**params)
                except:
                    result = {'ticker': ticker
                             , 'N': ' '
                             , 'MAPE': ' '
                             , 'date_start': ' '
                             , 'date_end': ' '
                             , 'model_type': ' '
                             , 'notes': f'error during {run_mode}'
                               }
                    pass

                experiment_results.append(result)

    return experiment_results