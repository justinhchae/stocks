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
import gc
import time

from utilities.run_experiment import run_experiment

if __name__ == '__main__':
    # uncomment one of two types of experiment modes
    # experiment_mode = 'class_data'
    experiment_mode = 'demo'

    # if debug mode is True, script will run normally without multiprocessing
    # set to False to run without multiprocessing, this is helpful for debugging things in serial
    debug_mode = False

    # initialize empty structs
    tickers_historical = None

    try:
        # to run a single ticker, slice the get stock tickers function which returns a list
        tickers_historical = get_stock_tickers()
        # debugging
    except:
        pass

    # set up tickers iterable object
    tickers = ['Amazon'] if experiment_mode == 'demo' else tickers_historical

    # configure gpu if available
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    mp.set_start_method('spawn')

    # get cpu count for max available processes
    CPUs = mp.cpu_count()
    max_process = CPUs // 2

    # print what's happening for warm fuzzy feels
    n_tickers = len(tickers)
    print(f'Experimenting with {n_tickers} tickers:')

    # capture exp start time for performance evaluation
    exp_start = time.time()

    # assign list of tickers to tqdm object for status tracking
    exp_pbar = tqdm(tickers, desc='Experiment Framework', position=0, leave=True)

    if not debug_mode and experiment_mode != 'demo':
        # by default run experiments in parallel for all tickers
        # configure a pool to run experiments
        exp_pool = mp.Pool(max_process)
        # pass function params with partial method, then call the partial during mp
        run_experiment_ = partial(run_experiment
                                , experiment_mode=experiment_mode
                                , device=device
                                , CPUs=CPUs
                                , run_modes=None)
        # with pooling, iterate through tickers and evaluate data/models
        # sub processes ARE NOT pooled
        exp_results = list(exp_pool.imap(run_experiment_, exp_pbar))
        exp_pool.close()
        exp_pool.join()
    else:
        # in debug or demo mode, run the experiment in series
        # sub processes ARE pooled
        # run_modes = ['lstm1', 'lstm2']
        run_modes = ['arima', 'prophet']
        exp_results = [run_experiment(experiment_mode=experiment_mode
                                      , device=device
                                      , CPUs=CPUs
                                      , ticker=i
                                      , run_modes=run_modes) for i in exp_pbar]

    # capture total clock time in experiment
    exp_end = time.time()
    run_time = exp_end - exp_start
    print(f'Total Run Time {run_time}')
    # consolidate exp outputs to a single dataframe from a list of dictionaries
    df = pd.concat([pd.DataFrame(i) for i in exp_results])
    # reset index and print dataframe head
    df = df.reset_index(drop=True)
    print(df.head())
    # export results to a csv for analysis
    df.to_csv('data/results.csv')
