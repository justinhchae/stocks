from src.runners.run_experiment import run_experiment

import pandas as pd

from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from functools import partial

import os
import logging
import time


def run_main(experiment_mode
             , tickers
             , debug_mode
             , data_folder
             , model_results_folder
             , historical_news_path=None
             , historical_stock_path=None
             , demo_run_mode=None
             , write_data=False
             , results_filename='results.csv'
             , dummy_news_filename = 'dummy_news.json'
             , dummy_stock_filename = 'price_minute.csv'
             ):
    """
    the run_main runner for this experiment
    :param experiment_mode: demo or class_data, default to demo mode
    :param tickers: a list of strings that represent stock tickers
    :param debug_mode: a boolean flag, default to False -> toggles multiprocessing
    :param demo_run_mode: a list of strings for configuring run_modes in demo mode
    :args -> runs a series of forecasting experiments
    :return: a dataframe of results
    """

    pd.set_option('display.max_columns', None)
    logging.basicConfig(level=logging.INFO)

    # demo_run_mode is a list that is toggled from the streamlit UI, ignore otherwise
    if demo_run_mode is not None:
        run_modes = demo_run_mode
    else:
        # EDIT this list to tailor run modes
        run_modes = ['arima', 'prophet', 'lstm1', 'lstm2']
        # run_modes = ['lstm1']

    # dummy data directories and paths
    dummy_news_path = os.sep.join([data_folder, 'dummies', dummy_news_filename])
    dummy_stock_path = os.sep.join([data_folder, 'dummies', dummy_stock_filename])


    # configure gpu if available
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    mp.set_start_method('spawn', force=True)

    # get cpu count for max available processes
    CPUs = mp.cpu_count()
    max_process = CPUs // 2

    # capture exp start time for performance evaluation
    exp_start = time.time()

    # assign list of tickers to tqdm object for status tracking
    exp_pbar = tqdm(tickers, desc='Experiments', leave=True)

    if not debug_mode and experiment_mode != 'demo':
        # by default run experiments in parallel for all tickers
        # configure a pool to run experiments
        exp_pool = mp.Pool(max_process)
        # pass function params with partial method, then call the partial during mp
        run_experiment_ = partial(run_experiment
                                  , experiment_mode=experiment_mode
                                  , device=device
                                  , CPUs=CPUs
                                  , dummy_news_path=dummy_news_path
                                  , dummy_stock_path=dummy_stock_path
                                  , historical_news_path=historical_news_path
                                  , historical_stock_path=historical_stock_path
                                  , model_results_folder=model_results_folder
                                  , run_modes=run_modes
                                  , write_data=write_data
                                  )
        # with pooling, iterate through tickers and evaluate data/models
        # sub processes ARE NOT pooled
        exp_results = list(exp_pool.imap(run_experiment_, exp_pbar))
        exp_pool.close()
        exp_pool.join()
    else:
        # outside of debug, run each forecasting thing in series but have each one run with multiprocessor

        exp_results = [run_experiment(experiment_mode=experiment_mode
                                      , device=device
                                      , CPUs=CPUs
                                      , ticker=i
                                      , dummy_news_path=dummy_news_path
                                      , dummy_stock_path=dummy_stock_path
                                      , historical_news_path=historical_news_path
                                      , historical_stock_path=historical_stock_path
                                      , model_results_folder=model_results_folder
                                      , run_modes=run_modes
                                      , write_data=write_data
                                      , progress_bar=exp_pbar
                                      ) for i in exp_pbar]

    # capture total clock time in experiment
    exp_end = time.time()
    run_time = exp_end - exp_start
    logging.info(f'Total Run Time {run_time}')
    if write_data:
        # consolidate exp outputs to a single dataframe from a list of dictionaries
        df = pd.concat([pd.DataFrame(i) for i in exp_results])
        # reset index and print dataframe head
        df = df.reset_index(drop=True)
        # export results to a csv for analysis
        results_output = os.sep.join([data_folder, results_filename])
        df.to_csv(results_output, index=False)

        return df
