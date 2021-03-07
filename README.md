# stock_sa

This project is about Sentiement Analysis for Stock Prediction by @justinhchae @knowledgewithin for COMP SCI 496, [Advanced Deep Learning](https://www.mccormick.northwestern.edu/artificial-intelligence/curriculum/descriptions/msai-449.html) .

[MSAI Northwestern](https://www.mccormick.northwestern.edu/artificial-intelligence/), Class of 2021.

## Source

Based on the paper ["Stock Price Prediction Using News Sentiment Analysis"](https://ieeexplore.ieee.org/document/8848203) by Saloni Mohan, Sahitya Mullapudi, Sudheer Sammeta, Parag Vijayvergia and David C. Anastasiu.

##

![results](https://github.com/justinhchae/stocks/blob/main/app_scatter.png)

## For Instructor Evaluation

1. The proprietary class data is ignored by Git, as a result, to run this program on class data, please follow this setup.

2. Unzip stock and news data to the data folder in accordance with the following structure:

```
stocks|
      |/data|
            |/class_data|
                        |/historical_price| # to contain all unzipped CSV files
                        |news.json # the provided news data
       
```

3. Follow Steps in *Getting Started*


4. Launch the app and select mode to *class_data*

A Screen shot of the folder structure:

![path](https://github.com/justinhchae/stocks/blob/main/data_cap.png)

## Get Started

1. Clone this repo
```terminal
git clone https://github.com/justinhchae/stocks
```

2. Create a new conda environment from the environment.yml file
```terminal
conda env create -f environment.yml
```

3. From terminal, run app
```terminal
streamlit run app.py
```

4. From app, make experiement selections. 

![application](https://github.com/justinhchae/stocks/blob/main/app_cap.png)

## Brief

1. Overview: Predict a stock price given its prior prices and sentiment from news articles about that stock.

2. Run forecasting experiements with ARIMA: train on a sequence of stock prices and predict the next in sequence.

3. Run forecasting experiements with Facebook Prophet: train on a sequence of stock prices and predict the next in sequence.

4. Run forecasting experiements with LSTM Neural Netowrk: train on a sequence of stock prices and predict the next in sequence.

5. Run forecasting experiements with LSTM Neural Netowrk: train on a sequence of stock prices with sentiment scores for that stock and predict the next in sequence.

## Data

The development data is based on a sample of dummy data news articles for Amazon (price history and news articles).

* We use VADER compound score as a single value for sentiment score of a given news article.

* We use closing price of a stock in a minute of a trading day.

* We create a dataframe of time 't', sentiment score 'compound', and price 'c'. The data is resampled on a two-day rolling average to produce a pair of scores for each minute of the trading day.

* News timesstamps that originate in UTC are converted to US/Eastern.

* Stock timestamps are assumed to be US/Eastern.

## Data Chunker

* Concept: Combine sentiment scores and prices per minute into a single dataframe, then parse it into chunks. 

* Each chunk of data is a 15-minute block of data that starts at 9 am each trading day and ends at 1600

* See [utilities/data_chunker.py](https://github.com/justinhchae/stocks/blob/main/utilities/data_chunker.py)

## ARIMA Methodology

* Train model on a period, forecast the period that immediately follows; consume chunked data

* Example: train on data from index 0 to index 14 and predict the value at the 15th index position.

* Data is sequenced at index 0 (equal to 9 am on the first trading day in dataset) and continues until the last in sequence.

* The model in each prediction window is a new model, only predicting the next sequence.

## Facebook Prophet Methodology

* Train model on a period, forecast the period that immediately follows; consume chunked data

* Example: train on data for from 09:00 to 09:14 and predict the value at 09:15.

* Unlike ARIMA, Prophet is time-aware so train and predict occur within the context of time as an index.

* The model in each prediction window is a new model, only predicting the next sequence.

## LSTM PyTorch Methodology

* Parse data for each trading day into windows of size 15 and increment step size by 1; ; consume batched data from PyTorch DataLoader objects

* Example: train on data for from index 0 to 14 and predict the value at index 15, then increment window size from [1:15] and predict index at 16, and so on.

* Unlike ARIMA and Prophet, the LSTM Model is used to predict sequences after the training window.

## Relevant Documentation

* [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

* [Facebook Prophet](https://facebook.github.io/prophet/)

* [ARIMA](https://www.statsmodels.org/stable/index.html)

* [VADER](https://pypi.org/project/vaderSentiment/)

## Key Resources

* [ARIMA Implementation](https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-an-arima-model-2e3b3080bd70)

* [Facebook Prophet Implementation](https://medium.com/spikelab/forecasting-multiples-time-series-using-prophet-in-parallel-2515abd1a245), [Supressing Prophet Output](https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python), and [enabling multi-processing](https://medium.com/spikelab/forecasting-multiples-time-series-using-prophet-in-parallel-2515abd1a245)

* [Chunking DataFrame](https://yaoyao.codes/pandas/2018/01/23/pandas-split-a-dataframe-into-chunks)

* [Enable multiprocessing with LSTM](https://pytorch.org/docs/stable/notes/multiprocessing.html)
