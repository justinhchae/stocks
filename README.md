# stock_sa

This project is about Sentiement Analysis for Stock Prediction by @justinhchae @knowledgewithin for COMP SCI 496, [Advanced Deep Learning](https://www.mccormick.northwestern.edu/artificial-intelligence/curriculum/descriptions/msai-449.html) .

[MSAI Northwestern](https://www.mccormick.northwestern.edu/artificial-intelligence/), Class of 2021.

## Source

Based on the paper ["Stock Price Prediction Using News Sentiment Analysis"](https://ieeexplore.ieee.org/document/8848203) by Saloni Mohan, Sahitya Mullapudi, Sudheer Sammeta, Parag Vijayvergia and David C. Anastasiu.

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

## ARIMA Methodology

* Parse data into window sizes of 15 and use the prior window to predict the start of the next window. 

* Example: train on data for from index 0 to index 14 and predict the value at the 15th index position.

* Data is sequenced at index 0 (equal to 9 am on the first trading day in dataset) and continues until the end.

* The model in each prediction window is a new model, only predicting the next sequence.

## Facebook Prophet Methodology

* Parse data for each trading day into time-index sizes of 15, starting at 9 am and end at 4 pm.

* Example: train on data for from 09:00 to 09:14 and predict the value at 09:15.

* Unlike ARIMA, Prophet is time-aware so train and predict occur within the context of time as an index.

* The model in each prediction window is a new model, only predicting the next sequence.

## LSTM PyTorch Methodology

* Parse data for each trading day into windows of size 15 and increment step size by 1

* Example: train on data for from index 0 to 14 and predict the value at index 15, then increment window size from [1:15] and predict index at 16, and so on.

* Unlike ARIMA and Prophet, the LSTM Model is used to predict sequences after the training window.


## Resources

* [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

* [Facebook Prophet](https://facebook.github.io/prophet/)

* [ARIMA](https://www.statsmodels.org/stable/index.html)
