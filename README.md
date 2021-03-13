# stock_sa

This project is about Sentiement Analysis for Stock Prediction by @justinhchae @knowledgewithin for COMP SCI 496, [Advanced Deep Learning](https://www.mccormick.northwestern.edu/artificial-intelligence/curriculum/descriptions/msai-449.html) .

[MSAI Northwestern](https://www.mccormick.northwestern.edu/artificial-intelligence/), Class of 2021.

## Source

Based on the paper ["Stock Price Prediction Using News Sentiment Analysis"](https://ieeexplore.ieee.org/document/8848203) by Saloni Mohan, Sahitya Mullapudi, Sudheer Sammeta, Parag Vijayvergia and David C. Anastasiu.

## High-level Results

* See our [streamlit app](https://share.streamlit.io/justinhchae/stocks/presentation_only/app.py) for more details. 

* Based on preliminary analysis, all models tend to perform well when the variance of price and sentiment data is relatively lower. 

* In the scatter plot of variance and error, plots that are small in size, low, and left are better. 

[![results](https://github.com/justinhchae/stocks/blob/main/app_scatter.png)](https://share.streamlit.io/justinhchae/stocks/presentation_only/app.py)

## For Instructor Evaluation

1. The proprietary class data is ignored by Git, as a result, to run this program on class data, please follow this setup.

2. Unzip stock and news data to the data folder in accordance with the following structure:

```terminal
<project_root>
  /data
  /class_data
    /historical_price # to contain all unzipped CSV files
    /news.json # the provided news data
       
```

3. Follow Steps in *Getting Started*

4. Launch the app and set experiment mode to *class_data*

A Screen shot of the folder structure:

![path](https://github.com/justinhchae/stocks/blob/main/data_cap.png)

## Getting Started

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

![application](https://github.com/justinhchae/stocks/blob/main/images/app_cap.png)

## Brief

1. Overview: Predict a stock price given its prior prices and sentiment from news articles about that stock.

2. Run forecasting experiments with ARIMA: train on a sequence of stock prices and predict the next in sequence.

3. Run forecasting experiments with Facebook Prophet: train on a sequence of stock prices and predict the next in sequence.

4. Run forecasting experiments with LSTM Neural Netowrk: train on a sequence of stock prices and predict the next in sequence.

5. Run forecasting experiments with LSTM Neural Netowrk: train on a sequence of stock prices and predict the next in sequence based on the current sentiment score.

## Data

### Overall Data

* News timesstamps that originate in UTC are converted to US/Eastern.

* Stock timestamps are assumed to be US/Eastern.

```python
# example code for datetime conversions
def get_news_dummies(filepath
                  , date_col='pub_time'
                  , date_conversion='US/Eastern'):
    
    # after reading json data as data
    df = pd.DataFrame(data)
    df[date_col] = pd.to_datetime(df[date_col])

    if date_conversion:
        date_est =  date_col + '_est'
        df[date_est] = (df[date_col].dt.tz_convert(date_conversion))
        df[date_est] = df[date_est].dt.tz_localize(tz=None)
        df.drop(columns=date_col, inplace=True)
```

* We use VADER for sentiment score as a proxy for the polarity score in the source paper.

```python
# example code for sentiment score
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def score_sentiment(df
                  , data_col
                  , date_col
                  , score_type='compound'
                  ):

df[score_type] = [analyzer.polarity_scores(v)[score_type] for v in df[data_col]]
```
  
### Demo Data

* The demo data is based on a sample of dummy data news articles for Amazon (price history and news articles).

* We use closing price of a stock in a minute of a trading day wherein a trading day is defined as weekdays from 9 am to 4 pm US Eastern time.

* We create a dataframe of time 't', sentiment score 'compound', and price 'c'. To fill in missing data points, the data is resampled on a two-day rolling average to produce a pair of scores for each minute of the trading day.

```python
# example code for price resample
def get_stock_dummies(filepath
                      , time_col='t'
                      , data_col='c'
                      , window_minutes=2880
                      ):
    # ...
    df = df.set_index(time_col)

    df = df.resample('1min').fillna('nearest')
    df['c'] = df['c'].rolling(window_minutes).mean()
    # ... 
```

### Experiment Data (Class Data)

* Class data is based on historical prices and news data for approximately 81 tickers.

* Per the source paper methods, we resample the data to obtain a daily price at closing and a single sentiment score for a given stock. For stock prices, we return the last closing price for each day as the daily closing price. For news sentiment score, we take the average compound score per day.

```python
# example code for resample price to daily

#...
stock_df = stock_df.groupby(
        [pd.Grouper(key=time_col, freq=frequency)]).agg(stock_aggregator).dropna().reset_index()
#...
```

* There are large gaps of time periods wherein we have stock data but no news data. As a result, to fill in missing time periods without sentiment scores, engineer new features with spline to interpolate missing scores. We find that spline adequately fits a line representing a somewhat nuetral score for a given stock-setiment combination.

```python
# example code for resample with spline
df['resampled_compound'] = df[sentiment_col].interpolate(method='spline', order=4)
```

* For the experiement, we focus on replicating the concept of training with 4 days of data to predict the 5th day in the sequence for all variations of forecasting.

## Data Chunker and Data Sequencer

### Demo Concept

* Combine sentiment scores and prices per minute into a single dataframe, then parse it into chunks.

* Each chunk of data is a 15-minute block of data that starts at 9 am each trading day and ends at 1600

* See [utilities/data_chunker.py](https://github.com/justinhchae/stocks/blob/main/utilities/data_chunker.py)

### Experiment Concept

* We combine price and sentiment into a single data frame, then parse it into sequences.

* With time as an index, parse data into sliding sequences based on a window size of n and a step size of 1. The sliding sequnces are used to train, predict, and evaluate the ARIMA and Prophet methods. The sliding sequences are similar to the DataLoader objects we utilize to train, validate, and test the PyTorch LSTM models.

* Data sequencer for ARIMA and Prophet:
  
```python
# example code for sequence slider
    #...
    x, y = [], []

    N = len(df)

    if target_len is None:
      target_len = seq_len
    
    for i in range(N - seq_len):
      x_i = df[i: i + seq_len].reset_index(drop=True)
      y_i = df[i + seq_len: i + seq_len + target_len].reset_index(drop=True)

      x.append(x_i)
      y.append(y_i)

    chunked_data = list(zip(x,y))
    #...
```

* The equivialent sequence is provided in the LSTM2 model with the following DataLoader code example:

```python
    #...
    def __getitem__(self, index):
        x = self.data.iloc[index: index + self.window, 1:]
        y = self.data.iloc[index + self.window, self.yhat:self.yhat+1]

        # set sentiment score to that of the one corresponding to the target price
        if self.sentiment in self.data.columns:
            y_sentiment = self.data[self.sentiment].iloc[index + self.window]
            x[self.sentiment] = y_sentiment

        return torch.tensor(x.to_numpy(), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    #...
```

## ARIMA Methodology

* Train model on a period, forecast the period that immediately follows; consume chunked or sequened data.

* Example: train on data from index 0 to index 14 and predict the value at the 15th index position.

* The model in each prediction window is a new model, only predicting the next sequence.

* The ARIMA [module](https://github.com/justinhchae/stocks/blob/main/utilities/run_arima.py)

## Facebook Prophet Methodology

* Train model on a period, forecast the period that immediately follows; consume chunked data

* Example: train on data for from 09:00 to 09:14 and predict the value at 09:15.

* Unlike ARIMA, Prophet is time-aware so train and predict occur within the context of time as an index.

* The model in each prediction window is a new model, only predicting the next sequence.

* The PyTorch [module](https://github.com/justinhchae/stocks/blob/main/utilities/run_prophet.py)

## LSTM PyTorch Methodology

* Unlike ARIMA and Prophet, the LSTM model is used to predict sequences that are far outisde the training window.

* Parse data for each trading day into windows of size 15 and increment step size by 1 then consume batched data from PyTorch DataLoader objects

* In addition to sequence alone, a second LSTM model combines a sequence of past prices with the sentiment score of the sentiment score associated with the time index of the predicted price.

* Example: Train on data for from index 0 to 14 and predict the value at index 15, then increment window size from [1:15] and predict index at 16, and so on.

* The LSTM [model](https://github.com/justinhchae/stocks/blob/main/model/LSTM.py)

* The LSTM [training configuration](https://github.com/justinhchae/stocks/blob/main/model/lstm_approach_1.py)

## Early Stopping in LSTM Training

* We set default epochs to n=50 but set several early stopping criteria to enable learning while avoiding overfitting.

* Currently, stopping criteria is configured to evaluate after the completion of each epoch (we plan to implement mid-epoch stopping at a later date).

* Break training if the current validation loss is greater than the prior two losses.

* Break training if the variance of the last 3 validation losses are less than 0.000001 (a number that we feel is effectively close to zero).

* Allow model to continue trainin beyond set epoch number but break if the number of epochs more than double (this feature currently being debugged).

```python
# example code for early stopping in LSTM

        # from within the training loop in LSTM training
        if epoch > patience:
            curr_loss = valid_losses[-1]
            last_loss = valid_losses[-2]
            try:
                last_prior_loss = valid_losses[-3]
            except:
                last_prior_loss = 1000000

            # ave_loss = np.mean(valid_losses)
            last_n_losses = valid_losses[-2:]
            variance = np.var(last_n_losses)

            # if loss increases, break if the ave loss is below target loss threshold
            if curr_loss > last_loss and curr_loss > last_prior_loss:
                # stop training if the epoch loss increases
                stop_reason = 'loss started increasing'
                break

            elif variance < min_variance:
                # stop training if loss effectively stops changing
                stop_reason = 'loss stopped changing'
                break
```

## Program Architecture

* We leverage the PyTorch multiprocessing (MP) library to compute in parallel. 

* MP is enabled by default (debug = False) but it can be disabled by setting debug = True

* In demo mode, we evaluate a single stock and iterate through each of the types of models; within each model evaluation, we use MP to speed up the computations. 

* In class_data mode, we evaluate many stocks and apply MP to iterate through the entire training cycle for each stock in parallel.

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
