from src.application.config import *

def narrative_introduction():
    st.header('Concept')
    st.write('1. Train various models on stock data with a focus on timeseries and sentiment data from news about the stock.')
    st.write('2. Baseline forecasting with ARIMA, Facebook Prophet, and LSTM.')
    st.write('3. Combine prices with sentiment scores and train a second LSTM having sentiment scores.')
    st.write('4. Compare results with MAPE (Mean Absolute Percentage Error)')

def narrative_conclusion():
    st.header('Conclusions')
    st.write('1. Feature engineering was the most complex part of this project.')
    st.write('2. Standard models like ARIMA and Prophet do very well within the training range.')
    st.write('3. LSTMs do okay, both with price and with sentiment, but underperform overall.')
    st.write('4. High levels of variance in sentiment and in price have major impacts on model accuracy.')
    st.write('5. Combination of a forecasting model with a reinforcement learning agent may be interesting.')
    st.write('6. Despite lower accuracy for LSTMs, they still tracked the direction of movement and only missed the magnitude.')
    st.write(' ')