from application.config import *


def lstm_math():
    # source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    st.markdown('LSTM Math [from PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)')
    st.latex(r'''i_t=σ(W_{ii}x_t+b_{ii}+W_{hi}h_{t−1}+b_{hi})''')
    st.latex(r'''f_t=σ(W_{if}x_t+b_{if}+W_{hf}h_{t−1}+b_{hf})''')
    st.latex(r'''g_t=tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t−1}+b_{hg})''')
    st.latex(r'''o_t=σ(W_{io}x_t+b_{io}+W_{ho}h_{t−1}+b_{ho})''')
    st.latex(r'''c_t=f_t⊙c_{t−1}+i_t⊙g_t''')
    st.latex(r'''h_t=o_t⊙tanh(c_t)''')

def arima_math():
    # source: https://people.duke.edu/~rnau/411arim.htm
    st.markdown('ARIMA Math [from Duke](https://people.duke.edu/~rnau/411arim.htm)')
    st.latex('yhat_t = μ + Y_{t-1}')
    st.write('ARIMA stands for Autoregressive Integrated Moving Average')
    st.write('Or in other words, a regression fit with conditional sum of square or MLE.')
    st.write('We use parameters of (0,1,0) which is known as random walk.')


def prophet_math():
    # source: https://peerj.com/preprints/3190.pdf
    st.markdown('Facebook Prophet Math [from Facebook](https://peerj.com/preprints/3190.pdf)')
    st.latex('y(t) = g(t) + s(t) + h(t) + e(t)')
    st.write('Or in other words, a regression fit that accounts for trend(g), seasonality (s), holidays (h), and error (e).')
