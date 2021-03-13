import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima(x, order, disp=False):
    """
    :params x: a list of ordered elements
    :params order: a 3-tuple of the stata arima order for seasonality
    :usage:
        x = [2.9 5.4 8.8 3.9]
        order = (2,1,2)

        where x[0] is the value at time T=0
    """
    model = SARIMAX(x, trend='c', order=order)
    model_fit = model.fit(disp=disp)
    return model_fit