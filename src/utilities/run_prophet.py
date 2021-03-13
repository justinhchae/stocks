from fbprophet import Prophet
# https://facebook.github.io/prophet/docs/quick_start.html#python-api
import pandas as pd
import os

# https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
# https://medium.com/spikelab/forecasting-multiples-time-series-using-prophet-in-parallel-2515abd1a245


class suppress_stdout_stderr(object):
    '''
    # https://github.com/facebook/prophet/issues/223
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def run_prophet(chunked_data, seasonal_unit, prediction_frequency, n_prediction_units=1):
    # install gotcha: https://github.com/facebook/prophet/issues/775#issuecomment-449139135
    ds_col = 'ds'
    results = []

    # if forecasting sub-daily
    if seasonal_unit == 'day':
        # enumerate through chunked data pairs
        for x, y in chunked_data:
            # new model for each go around
            m = Prophet(uncertainty_samples=False)

            with suppress_stdout_stderr():
                m.fit(x)

            # set an index of times to predict for
            future = m.make_future_dataframe(periods=n_prediction_units
                                             , freq=prediction_frequency
                                             , include_history=False)
            # make predictions
            yhat = m.predict(future)[[ds_col, 'yhat']]
            # unify targets and predictions
            result = pd.merge(y, yhat, left_on=ds_col, right_on=ds_col).drop(columns=seasonal_unit)
            # capture results in a list
            results.append(result)
        # return a list of results
        return results

    elif seasonal_unit == 'week'or seasonal_unit =='sliding_sequence':

        x = chunked_data[0]
        y = chunked_data[1]

        m = Prophet(uncertainty_samples=False)

        with suppress_stdout_stderr():
            m.fit(x)

        future = m.make_future_dataframe(periods=n_prediction_units
                                         , freq=prediction_frequency
                                         , include_history=False)

        # this is a hack, just return the next n values, whatever they are
        # TODO: address prediction date and frequency if not predicting on the expected date
        yhat = m.predict(future)[['yhat']]
        y['yhat'] = yhat

        return y




