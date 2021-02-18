import texthero as hero
from texthero import preprocessing as pp
import pandas as pd
from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay

def cleaner(df, col):
    """
    :param df: a pandas dataframe
    :param col: a string (name of col to clean)
    :return: a dataframe cleaned with text hero pipeline
    """
    pipeline = [
         pp.fillna
        , pp.remove_digits
        , pp.lowercase
        , pp.remove_punctuation
        , pp.remove_diacritics
        , pp.remove_stopwords
        , pp.remove_whitespace
        , pp.stem
                ]
    print('======================================== Done Cleaning')
    df[col] = hero.clean(df[col], pipeline=pipeline)
    print('======================================== DataFrame Head')

    return df

class GothamBusinessCalendar(AbstractHolidayCalendar):
    # https://towardsdatascience.com/holiday-calendars-with-pandas-9c01f1ee5fee
   rules = [
     Holiday('New Year', month=1, day=1, observance=sunday_to_monday),
     Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)]),
     Holiday('Labor Day', month=5, day=1, observance=sunday_to_monday),
     Holiday('July 4th', month=7, day=4, observance=nearest_workday),
     Holiday('Christmas', month=12, day=25, observance=nearest_workday)
   ]

Gotham_BD = CustomBusinessDay(calendar=GothamBusinessCalendar())



def trading_days(df1, df2, date_col = 't'):
    """
    return data from weekdays and between trading hours
    simulated
    """
    df = pd.merge(df1, df2, on=date_col)
    df.set_index('t', inplace=True)

    start_range = '2020-10-12'
    end_range = '2021-10-15'

    start_time = '09:00'
    end_time = '16:00'

    df = df[df.index.dayofweek < 5]
    df = df.between_time(start_time,end_time)

    df.reset_index(inplace=True)

    return df

