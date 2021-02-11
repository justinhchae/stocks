import texthero as hero
from texthero import preprocessing as pp

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