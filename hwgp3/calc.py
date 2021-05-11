import numpy as np 
import pandas as pd 

def sum_col(df: pd.DataFrame, reg) -> np.ndarray:
    """sum the cols in df that full match the reg.
    
    ..note::
        nan, type error, etc.. are not handled.
    """
    result = np.zeros((df.shape[0],), np.float64)
    for col in df.columns:
        if reg.fullmatch(col) is not None:
            result += df[col].values
    return result


def rank(x):
    return np.argsort(np.argsort(x))