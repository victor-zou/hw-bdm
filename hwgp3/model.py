import os
import typing as tp
import numpy as np
import pandas as pd
import hwgp3.constant as cst

ModelImplFunT = tp.Callable[[pd.DataFrame, np.ndarray], pd.DataFrame]
""" Function to impl the prediction model.

Input:
    1) data df
    2) train result, average of the result before the round to pred

Output:
    pred result, 0-1. The threthold is unspecified. 
    Column is :const:`cst.PRED_DF_COLS`.

"""

class Model(object):

    def __init__(self, name: str, in_df: pd.DataFrame, impl_fun: ModelImplFunT):
        self.name = name
        self.out_dir = os.path.join(cst.OUTPUT_DIR, name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.in_df = in_df
        self.impl_fun = impl_fun
        self.__ram_cache = list()  # type: tp.List[int, tp.Optional[pd.DataFrame]]

    def get_cmp_i_pred(self, i: int) -> pd.DataFrame:
        """Calc or load from cache the pred result of i'th cmp
        
        Assume the data weight between rounds are equal.
        """
        assert i > 1
        cache_id = i-2
        if 

    