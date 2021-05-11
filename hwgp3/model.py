import os
import typing as tp
import numpy as np
import pandas as pd
import hwgp3.constant as cst

ModelImplFunT = tp.Callable[[pd.DataFrame, np.ndarray], pd.DataFrame]
""" Function to impl the prediction model.

As the prediction is for promoting to the unaccepted customers, the pred
model may not use wether accepted last time as input.

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
        self.__ram_cache = list()  # type: tp.List[tp.Optional[pd.DataFrame]]

    def get_cmp_i_pred(self, i: int) -> pd.DataFrame:
        """Calc or load from cache the pred result of i'th cmp
        
        Assume the data weights among rounds are equal.
        """
        assert i > 1
        cache_id = i-2
        try:
            assert self.__ram_cache[cache_id] is not None
            return self.__ram_cache[cache_id]
        except AssertionError:
            pass
        except IndexError:
            for _ in range(len(self.__ram_cache)-cache_id+1):
                self.__ram_cache.append(None)
        cache_f_path = os.path.join(self.out_dir, f'pred_{i}.csv')
        if os.path.isfile(cache_f_path):
            df = pd.read_csv(cache_f_path)
            self.__ram_cache[cache_id] = df
            return df
        target = self.in_df[cst.HT_ACCEPTED_CMP % 1].values.copy()
        for ii in range(2, i):
            target = np.maximum(target, self.in_df[cst.HT_ACCEPTED_CMP % ii].values)
        target = target * (1.0/(i-1))
        pred_df = self.impl_fun(self.in_df, target)[cst.PRED_DF_COLS]
        pred_df.to_csv(cache_f_path, index=False)
        self.__ram_cache[cache_id] = pred_df
        return pred_df


def get_model(name: str, df) -> Model:
    from importlib import import_module
    module = import_module(f'hwgp3.models.{name}')
    return module.get_model(df)
