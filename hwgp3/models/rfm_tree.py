import numpy as np 
import pandas as pd
import re
import hwgp3.constant as cst
import hwgp3.model as md 
import hwgp3.calc as cc
np.random.seed(10)
from sklearn.metrics import roc_curve,accuracy_score,recall_score
from sklearn.ensemble import RandomForestRegressor

REG_M = re.compile('Mnt.*?')
REG_F = re.compile('Num.*?Purchases')


def df_zscore(df):
    return (df-df.mean()) / df.std()


def pred_is_accept(df, target):
    r = df[cst.H_RECENCY].values
    m = cc.sum_col(df, REG_M)
    f = cc.sum_col(df, REG_F)
    rfm_df = pd.DataFrame(dict(r=r, m=m, f=f))
    x = rfm_df.values
    tree = RandomForestRegressor()
    est = tree.fit(x, target)
    pred = est.predict(x)
    np.clip(pred, 0., 1., out=pred)
    return pd.DataFrame(
        {
            cst.H_ID: df[cst.H_ID],
            cst.H_PRED: pred
        }
    )


def get_model(df) -> md.Model:
    return md.Model('rfm_tree', df, pred_is_accept)


if __name__ == "__main__":
    from hwgp3.io import load_df
    model = get_model(load_df(cst.DATA_PATH))
    print(model.get_cmp_i_pred(2).head())
    print(model.get_cmp_i_pred(3).head())
    print(model.get_cmp_i_pred(4).head())
    print(model.get_cmp_i_pred(5).head())
