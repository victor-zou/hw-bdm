import numpy as np 
import pandas as pd
import re
import hwgp3.constant as cst
import hwgp3.model as md 
import hwgp3.calc as cc

N_GP = 5
REG_M = re.compile('Mnt.*?')
REG_F = re.compile('Num.*?Purchases')

def attr_to_gp(attr):
    rank = cc.rank(attr)
    gp = rank // (int(attr.size) // N_GP + 1)
    return gp


def get_gp_id(*attrs):
    gps = [attr_to_gp(attr) for attr in attrs]
    gp_id = np.zeros_like(gps[0])
    for igp in gps:
        gp_id *= N_GP
        gp_id += igp
    return gp_id


def pred_is_accept(df, target):
    r = df[cst.H_RECENCY].values
    m = cc.sum_col(df, REG_M)
    f = cc.sum_col(df, REG_F)
    gp = get_gp_id(r, m, f)
    gp_df = pd.DataFrame(dict(gp=gp, target=target))
    gp_dict = gp_df['target'].groupby(gp_df['gp']).mean().to_dict()
    pred = gp_df['gp'].map(gp_dict)
    return pd.DataFrame(
        {
            cst.H_ID: df[cst.H_ID],
            cst.H_PRED: pred
        }
    )


def get_model(df) -> md.Model:
    return md.Model('rfm_gpmean', df, pred_is_accept)


if __name__ == "__main__":
    from hwgp3.io import load_df
    model = get_model(load_df(cst.DATA_PATH))
    print(model.get_cmp_i_pred(2).head())
    print(model.get_cmp_i_pred(3).head())
    print(model.get_cmp_i_pred(4).head())
    print(model.get_cmp_i_pred(5).head())
