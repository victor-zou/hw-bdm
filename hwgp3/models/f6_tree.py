import numpy as np
import re
import hwgp3.constant as cst
import hwgp3.model as md 
import hwgp3.calc as cc
np.random.seed(10)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REG_M = re.compile('Mnt.*?')
REG_F = re.compile('Num.*?Purchases')


def df_zscore(df):
    return (df-df.mean()) / df.std()


# f7
def pred_is_accept(df, target):
    data = dict(r=df[cst.H_RECENCY].values)
    data['m'] = cc.sum_col(df, REG_M)
    data['f'] = cc.sum_col(df, REG_F)
    data['inc'] = df[cst.H_INCOME].values
    data['yr'] = df['Year_Birth']
    data['edu'] = df[cst.H_EDUCATION].values
    data['vf'] = df['NumWebVisitsMonth'].values
    data['child'] = df['Kidhome'].values+df['Teenhome'].values
    rfm_df = pd.DataFrame(data)
    x = rfm_df.values
    tree = RandomForestClassifier(
        max_samples=0.8, oob_score=True, max_depth=2, 
        min_samples_split=15, min_samples_leaf=10, 
        min_impurity_decrease=0.0001)
    est = tree.fit(x, target.astype(bool))
    pred = tree.predict_proba(x)[:,1].copy()
    print(rfm_df.columns)
    print(tree.feature_importances_)
    np.clip(pred, 0., 1., out=pred)
    return pd.DataFrame(
        {
            cst.H_ID: df[cst.H_ID],
            cst.H_PRED: pred
        }
    )


def get_model(df) -> md.Model:
    return md.Model('f6_tree', df, pred_is_accept)


if __name__ == "__main__":
    from hwgp3.io import load_df
    model = get_model(load_df(cst.DATA_PATH))
    print(model.get_cmp_i_pred(2).head())
    print(model.get_cmp_i_pred(3).head())
    print(model.get_cmp_i_pred(4).head())
    print(model.get_cmp_i_pred(5).head())
