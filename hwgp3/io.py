import pandas as pd 
import numpy as np
import datetime as dt
import typing as tp
import hwgp3.constant as cst


__all__ = [
    'load_df'
]

_g_std_func_map: tp.Dict[str, tp.Callable[[pd.DataFrame], None]] = dict()


def register_std_func(col: str):

    def _register_decorator(f):
        _g_std_func_map[col] = f
        return f

    return _register_decorator


@register_std_func(cst.H_INCOME)
def std_income(df: pd.DataFrame) -> None:
    """Change "$84,835.00 " to float; then get log1p"""
    df[cst.H_INCOME] = np.log1p(
        df[cst.H_INCOME].map(
            lambda s: float(s[1:-1].replace(',', '')) if isinstance(s, str) else np.nan
        )
    )
    df[cst.H_INCOME].fillna(df[cst.H_INCOME].mean(), inplace=True)


@register_std_func(cst.H_DT_CUSTOMER)
def std_dt_customer(df: pd.DataFrame) -> None:
    """Change 6/16/14 to date after 1/1/00"""
    def _parse_date(s: str) -> float:
        month, day, yr = (int(i) for i in s.split('/'))
        return float((dt.date(yr+2000, month, day)-dt.date(2000, 1, 1)).days)
    df[cst.H_DT_CUSTOMER] = df[cst.H_DT_CUSTOMER].map(_parse_date)


@register_std_func(cst.H_EDUCATION)
def std_education(df: pd.DataFrame) -> None:
    level_map = {
        "Basic": 0.,
        "Graduation": 1.,
        "Master": 2.,
        "2n Cycle": 1.8,
        "PhD": 3.
    }
    df[cst.H_EDUCATION] = df[cst.H_EDUCATION].map(level_map)


def apply_std(df: pd.DataFrame):
    for f in _g_std_func_map.values():
        f(df)


def load_df(path: str):
    df = pd.read_csv(cst.DATA_PATH)
    apply_std(df)
    return df


if __name__ == "__main__":
    df_ = pd.read_csv(cst.DATA_PATH).head()
    apply_std(df_)
    print(df_[list(_g_std_func_map.keys())])
