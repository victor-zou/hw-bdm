import numpy as np
import pandas as pd
import typing as tp


class ResultAnalyser(object):

    def __init__(self, pred: np.ndarray, act_ret: np.ndarray):
        assert pred.size == act_ret.size
        self.pred = pred.reshape(-1)
        self.pred_rank = np.argsort(-self.pred)
        self.act_ret = act_ret.reshape(-1)
        self.tot_posi = int(np.sum(act_ret))
        self.pi = self.tot_posi / pred.size

    def get_top_decile_lift(self, pct_levels: tp.Iterable[int] = (1, 2, 5, 10, 20)) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            self.get_pct_top_decile_lift(ipct) for ipct in pct_levels
        ])

    def get_pct_top_decile_lift(self, pct: int) -> dict:
        """"""
        num = (0.01 * pct * self.pred.size)
        flag = self.pred_rank < num
        cnt_posi = int(np.sum(self.act_ret[flag]))
        pi_p = cnt_posi / num
        lift = pi_p / self.pi
        return dict(pct=pct, pi_p=pi_p, lift=lift)
