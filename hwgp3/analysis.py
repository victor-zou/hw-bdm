import numpy as np
import pandas as pd
import typing as tp
import hwgp3.calc as calc


class GiniStatResult(tp.NamedTuple):
    M: int
    pi_df: pd.DataFrame
    """Col: l, pi_prime_l, pi_l"""
    gini_coef: float


class ResultAnalyser(object):

    def __init__(self, pred: np.ndarray, act_ret: np.ndarray):
        assert pred.size == act_ret.size
        self.pred = pred.reshape(-1)
        self.pred_rank = calc.rank(-self.pred)
        self.act_ret = act_ret.reshape(-1)
        self.tot_posi = int(np.sum(act_ret))
        self.pi = self.tot_posi / pred.size

    def get_top_decile_lift(self, pct_levels: tp.Iterable[int] = (5, 10, 20, 30, 50)) -> pd.DataFrame:
        # Of course, the following code is stupid: it should be O(N)
        # instead of O(N^2). But my finger is hurt, pardon me.
        return pd.DataFrame.from_records([
            self.get_pct_top_decile_lift(ipct) for ipct in pct_levels
        ])

    def get_gini_stat(self, M: int = 40) -> GiniStatResult:
        l = self.pred_rank // (self.pred_rank.size//M+1)
        df = pd.DataFrame(dict(l=l, act_ret=self.act_ret))
        gped = df['act_ret'].groupby(df['l']).agg(['count', 'sum'])
        gped = gped.sort_index().reset_index()
        gped['cum_cnt'] = gped['count'].cumsum()
        gped['cum_posi'] = gped['sum'].cumsum()
        gped['pi_l'] = gped['cum_cnt'] / self.pred.size
        gped['pi_prime_l'] = gped['cum_posi'] / self.tot_posi
        pi_df = gped[['l', 'pi_prime_l', 'pi_l']]
        gini_coef = (2/M) * (pi_df['pi_prime_l']-pi_df['pi_l']).sum()
        return GiniStatResult(M, pi_df, float(gini_coef))

    def get_pct_top_decile_lift(self, pct: int) -> dict:
        """"""
        num = (0.01 * pct * self.pred.size)
        flag = self.pred_rank < num
        cnt_posi = int(np.sum(self.act_ret[flag]))
        pi_p = cnt_posi / num
        lift = pi_p / self.pi
        return dict(pct=pct, pi_p=pi_p, lift=lift)


