import os
import pandas as pd
import hwgp3.constant as cst
import hwgp3.model as md
import hwgp3.analysis as anal
import hwgp3.io as io


def main_routine(model: md.Model):
    ginis = dict()
    for i in range(2, 6):
        i_pred = model.get_cmp_i_pred(i)
        act_ret = model.in_df[cst.HT_ACCEPTED_CMP % i].values
        stater = anal.ResultAnalyser(i_pred[cst.H_PRED].values, act_ret)
        pct_df = stater.get_top_decile_lift()
        pct_df.to_csv(os.path.join(model.out_dir, f'pct_{i}.csv'), index=False)
        gini = stater.get_gini_stat()
        ginis[i] = gini.gini_coef
        gini.pi_df.to_csv(os.path.join(model.out_dir, f'gini_pi_{i}.csv'), index=False)

    pd.Series(ginis).to_csv(os.path.join(model.out_dir, f'gini_coef.csv'))


if __name__ == '__main__':
    df = io.load_df(cst.DATA_PATH)
    model = md.get_model('rfm_logit', df)
    main_routine(model)
