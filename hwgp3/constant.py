"""
constant.py

For homework in Prof Chen's course only.

@created: 2021-05-09
@author: ZOU Qi-xiang (qxzou.20@saif.sjtu.edu.cn)
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'marketing_data.csv')
"""Download from https://www.kaggle.com/jackdaoud/marketing-data"""
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


###############
# Headers
###############
H_INCOME = 'Income'
H_YEAR_BIRTH = 'Year_Birth'
H_DT_CUSTOMER = 'Dt_Customer'
H_EDUCATION = 'Education'
H_ID = 'ID'
H_PRED = 'Pred'
H_RECENCY = 'Recency'
HT_ACCEPTED_CMP = 'AcceptedCmp%d'

PRED_DF_COLS = [H_ID, H_PRED]
