# Homework for Group 3

## Models
### 1. rfm with group average: rfm_gpmean
### 2. rfm with logit regression: rfm_logit

Last round result

                           Logit Regression Results
==============================================================================
Dep. Variable:                      y   No. Observations:                 2240
Model:                          Logit   Df Residuals:                     2236
Method:                           MLE   Df Model:                            3
Date:                Tue, 11 May 2021   Pseudo R-squ.:                     inf
Time:                        15:39:24   Log-Likelihood:                -168.00
converged:                       True   LL-Null:                        0.0000
Covariance Type:            nonrobust   LLR p-value:                     1.000
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -3.2474      0.119    -27.215      0.000      -3.481      -3.014
r             -0.0594      0.103     -0.574      0.566      -0.262       0.143
m              0.6000      0.119      5.048      0.000       0.367       0.833
f              0.0298      0.142      0.210      0.834      -0.249       0.309
==============================================================================

### 3. rfm with random forest: rfm_tree
Index(['r', 'm', 'f'], dtype='object')
[0.0946486  0.60019611 0.30515528]
Index(['r', 'm', 'f'], dtype='object')
[0.13636322 0.47501402 0.38862276]
Index(['r', 'm', 'f'], dtype='object')
[0.17568842 0.44757385 0.37673774]
Index(['r', 'm', 'f'], dtype='object')
[0.09505596 0.41968756 0.48525649]

### 4. seven factor random forest: f6_tree
Index(['r', 'm', 'f', 'edu', 'vf', 'kid', 'teen'], dtype='object')
[0.02490093 0.43143955 0.16934367 0.00751824 0.16782359 0.10804483
 0.09092919]
Index(['r', 'm', 'f', 'edu', 'vf', 'kid', 'teen'], dtype='object')
[0.01558617 0.42313562 0.22052691 0.00749109 0.13344519 0.10193835
 0.09787667]
Index(['r', 'm', 'f', 'edu', 'vf', 'kid', 'teen'], dtype='object')
[0.03948226 0.36171709 0.29322152 0.02547299 0.07490668 0.05990999
 0.14528948]
Index(['r', 'm', 'f', 'edu', 'vf', 'kid', 'teen'], dtype='object')
[0.02486366 0.49589902 0.27275981 0.01404626 0.04198914 0.1278939
 0.0225482 ]

## Requires
Python >= 3.6. Only depend on popular packages like numpy, pandas, statmodels, and sklearn.

