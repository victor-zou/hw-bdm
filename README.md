# Homework for Group 3

## Models
### 1. rfm with group average: rfm_gpmean
### 2. rfm with logit regression: rfm_logit

Last round result
                 coef    std err          z      P>|z|      [0.025      0.975]

const         -3.2474      0.119    -27.215      0.000      -3.481      -3.014
r             -0.0594      0.103     -0.574      0.566      -0.262       0.143
m              0.6000      0.119      5.048      0.000       0.367       0.833
f              0.0298      0.142      0.210      0.834      -0.249       0.309


### 3. rfm with random forest: rfm_tree
Index(['r', 'm', 'f'], dtype='object')
[0.0946486  0.60019611 0.30515528]
Index(['r', 'm', 'f'], dtype='object')
[0.13636322 0.47501402 0.38862276]
Index(['r', 'm', 'f'], dtype='object')
[0.17568842 0.44757385 0.37673774]
Index(['r', 'm', 'f'], dtype='object')
[0.09505596 0.41968756 0.48525649]

### 4. eight factor random forest: f8_tree
Index(['r', 'm', 'f', 'inc', 'yr', 'edu', 'vf', 'child'], dtype='object')
[0.01348723 0.34449994 0.09046977 0.28865613 0.01628517 0.00869951
 0.08001782 0.15788443]
Index(['r', 'm', 'f', 'inc', 'yr', 'edu', 'vf', 'child'], dtype='object')
[0.01250737 0.29825681 0.11651353 0.34731074 0.01535743 0.00328625
 0.07719467 0.1295732 ]
Index(['r', 'm', 'f', 'inc', 'yr', 'edu', 'vf', 'child'], dtype='object')
[0.0260904  0.31956531 0.12986956 0.33822537 0.02920092 0.00866598
 0.04993307 0.09844938]
Index(['r', 'm', 'f', 'inc', 'yr', 'edu', 'vf', 'child'], dtype='object')
[0.01779839 0.35813637 0.20127974 0.25168858 0.03264069 0.00517402
 0.02622572 0.10705648]

## Requires
Python >= 3.6. Only depend on popular packages like numpy, pandas, statmodels, and sklearn.

