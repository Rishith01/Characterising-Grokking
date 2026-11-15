# F2 final study: automatic summary

Runs analyzed: {'mnist': 64, 'modular': 160}; cells: {'mnist': 16, 'modular': 40}.
Cost = regret + lambda x (training after t_eps) / T. Settings are chosen on the calibration runs and scored on the test runs. CIs: 95% bootstrap over hyperparameter cells. p: two-sided sign test over cells.

## modular, seeds split (calibrate (0, 1), test (2, 3)), lambda = 0.1

| family | setting | test_cost | ci_lo | ci_hi | premature | never | d_horizon | d_horizon_lo | d_horizon_hi | cells_better_h | cells_worse_h | p_h | d_fixed | d_fixed_lo | d_fixed_hi | p_f | hindsight_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| horizon | horizon | 0.1284 | 0.0857 | 0.1876 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.0519 | 0.0027 | 0.1072 | 0 | 0.1284 |
| fixed_frac | fixed_frac c=0.5 | 0.0764 | 0.0485 | 0.1149 | 0.075 | 0 | -0.0519 | -0.1072 | -0.0027 | 37 | 3 | 0 | 0 | 0 | 0 | 1 | 0.0764 |
| bound | bound M lag=0.15 w=3 floor=0 est=ratio eps=0.1 gate=norm | 0.0554 | 0.0331 | 0.0847 | 0.0375 | 0.25 | -0.073 | -0.1105 | -0.0458 | 32 | 0 | 0 | -0.021 | -0.0604 | 0.0141 | 0.2682 | 0.0522 |
| rel_delta | rel_delta G lag=0.25 w=3 floor=1 eps=0.01 gate=none | 0.0743 | 0.047 | 0.111 | 0 | 0.5 | -0.0541 | -0.0853 | -0.0285 | 21 | 0 | 0 | -0.0022 | -0.0501 | 0.0427 | 1 | 0.0557 |
| const_delta | const_delta G c=0.02 w=3 floor=1 eps=0.003 gate=norm | 0.0734 | 0.0441 | 0.1126 | 0.0125 | 0.125 | -0.055 | -0.0949 | -0.0282 | 33 | 3 | 0 | -0.0031 | -0.0338 | 0.0245 | 0.6358 | 0.0484 |
| msd | msd M factor=2 thr=0.5 gate=none | 0.0861 | 0.0534 | 0.1292 | 0 | 0.575 | -0.0423 | -0.0868 | 0.0032 | 17 | 1 | 0.0001 | 0.0096 | -0.0392 | 0.059 | 0.1539 | 0.0861 |
| geweke | geweke G factor=2 thr=2 gate=norm | 0.1365 | 0.0829 | 0.2062 | 0 | 0.95 | 0.0081 | -0.0047 | 0.0299 | 2 | 1 | 1 | 0.06 | 0.0099 | 0.116 | 0 | 0.1176 |
| F2 best | const_delta G c=0.02 w=3 floor=1 eps=0.003 gate=norm | 0.0734 | 0.0441 | 0.1126 | 0.0125 | 0.125 | -0.055 | -0.0949 | -0.0282 | 33 | 3 | 0 | -0.0031 | -0.0338 | 0.0245 | 0.6358 | 0.0484 |
| oracle_cos | oracle_cos M c=0.995 | 0.0827 | 0.0382 | 0.1444 | 0.05 | 0 | -0.0457 | -0.0694 | -0.0222 | 35 | 3 | 0 | 0.0063 | -0.0316 | 0.059 | 0.6358 | 0.0439 |
| oracle_settle | oracle_settle M delta=0.5 | 0.0612 | 0.0295 | 0.1007 | 0.2 | 0 | -0.0672 | -0.125 | -0.0158 | 34 | 5 | 0 | -0.0153 | -0.0416 | 0.0154 | 0.0022 | 0.0606 |
| oracle_teps | oracle_teps | 0.0064 | 0.0052 | 0.0075 | 0 | 0 | -0.122 | -0.1805 | -0.0791 | 39 | 1 | 0 | -0.0701 | -0.1087 | -0.0426 | 0 | 0.0064 |


## modular, operation split (calibrate ('modular x+y',), test ('modular x*y', 'modular x/y')), lambda = 0.1

| family | setting | test_cost | ci_lo | ci_hi | premature | never | d_horizon | d_horizon_lo | d_horizon_hi | cells_better_h | cells_worse_h | p_h | d_fixed | d_fixed_lo | d_fixed_hi | p_f | hindsight_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| horizon | horizon | 0.138 | 0.0769 | 0.2282 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.0475 | -0.0122 | 0.1133 | 0.0042 | 0.138 |
| fixed_frac | fixed_frac c=0.4 | 0.0905 | 0.0469 | 0.1462 | 0.25 | 0 | -0.0475 | -0.1133 | 0.0122 | 14 | 2 | 0.0042 | 0 | 0 | 0 | 1 | 0.0462 |
| bound | bound W lag=0.15 w=3 floor=1 est=ratio eps=0.3 gate=norm | 0.0862 | 0.0449 | 0.1342 | 0.0469 | 0.25 | -0.0518 | -0.1091 | 0.0013 | 14 | 1 | 0.001 | -0.0043 | -0.057 | 0.0494 | 1 | 0.0591 |
| rel_delta | rel_delta G lag=0.25 w=3 floor=1 eps=0.01 gate=none | 0.0812 | 0.0424 | 0.1367 | 0 | 0.4844 | -0.0568 | -0.0918 | -0.0287 | 10 | 0 | 0.002 | -0.0093 | -0.0532 | 0.0309 | 0.4545 | 0.0547 |
| const_delta | const_delta G c=0.1 w=1 floor=0 eps=0.1 gate=none | 0.068 | 0.0301 | 0.1216 | 0 | 0.0938 | -0.07 | -0.1065 | -0.0425 | 16 | 0 | 0 | -0.0225 | -0.0647 | 0.0163 | 0.4545 | 0.0457 |
| msd | msd M factor=2 thr=0.5 gate=none | 0.0774 | 0.048 | 0.1239 | 0 | 0.6406 | -0.0606 | -0.112 | -0.018 | 9 | 0 | 0.0039 | -0.013 | -0.0678 | 0.0341 | 0.8036 | 0.0539 |
| geweke | geweke M factor=2 thr=5 gate=none | 0.2182 | 0.1068 | 0.349 | 0.1406 | 0.5625 | 0.0802 | -0.0561 | 0.2284 | 6 | 4 | 0.7539 | 0.1277 | -0.0024 | 0.2672 | 0.8036 | 0.1095 |
| F2 best | bound W lag=0.15 w=3 floor=1 est=ratio eps=0.3 gate=norm | 0.0862 | 0.0449 | 0.1342 | 0.0469 | 0.25 | -0.0518 | -0.1091 | 0.0013 | 14 | 1 | 0.001 | -0.0043 | -0.057 | 0.0494 | 1 | 0.0457 |
| oracle_cos | oracle_cos M c=0.99 | 0.0584 | 0.0205 | 0.1098 | 0 | 0 | -0.0797 | -0.1208 | -0.0501 | 16 | 0 | 0 | -0.0321 | -0.0729 | 0.0023 | 0.4545 | 0.0351 |
| oracle_settle | oracle_settle G delta=0.3 | 0.0699 | 0.0131 | 0.1483 | 0 | 0 | -0.0681 | -0.0877 | -0.0505 | 16 | 0 | 0 | -0.0206 | -0.0728 | 0.0324 | 0.4545 | 0.0368 |
| oracle_teps | oracle_teps | 0.0073 | 0.0057 | 0.0086 | 0 | 0 | -0.1307 | -0.221 | -0.0694 | 16 | 0 | 0 | -0.0832 | -0.1384 | -0.0398 | 0 | 0.0073 |


## mnist, seeds split (calibrate (0, 1), test (2, 3)), lambda = 0.1

| family | setting | test_cost | ci_lo | ci_hi | premature | never | d_horizon | d_horizon_lo | d_horizon_hi | cells_better_h | cells_worse_h | p_h | d_fixed | d_fixed_lo | d_fixed_hi | p_f | hindsight_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| horizon | horizon | 0.0963 | 0.0668 | 0.1375 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.0619 | 0.0277 | 0.1079 | 0.0213 | 0.0963 |
| fixed_frac | fixed_frac c=0.3 | 0.0344 | 0.027 | 0.0421 | 0.4062 | 0 | -0.0619 | -0.1079 | -0.0277 | 13 | 3 | 0.0213 | 0 | 0 | 0 | 1 | 0.0337 |
| bound | bound G lag=0.15 w=3 floor=1 est=fit eps=0.03 gate=norm | 0.0895 | 0.0601 | 0.1323 | 0 | 0.8438 | -0.0068 | -0.0181 | 0 | 3 | 0 | 0.25 | 0.0552 | 0.0208 | 0.1001 | 0.0768 | 0.0879 |
| rel_delta | rel_delta W lag=0.15 w=3 floor=0 eps=0.003 gate=none | 0.1037 | 0.0744 | 0.1358 | 0.0938 | 0.9062 | 0.0074 | -0.013 | 0.0352 | 1 | 1 | 1 | 0.0693 | 0.0372 | 0.1039 | 0.0042 | 0.0918 |
| const_delta | const_delta W c=0.1 w=3 floor=1 eps=0.3 gate=none | 0.0727 | 0.0493 | 0.0974 | 0.375 | 0.125 | -0.0236 | -0.0569 | 0.0105 | 11 | 3 | 0.0574 | 0.0383 | 0.0186 | 0.0623 | 0.0213 | 0.0727 |
| msd | msd G factor=4 thr=0.5 gate=none | 0.0702 | 0.037 | 0.1163 | 0.1875 | 0.6875 | -0.0261 | -0.0471 | -0.0088 | 5 | 0 | 0.0625 | 0.0358 | 0.0001 | 0.0856 | 0.8036 | 0.0702 |
| geweke | geweke W factor=4 thr=20 gate=none | 0.0807 | 0.0488 | 0.1258 | 0.0312 | 0.8438 | -0.0156 | -0.0354 | 0 | 3 | 0 | 0.25 | 0.0464 | 0.0102 | 0.0949 | 0.2101 | 0.0806 |
| F2 best | msd G factor=4 thr=0.5 gate=none | 0.0702 | 0.037 | 0.1163 | 0.1875 | 0.6875 | -0.0261 | -0.0471 | -0.0088 | 5 | 0 | 0.0625 | 0.0358 | 0.0001 | 0.0856 | 0.8036 | 0.0702 |
| oracle_cos | oracle_cos W c=0.999 | 0.0822 | 0.0676 | 0.0946 | 0.125 | 0 | -0.0141 | -0.0638 | 0.0179 | 6 | 3 | 0.5078 | 0.0478 | 0.0331 | 0.0607 | 0.0042 | 0.0822 |
| oracle_settle | oracle_settle W delta=0.05 | 0.0864 | 0.0725 | 0.0978 | 0.1562 | 0 | -0.0099 | -0.0549 | 0.0215 | 7 | 2 | 0.1797 | 0.052 | 0.0385 | 0.0627 | 0.0005 | 0.0864 |
| oracle_teps | oracle_teps | 0.0086 | 0.0072 | 0.0097 | 0 | 0 | -0.0877 | -0.1293 | -0.0578 | 16 | 0 | 0 | -0.0258 | -0.0334 | -0.0184 | 0 | 0.0086 |


## mnist, depth split (calibrate ('mnist d3',), test ('mnist d8',)), lambda = 0.1

| family | setting | test_cost | ci_lo | ci_hi | premature | never | d_horizon | d_horizon_lo | d_horizon_hi | cells_better_h | cells_worse_h | p_h | d_fixed | d_fixed_lo | d_fixed_hi | p_f | hindsight_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| horizon | horizon | 0.1182 | 0.0572 | 0.2084 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.079 | 0.0078 | 0.1783 | 0.2891 | 0.1182 |
| fixed_frac | fixed_frac c=0.4 | 0.0392 | 0.0284 | 0.0498 | 0.25 | 0 | -0.079 | -0.1783 | -0.0078 | 6 | 2 | 0.2891 | 0 | 0 | 0 | 1 | 0.0392 |
| bound | bound W lag=0.15 w=3 floor=1 est=ratio eps=0.3 gate=norm | 0.1872 | 0.1037 | 0.2733 | 0.25 | 0.75 | 0.069 | 0 | 0.1626 | 0 | 2 | 0.5 | 0.148 | 0.0638 | 0.2393 | 0.0078 | 0.0941 |
| rel_delta | rel_delta G lag=0.15 w=3 floor=1 eps=0.003 gate=none | 0.1182 | 0.0572 | 0.2084 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.079 | 0.0078 | 0.1783 | 0.2891 | 0.1046 |
| const_delta | const_delta G c=0.1 w=3 floor=1 eps=0.3 gate=none | 0.1185 | 0.0837 | 0.1543 | 0.375 | 0.5 | 0.0003 | -0.0752 | 0.0669 | 2 | 2 | 1 | 0.0793 | 0.0431 | 0.1217 | 0.0703 | 0.0957 |
| msd | msd M factor=2 thr=0.5 gate=none | 0.0832 | 0.0211 | 0.1832 | 0.375 | 0.5 | -0.035 | -0.0623 | -0.0111 | 4 | 0 | 0.125 | 0.0439 | -0.0275 | 0.1526 | 1 | 0.0783 |
| geweke | geweke W factor=4 thr=20 gate=none | 0.0982 | 0.0329 | 0.195 | 0.0938 | 0.7812 | -0.02 | -0.0441 | 0 | 2 | 0 | 0.5 | 0.059 | -0.015 | 0.1659 | 1 | 0.0843 |
| F2 best | const_delta G c=0.1 w=3 floor=1 eps=0.3 gate=none | 0.1185 | 0.0837 | 0.1543 | 0.375 | 0.5 | 0.0003 | -0.0752 | 0.0669 | 2 | 2 | 1 | 0.0793 | 0.0431 | 0.1217 | 0.0703 | 0.0783 |
| oracle_cos | oracle_cos G c=0.9 | 0.1951 | 0.0968 | 0.3108 | 0.25 | 0 | 0.0769 | -0.0317 | 0.2166 | 2 | 2 | 1 | 0.1558 | 0.0551 | 0.2739 | 0.0703 | 0.0912 |
| oracle_settle | oracle_settle G delta=0.5 | 0.194 | 0.0942 | 0.3149 | 0.25 | 0 | 0.0758 | -0.041 | 0.2212 | 2 | 2 | 1 | 0.1547 | 0.0534 | 0.276 | 0.0703 | 0.0989 |
| oracle_teps | oracle_teps | 0.0083 | 0.0059 | 0.0097 | 0 | 0 | -0.1099 | -0.1999 | -0.0488 | 8 | 0 | 0.0078 | -0.0309 | -0.0418 | -0.0196 | 0.0078 | 0.0083 |


## Diagnostics (medians per application)

| app | align_M_at_teps | angle_M_at_teps | logw_change_after_teps | erank_ratio_after_teps | align_M_at_tmem | angle_M_at_tmem | logw_change_after_tmem | erank_ratio_after_tmem | spearman_R_default | spearman_theta_const |
|---|---|---|---|---|---|---|---|---|---|---|
| mnist | 0.358 | 1.205 | -0.569 | 0.405 | 0.194 | 1.375 | -0.897 | 0.113 | -0.066 | 0.758 |
| modular | 0.805 | 0.636 | -0.102 | 1.018 | 0.424 | 1.132 | -0.314 | 0.361 | 0.365 | 0.787 |


## Per category (seed split, test runs)

| app | family | category | runs | cost | premature | regret | late |
|---|---|---|---|---|---|---|---|
| modular | horizon | fast | 26 | 0.195 | 0 | 0.0983 | 0.9668 |
| modular | horizon | grokker | 32 | 0.0684 | 0 | 0 | 0.684 |
| modular | horizon | joint_plateau | 2 | 0.5051 | 0 | 0.4097 | 0.9548 |
| modular | horizon | late_collapse | 4 | 0.0726 | 0 | 0.0246 | 0.4799 |
| modular | horizon | never | 16 | 0.107 | 0 | 0.007 | 0.9999 |
| modular | fixed_frac | fast | 26 | 0.0602 | 0 | 0.0118 | 0.4838 |
| modular | fixed_frac | grokker | 32 | 0.0477 | 0.125 | 0.0244 | 0.2326 |
| modular | fixed_frac | joint_plateau | 2 | 0.575 | 0 | 0.5278 | 0.4718 |
| modular | fixed_frac | late_collapse | 4 | 0.2338 | 0.5 | 0.2103 | 0.2351 |
| modular | fixed_frac | never | 16 | 0.0587 | 0 | 0.007 | 0.5169 |
| modular | F2 best | fast | 26 | 0.0891 | 0 | 0.0482 | 0.409 |
| modular | F2 best | grokker | 32 | 0.037 | 0 | 0 | 0.3698 |
| modular | F2 best | joint_plateau | 2 | 0.5977 | 0 | 0.5583 | 0.3945 |
| modular | F2 best | late_collapse | 4 | 0.1337 | 0.25 | 0.1034 | 0.3025 |
| modular | F2 best | never | 16 | 0.0401 | 0 | 0.007 | 0.3302 |
| modular | oracle_cos | fast | 26 | 0.1206 | 0 | 0.0748 | 0.4584 |
| modular | oracle_cos | grokker | 32 | 0.0424 | 0.0625 | 0.0078 | 0.3457 |
| modular | oracle_cos | joint_plateau | 2 | 0.6347 | 0 | 0.5728 | 0.6192 |
| modular | oracle_cos | late_collapse | 4 | 0.1356 | 0.5 | 0.1068 | 0.2886 |
| modular | oracle_cos | never | 16 | 0.0196 | 0 | 0.0071 | 0.1251 |
| modular | oracle_teps | fast | 26 | 0.0074 | 0 | 0.0074 | 0 |
| modular | oracle_teps | grokker | 32 | 0.0081 | 0 | 0.0081 | 0 |
| modular | oracle_teps | joint_plateau | 2 | 0.0083 | 0 | 0.0083 | 0 |
| modular | oracle_teps | late_collapse | 4 | 0.0087 | 0 | 0.0087 | 0 |
| modular | oracle_teps | never | 16 | 0.0004 | 0 | 0.0004 | 0 |
| mnist | horizon | fast | 14 | 0.1031 | 0 | 0.0083 | 0.9482 |
| mnist | horizon | grokker | 10 | 0.0629 | 0 | 0.0057 | 0.5721 |
| mnist | horizon | late_collapse | 2 | 0.3642 | 0 | 0.2984 | 0.6575 |
| mnist | horizon | never | 6 | 0.0467 | 0 | 0.0004 | 0.4634 |
| mnist | fixed_frac | fast | 14 | 0.0313 | 0 | 0.0051 | 0.2622 |
| mnist | fixed_frac | grokker | 10 | 0.0287 | 0.7 | 0.027 | 0.0174 |
| mnist | fixed_frac | late_collapse | 2 | 0.0215 | 1 | 0.0215 | 0 |
| mnist | fixed_frac | never | 6 | 0.0552 | 0.6667 | 0.0448 | 0.1044 |
| mnist | F2 best | fast | 14 | 0.0663 | 0.2857 | 0.0092 | 0.5706 |
| mnist | F2 best | grokker | 10 | 0.0508 | 0.2 | 0.009 | 0.4182 |
| mnist | F2 best | late_collapse | 2 | 0.3642 | 0 | 0.2984 | 0.6575 |
| mnist | F2 best | never | 6 | 0.0135 | 0 | 0.0004 | 0.1316 |
| mnist | oracle_cos | fast | 14 | 0.1001 | 0 | 0.0083 | 0.9171 |
| mnist | oracle_cos | grokker | 10 | 0.0606 | 0 | 0.0064 | 0.5426 |
| mnist | oracle_cos | late_collapse | 2 | 0.0202 | 0 | 0.0139 | 0.0635 |
| mnist | oracle_cos | never | 6 | 0.0971 | 0.6667 | 0.0638 | 0.3331 |
| mnist | oracle_teps | fast | 14 | 0.0085 | 0 | 0.0085 | 0 |
| mnist | oracle_teps | grokker | 10 | 0.0098 | 0 | 0.0098 | 0 |
| mnist | oracle_teps | late_collapse | 2 | 0.0098 | 0 | 0.0098 | 0 |
| mnist | oracle_teps | never | 6 | 0.0066 | 0 | 0.0066 | 0 |

