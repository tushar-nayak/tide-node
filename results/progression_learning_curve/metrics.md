# Progression call: learning curve

374 rated follow-ups, 79 patients. Each outer fold trains on a random subset of its training patients (10 subsets per fraction below 100%) and is scored on the whole held-out fold; out-of-fold probabilities averaged over 3 fold seeds. Mean AUROC across subsets (SD).

| model | 25% of training patients | 50% of training patients | 75% of training patients | 100% of training patients | gain 75% -> 100% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| released: V+R+S, all scans | 0.767 (0.012) | 0.805 (0.018) | 0.812 (0.006) | 0.820 | +0.008 |
| released: V+R+S+M, all scans | 0.781 (0.013) | 0.816 (0.018) | 0.828 (0.004) | 0.835 | +0.007 |
| fully automated: V, all scans | 0.754 (0.015) | 0.785 (0.017) | 0.781 (0.009) | 0.785 | +0.004 |
| fully automated: V+M, all scans | 0.772 (0.012) | 0.797 (0.017) | 0.797 (0.007) | 0.797 | -0.000 |
| released: V+R+S, previous scan not PD | 0.696 (0.014) | 0.725 (0.021) | 0.723 (0.012) | 0.728 | +0.005 |
| released: V+R+S+M, previous scan not PD | 0.719 (0.019) | 0.745 (0.022) | 0.752 (0.009) | 0.760 | +0.008 |
| fully automated: V, previous scan not PD | 0.706 (0.020) | 0.734 (0.018) | 0.731 (0.010) | 0.734 | +0.003 |
| fully automated: V+M, previous scan not PD | 0.725 (0.018) | 0.753 (0.019) | 0.753 (0.009) | 0.756 | +0.003 |

Training patients per fold at each fraction: 16, 32, 47, 63.
