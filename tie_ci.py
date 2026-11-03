import numpy as np
import pandas as pd

d = pd.read_csv('tie_pool.csv')
ok = d[d.n_pool > 0]
rng = np.random.default_rng(20261028)
B = 5000
out = {}
for col in ['gap_spread', 'expo_spread']:
    x = ok[col].to_numpy(float)
    draws = rng.choice(x, size=(B, x.size), replace=True)
    med = np.median(draws, axis=1)
    out[col] = (np.median(x), np.percentile(med, 2.5), np.percentile(med, 97.5))
    print(col, 'median %.4f  95%% CI [%.4f, %.4f]' % out[col])
print('ratio of medians %.1f' % (out['expo_spread'][0] / out['gap_spread'][0]))
