

'''Need to get both datasets for this one'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# sns.reset_orig()
import pandas as pd

from src.analysis_utils import *


"""

D_out[:, 0] = won_lost (NOT RELEVANT)
D_out[:, 1] = ELO0
D_out[:, 2] = ini_actions_prop0
D_out[:, 3] = ini_objs0
D_out[:, 4] = ini_objs_prop0
D_out[:, 5] = ini_targets_prop0
D_out[:, 6] = ini_group_size_avg0
D_out[:, 7] = ELO1
D_out[:, 8] = ini_actions_prop1
D_out[:, 9] = ini_objs1
D_out[:, 10] = ini_objs_prop1
D_out[:, 11] = ini_targets_prop1
D_out[:, 12] = ini_group_size_avg1
D_out[:, 13] = time_cut
D_out[:, 14] = profile_id_save
D_out[:, 15] = match_time
D_out[:, 16] = t0_ratio
D_out[:, 17] = t_end

D_out[:, 0] = D_flat[:, 0]
D_out[:, 1] = ELO_diff
D_out[:, 2] = ini_actions_prop_diff
D_out[:, 3] = ini_objs_diff
D_out[:, 4] = ini_objs_prop_diff
D_out[:, 5] = ini_targets_prop_diff
D_out[:, 6] = ini_group_size_avg_diff
D_out[:, 7] = time_cut (t!)
D_out[:, 8] = t0_ratio
D_out[:, 9] = t_end
D_out[:, 10] = ELO_avg

"""

D_raw = np.load('./data_proc/D_comb.npy')  # OBS DONT LOOK AT LEN HERE
D_diffs = np.load('./data_proc/D_diffs.npy')

TITLE = 'Initiative and Elo'
XLABEL = 'Elo'
YLABEL = 'Initiative'

'''ONLY WANT DATA WHERE T_CUT == 1'''
# D_raw = D_raw[np.where(D_raw[:, 13] > 0.95)[0], :]
# D_diffs = D_diffs[np.where(D_diffs[:, 7] > 0.95)[0], :]

'''GEN INI FEATURE'''
D_diffs[:, 2] = min_max_normalization(D_diffs[:, 2], y_range=[-1, 1])
D_diffs[:, 3] = min_max_normalization(D_diffs[:, 3], y_range=[-1, 1])
D_diffs[:, 4] = min_max_normalization(D_diffs[:, 4], y_range=[-1, 1])
D_diffs[:, 5] = min_max_normalization(D_diffs[:, 5], y_range=[-1, 1])

D_COL = D_diffs[:, 2] + D_diffs[:, 3] + D_diffs[:, 4] + D_diffs[:, 5]
D_COL = min_max_normalization(D_COL, y_range=[-1, 1])

win_rows = np.where(D_diffs[:, 0] > 0.5)[0]
loss_rows = np.where(D_diffs[:, 0] < 0.5)[0]

elos_ = np.zeros(shape=(len(D_raw)), dtype=float)
elos_[win_rows] = D_raw[win_rows, 1]
elos_[loss_rows] = D_raw[loss_rows, 7]

fig, ax0 = plt.subplots(figsize=(7, 5))

elo_cats = np.zeros(shape=(len(D_raw),), dtype=np.float32)
to_match_elos = list(range(800, 3100, 300))

for i in range(0, len(to_match_elos) - 1):
    matches = np.where((elos_ >= to_match_elos[i]) & (elos_ < to_match_elos[i + 1]))[0]
    elo_cats[matches] = to_match_elos[i] + 100

# elo_cats[:] = np.full(shape=(len(elo_cats),), fill_value=1000)

'''t0_ratio vs ELO'''

# NEW: filter on rows!
# rows = np.where((elos_ > 1100) & (elos_ < 1300))[0]
# rows = np.where((D_diffs[:, 0] < 0.5) & (elos_ > 1100) & (elos_ < 1300))[0]
df = pd.DataFrame({'Winner?': pd.Series(D_diffs[:, 0], dtype='bool'),
                   # XLABEL: pd.Series(elos_[:,], dtype='int'),
                   XLABEL: pd.Series(elo_cats[:], dtype='int'),
                   YLABEL: pd.Series(D_COL[:], dtype='float')})

# Not as good as box
# ax = sns.violinplot(data=df, x=XLABEL, y=YLABEL, hue="Winner?", split=True, orient='v',
#                     # hue_order=[True, False],
#                     palette={True: 'blue', False: 'red'},
#                     # cut=0,
#                     inner=None,
#                     density_norm='area',
#                     )


ax = sns.boxplot(data=df, x=XLABEL, y=YLABEL, hue="Winner?",
                 hue_order=[True, False],
                 palette={True: 'blue', False: 'red'},
                 fliersize=0,
                 whis=[5, 95]
                 )

# sns.scatterplot(data=df, x=XLABEL, y=YLABEL, size=0.1)
# sns.scatterplot(x=elos_[loss_rows], y=D_COL[loss_rows], size=0.1)
# slope_blue, intercept, r_value, p_value_blue, std_err = stats.linregress(x=elos_[loss_rows], y=D_COL[loss_rows])
# R2 = r_value ** 2
# print("R2: " + str(R2))
# DOESNT WORK
# ax_blue_reg = sns.regplot(x=df[XLABEL], y=df[YLABEL], ci=False, line_kws={'color': 'blue', 'alpha': 0.3}, scatter=False)

plt.title(TITLE, fontsize=15)
plt.xlabel(XLABEL, fontsize=15)
plt.ylabel(YLABEL, fontsize=15)
plt.legend(loc='lower right', title='Winner?', title_fontsize=14, fontsize=14)

plt.show()
dfdf = 5


