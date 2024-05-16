
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# sns.reset_orig()
import pandas as pd

from src.analysis_utils import *

"""

OBS NOT POSTPROCESSED

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
"""

D = np.load('./data_proc/D_comb.npy')  # OBS DONT LOOK AT LEN HERE

'''OBS NEW. No need to remove 50% of data. New array is 2x len of D, first half is t_0, second is t_end'''
# D_flat = flatten_winner_loser(D, TIME_CUT=0.1)  # ONLY NEED 1 ROW/MATCH
D = D[np.where(D[:, 13] < 0.15)[0], :]
D_ = np.zeros(shape=(len(D) * 2, 3))
D_[0:len(D), 0] = 0  # t_0
D_[0:len(D), 1] = D[:, 16] * D[:, 17]
D_[0:len(D), 2] = (D[:, 1] + D[:, 7]) / 2
D_[len(D):, 0] = 1  # t_0
D_[len(D):, 1] = D[:, 17]
D_[len(D):, 2] = (D[:, 1] + D[:, 7]) / 2

'''Elos need categorization'''
elos = np.zeros(shape=(len(D_),), dtype=np.float32)

to_match_elos = list(range(800, 3100, 300))

for i in range(0, len(to_match_elos) - 1):
    matches = np.where((D_[:, 2] >= to_match_elos[i]) & (D_[:, 2] < to_match_elos[i + 1]))[0]
    elos[matches] = to_match_elos[i] + 100

D_[:, 2] = elos

'''t0_ratio vs ELO'''
TITLE = 't_0, t_end vs Elo'
XLABEL = 'Elo'
YLABEL = 't'

fig, ax0 = plt.subplots(figsize=(8, 7))

'''t0_ratio vs ELO. OBS MODIFIED'''
df = pd.DataFrame({
    't_0 or t_end?': pd.Series(D_[:, 0], dtype='bool'),
    XLABEL: pd.Series(D_[:, 2], dtype=int),
    YLABEL: pd.Series(D_[:, 1]),
})

sns.violinplot(data=df, x=XLABEL, y=YLABEL, hue="t_0 or t_end?", split=True, orient='v',
               # hue_order=[True, False],
               palette={True: 'blue', False: 'red'},
               # cut=0,
               inner='quart',
               density_norm='area',
               legend=False
               )

# ax = sns.violinplot(data=df, x=XLABEL, y=YLABEL, hue=None, split=True, orient='v',
#                     # hue_order=[True, False],
#                     # palette={True: 'blue', False: 'red'},
#                     # cut=0,
#                     density_norm='area',
#                     inner='quart'
#                     )


# plt.gca().invert_yaxis()  # DOESNT WORK FOR INNER
# ax.set_ylim([500, 3000])  # seems to mess up with orient=h, USE PAINT

plt.title(TITLE, fontsize=15)
plt.xlabel(XLABEL, fontsize=15)
plt.ylabel(YLABEL, fontsize=15)
# plt.legend(loc='upper right', title='t_0 t_end?', title_fontsize=14, fontsize=14)
plt.show()
adf = 5