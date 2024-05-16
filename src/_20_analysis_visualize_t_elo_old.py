
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
# D = D[np.where((D[:, 1] > 10) & (D[:, 7] > 10))[0], :]
# aa = np.where((D[:, 1] < 10) | (D[:, 7] < 10))[0]

# temp get stats from data
# D = D[np.where(D[:, 13] > 0.95)[0], :]
# t_end_sum = np.sum(D[:, 17]) / 3600

# TIME_CUT = 1.0
# D = D[np.where(D[:, 13] < TIME_CUT)[0], :]

# aa = np.mean(D[:, 1])
# bb = np.mean(D[:, 7])

'''Keep matches where diff in ELO is low'''
rows_to_keep = []
for i in range(0, len(D)):
    diff_elo = abs(D[i, 1] - D[i, 7])
    if diff_elo < 2000:
        rows_to_keep.append(i)

print("Rows before: " + str(len(D)) + "  Rows aft: " + str(len(rows_to_keep)))
D = D[rows_to_keep, :]   # break here to see how many were kept

'''t0_ratio vs ELO. ONLY 1 TIME CUT NEEDED'''
D_flat = flatten_winner_loser(D, TIME_CUT=0.1)  # ONLY NEED 1 ROW/MATCH

# [ELO, ini_actions_prop, ini_objs, ini_objs_prop, ini_targets_prop, ini_group_size_avg]
# COLS = [0, 1, 2, 3, 4, 5]
D_COL = D_flat[:, 16]  # THE SAME VALUE FOR WINNER/LOSER

'''New: change from ratio to absolute for nicer plotting'''
D_COL_0 = D_flat[:, 16] * D_flat[:, 17]  # THE SAME VALUE FOR WINNER/LOSER
D_COL_end = D_flat[:, 17]  # THE SAME VALUE FOR WINNER/LOSER

# COL_TO_TEST_INDEX = 4  # THIS IS AN INDEX!!!
# WIN_COLS = [1, 2, 3, 4, 5, 6]
# LOSS_COLS = [7, 8, 9, 10, 11, 12]

# aa = min_max_normalization(D[:, 3], y_range=[-1, 0])

'''t0_ratio vs ELO'''
TITLE = 't_0/t_end and Elo'
XLABEL = 'Elo'
YLABEL = 't_0/t_end'
YLABEL_0 = 't_0'
YLABEL_end = 't_end'

win_rows = np.where(D_flat[:, 0] > 0.5)[0]
losses_rows = np.where(D_flat[:, 0] < 0.5)[0]

# wins_and_rat = np.mean(D[win_rows, 2])
# wins_and_1_rat = np.where((D[:, 1] > 0.5) & (np.isclose(D[:, 2], 1.0)))[0]
# losses_and_rat = np.mean(D[losses_rows, 2])

# first_rows = np.where((D[:, COL + 3] > 0) & (D[:, COL + 3] < 10))[0]
# secs_rows = np.where((D[:, COL + 3] < 1) & (D[:, COL + 3] < 10))[0]

# wins = D[win_rows, :]
# losses = D[losses_rows, :]
# all_ok_rows_winner = np.where(wins[:, COL] > min_)[0]
# all_ok_rows_loser = np.where(losses[:, COL] > min_)[0]

# wins_and_first = np.where((wins[:, COL + 3] > 0.5) & (wins[:, COL] > min_))[0]
# wins_and_sec = np.where((wins[:, COL + 3] < 0.5) & (wins[:, COL] > min_))[0]

# losses_and_first = np.where((losses[:, COL + 3] > 0.5) & (losses[:, COL] > min_))[0]
# losses_and_sec = np.where((losses[:, COL + 3] < 0.5) & (losses[:, COL] > min_))[0]

# firsts = D[first_rows, :]
# secs = D[secs_rows, :]

'''some stats'''
# won_and_COL = np.mean(D[:, WIN_COLS[COL_TO_TEST_INDEX]])
# loss_and_COL = np.mean(D[:, LOSS_COLS[COL_TO_TEST_INDEX]])
#
# print("won_and_COL: " + str(won_and_COL))
# print("lost_and_COL: " + str(loss_and_COL))

# won_and_ELO = np.mean(wins[:, 0])
# lost_and_ELO = np.mean(losses[:, 0])
# gradient = 1 - lost_and_ELO / won_and_ELO
# print("gradient ELO diff: " + str(gradient))


fig, ax0 = plt.subplots(figsize=(8, 7))

'''
Scatter plot and reg line
DEPR CUZ IT ONLY DOES 1 ELO
'''
# temp = np.zeros(shape=(len(D), 2), dtype=int)
# temp[:, 0], temp[:, 1] = D[:, WIN_COLS[0]], D[:, WIN_COLS[COL_TO_TEST_INDEX]]
# df_blue = pd.DataFrame(temp, columns=['ELO', 'COL'])
# ax_blue = ax0.scatter(df_blue['ELO'], df_blue['COL'], c='blue', s=100, alpha=0.1)

# # res = stats.goodness_of_fit(stats.norm, A, statistic='ks', random_state=np.random.default_rng())
# slope_blue, intercept, r_value, p_value_blue, std_err = stats.linregress(wins[all_ok_rows_winner, 0], wins[all_ok_rows_winner, COL])
# slope_red, intercept, r_value, p_value_red, std_err = stats.linregress(losses[all_ok_rows_loser, 0], losses[all_ok_rows_loser, COL])
# print("slope_blue: " + str(slope_blue) + "  p_value blue: " + str(p_value_blue))
# print("slope_loser: " + str(slope_red) + "  p_value red: " + str(p_value_red))
#
# ax_blue_reg = sns.regplot(x=df_blue['ELO'], y=df_blue['time'], ci=True, line_kws={'color': 'blue', 'alpha': 0.3}, scatter=False)
# ax_red_reg = sns.regplot(x=df_red['ELO'], y=df_red['time'], ci=True, line_kws={'color': 'red', 'alpha': 0.3}, scatter=False)


'''
violin
Needs the winner and loss data to be stacked
PROBABLY DEPR UE TO MOVED ELSEWHERE
'''

# # [elo_cat, won_lost, COL_TO_TEST value]
# V = np.zeros(shape=(len(D) * 2, 3), dtype=np.float32)  # the input to the violin plot
# win_rows = np.arange(0, len(D))
# loss_rows = np.arange(len(D), len(D) * 2)
#
# V[win_rows, 1] = 1
# V[loss_rows, 1] = 0
#
# V[win_rows, 2] = D[:, WIN_COLS[COL_TO_TEST_INDEX]]
# V[loss_rows, 2] = D[:, LOSS_COLS[COL_TO_TEST_INDEX]]
#
# elos = np.zeros(shape=(len(V),), dtype=np.float32)
# elos[win_rows] = D[:, WIN_COLS[0]]
# elos[loss_rows] = D[:, LOSS_COLS[0]]

# to_match = np.linspace(1000, 2900, 10, dtype=int)

elos = np.zeros(shape=(len(D_flat),), dtype=np.float32)
elos[win_rows] = D_flat[win_rows, 1]
elos[losses_rows] = D_flat[losses_rows, 7]
to_match_elos = list(range(800, 3100, 300))

for i in range(0, len(to_match_elos) - 1):
    matches = np.where((elos >= to_match_elos[i]) & (elos < to_match_elos[i + 1]))[0]
    elos[matches] = to_match_elos[i] + 100

    aa = 5

'''t0_ratio vs ELO. OBS MODIFIED'''
df = pd.DataFrame({
    # 'Winner?': pd.Series(D_flat[:, 0], dtype='bool'),
    XLABEL: pd.Series(elos, dtype='int'),
    YLABEL: pd.Series(D_COL),
    YLABEL_0: pd.Series(D_COL_0, dtype='float'),
    YLABEL_end: pd.Series(D_COL_end, dtype='float')
})

# sns.violinplot(data=df, x=XLABEL, y=YLABEL, hue="Winner?", split=True, orient='h',
#                # hue_order=[True, False],
#                palette={True: 'blue', False: 'red'},
#                # cut=0,
#                # inner=None,
#                density_norm='count'
#                )

# ax = sns.violinplot(data=df, x=XLABEL, y=YLABEL, hue=None, split=True, orient='v',
#                     # hue_order=[True, False],
#                     # palette={True: 'blue', False: 'red'},
#                     # cut=0,
#                     density_norm='area',
#                     inner='quart'
#                     )

'''New: Hue based on t_0 or t_end'''
ax = sns.violinplot(data=df, x=XLABEL, y=YLABEL_end, hue=None, split=True, orient='v',
                    # hue_order=[True, False],
                    # palette={True: 'blue', False: 'red'},
                    # cut=0,
                    density_norm='area',
                    inner='quart'
                    )

# plt.gca().invert_yaxis()  # DOESNT WORK FOR INNER
# ax.set_ylim([500, 3000])  # seems to mess up with orient=h, USE PAINT

plt.title(TITLE, fontsize=15)
plt.xlabel(XLABEL, fontsize=15)
plt.ylabel(YLABEL, fontsize=15)
# plt.legend(loc='upper right', title='Winner?', title_fontsize=14, fontsize=14)
plt.show()
adf = 5