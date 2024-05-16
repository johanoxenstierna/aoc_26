
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# sns.reset_orig()
import pandas as pd

from src.analysis_utils import *

D = np.load('./data_proc/D_diffs.npy')  # OBS DONT LOOK AT LEN HERE
# D = D[np.where((D[:, 1] > 0) & (D[:, 7] > 0))[0], :]


"""

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

# TIME_CUT = 0.2  # ONLY FOR ELO
# D = D[np.where(D[:, 7] < TIME_CUT)[0], :]
# D = D[np.where(D[:, 10] > 2000)[0], :]


TITLE = 'Initiative through time'
XLABEL = 'Time (T_i)'
YLABEL = 'Initiative'
# D[:, 1] = min_max_normalization(D[:, 1], y_range=[-1, 1])  # ONLY FOR ELO

# TITLE = 'Initiative and Elo (pvp difference)'
# XLABEL = 'Elo (pvp difference)'
# YLABEL = 'Initiative'
# D[:, 1] = np.abs(D[:, 1])

D[:, 2] = min_max_normalization(D[:, 2], y_range=[-1, 1])
D[:, 3] = min_max_normalization(D[:, 3], y_range=[-1, 1])
D[:, 4] = min_max_normalization(D[:, 4], y_range=[-1, 1])
D[:, 5] = min_max_normalization(D[:, 5], y_range=[-1, 1])
# D[:, 10] = min_max_normalization(D[:, 10], y_range=[-1, 1])  # NOT ZERO SUM

COL = 1
# D_COL = D[:, 1]  # NOT USEFUL OUTSIDE RF
# D_COL = D[:, 1] + D[:, 2] + D[:, 3] + D[:, 4] + D[:, 5]
D_COL = D[:, 2] + D[:, 3] + D[:, 4] + D[:, 5]

D_COL = min_max_normalization(D_COL, y_range=[-1, 1])

win_rows = np.where(D[:, 0] > 0.5)[0]
loss_rows = np.where(D[:, 0] < 0.5)[0]

'''some stats'''
# won_and_COL = np.mean(D[win_rows, COL])
# loss_and_COL = np.mean(D[loss_rows, COL])
#
# print("won_and_COL: " + str(won_and_COL))
# print("lost_and_COL: " + str(loss_and_COL))

# won_and_ELO = np.mean(wins[:, 0])
# lost_and_ELO = np.mean(losses[:, 0])
# gradient = 1 - lost_and_ELO / won_and_ELO
# print("gradient ELO diff: " + str(gradient))



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
'''

# [elo_cat, won_lost, COL_TO_TEST value]
# V = np.zeros(shape=(len(D) * 2, 3), dtype=np.float32)  # the input to the violin plot
# win_rows = np.arange(0, len(D))
# loss_rows = np.arange(len(D), len(D) * 2)
#
# V[win_rows, 1] = 1
# V[loss_rows, 1] = 0
#
# V[win_rows, 2] = D[:, WIN_COLS[COL_TO_TEST_INDEX]]
# V[loss_rows, 2] = D[:, LOSS_COLS[COL_TO_TEST_INDEX]]

elos = np.zeros(shape=(len(D),), dtype=np.float32)
elos[win_rows] = abs(D[win_rows, 1])
elos[loss_rows] = abs(D[loss_rows, 1])

# times = np.zeros(shape=(len(D),), dtype=np.float32)
times = D[:, 7]
times = np.rint(times * 10).astype(int)

# to_match_elos = [0, 0.03, 0.06, 0.125, 0.25, 0.5, 1.0]  # NOT NEEDED FOR TIMES
to_match_elos = [0, 12, 25, 50, 100, 200, 400, 800, 1600]  # NOT NEEDED FOR TIMES

for i in range(0, len(to_match_elos) - 1):
    _matches_elos = np.where((elos >= to_match_elos[i]) & (elos < to_match_elos[i + 1]))[0]
    # elos[_matches_elos] = int(0.5 * (to_match_elos[i] + to_match_elos[i + 1]) * 100)  # DEPR
    elos[_matches_elos] = int(0.5 * (to_match_elos[i] + to_match_elos[i + 1]))

df = pd.DataFrame({'Winner?': pd.Series(D[:, 0], dtype='bool'),
                   # XLABEL: pd.Series(elos, dtype='int'),
                   XLABEL: pd.Series(times, dtype='int'),
                   YLABEL: pd.Series(D_COL, dtype='float')})

fig, ax0 = plt.subplots(figsize=(7, 5))
# plt.gca().invert_yaxis()  # DOESNT WORK FOR INNER

# ELOS =============================
# ax = sns.violinplot(data=df, x=XLABEL, y=YLABEL, hue="Winner?", split=True, orient='v',
#                # hue_order=[True, False],
#                palette={True: 'blue', False: 'red'},
#                cut=0,
#                density_norm='width',
#                )

# ELO_DIFFS =============================================
# ax = sns.boxplot(data=df, x=XLABEL, y=YLABEL, hue="Winner?",
#                  hue_order=[True, False],
#                  palette={True: 'blue', False: 'red'},
#                  fliersize=0,
#                  whis=[5, 95]
#                  )

# DO IT IN PAINT ===========

# TIMES =================
ax = sns.lineplot(data=df, x=XLABEL, y=YLABEL, hue='Winner?', hue_order=[True, False],
             errorbar=('sd', 1), err_style='band')  # ('ci', 99)

# ax = sns.lineplot(data=df, x=XLABEL, y=YLABEL, errorbar='sd', err_style='band')

# ax = plt.gca()
# ax.set_xlim([xmin, xmax])
ax.set_ylim([-1, 1])

plt.title(TITLE, fontsize=15)
plt.xlabel(XLABEL, fontsize=15)
plt.ylabel(YLABEL, fontsize=15)
plt.legend(loc='upper left', title='Winner?', title_fontsize=14, fontsize=14)
# plt.legend(loc='lower right', title='Winner?', title_fontsize=3, fontsize=3)
plt.show()
adf = 5