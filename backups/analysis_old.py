
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# sns.reset_orig()
import pandas as pd

from src.analysis_utils import *

D = np.load('./data_proc/DD3_6000.npy')

# DD = DD[0:2000, :]
# np.save('./data_proc/DD3_2000.npy', DD)

# D = convert_to_single_row(DD)

fig, ax0 = plt.subplots(figsize=(12, 12))

"""
0       1          2                  3              4                5                6
ELO,  won Y/N, ini_times_avg_rat, ini_objs_tot, ini_targets, ini_group_size_avg,   profile_id
"""

COL = 5

D = D[np.where((D[:, 0] > 0) & (D[:, COL] > 0))[0], :]  # only rows with ELO and good COL

# aa = min_max_normalization(D[:, 3], y_range=[-1, 0])

win_rows = np.where(D[:, 1] > 0.5)[0]
losses_rows = np.where(D[:, 1] < 0.5)[0]

wins_and_rat = np.mean(D[win_rows, 2])
wins_and_1_rat = np.where((D[:, 1] > 0.5) & (np.isclose(D[:, 2], 1.0)))[0]
# losses_and_rat = np.mean(D[losses_rows, 2])

# first_rows = np.where((D[:, COL + 3] > 0) & (D[:, COL + 3] < 10))[0]
# secs_rows = np.where((D[:, COL + 3] < 1) & (D[:, COL + 3] < 10))[0]

wins = D[win_rows, :]
losses = D[losses_rows, :]

min_ = 0
# all_ok_rows_winner = np.where(wins[:, COL] > min_)[0]
# all_ok_rows_loser = np.where(losses[:, COL] > min_)[0]

# wins_and_first = np.where((wins[:, COL + 3] > 0.5) & (wins[:, COL] > min_))[0]
# wins_and_sec = np.where((wins[:, COL + 3] < 0.5) & (wins[:, COL] > min_))[0]

# losses_and_first = np.where((losses[:, COL + 3] > 0.5) & (losses[:, COL] > min_))[0]
# losses_and_sec = np.where((losses[:, COL + 3] < 0.5) & (losses[:, COL] > min_))[0]

# firsts = D[first_rows, :]
# secs = D[secs_rows, :]

'''some stats'''
won_and_COL = np.mean(wins[:, COL])
lost_and_COL = np.mean(losses[:, COL])
print("won_and_COL: " + str(won_and_COL))
print("lost_and_COL: " + str(lost_and_COL))

won_and_ELO = np.mean(wins[:, 0])
lost_and_ELO = np.mean(losses[:, 0])
gradient = 1 - lost_and_ELO / won_and_ELO
print("gradient ELO diff: " + str(gradient))


''' cols using time'''
# D = wins

# wins_avg_time = np.mean(wins[wins_and_first, COL])
# losses_avg_time = np.mean(losses[losses_and_sec, COL])

# '''Scatter plot and reg line'''
# temp = np.zeros(shape=(len(all_ok_rows_winner), 2), dtype=int)
# temp[:, 0], temp[:, 1] = wins[all_ok_rows_winner, 0], wins[all_ok_rows_winner, COL]
# df_blue = pd.DataFrame(temp, columns=['ELO', 'time'])
# ax_blue = ax0.scatter(df_blue['ELO'], df_blue['time'], c='blue', s=100, alpha=0.1)
#
# temp = np.zeros(shape=(len(all_ok_rows_loser), 2), dtype=int)
# temp[:, 0], temp[:, 1] = losses[all_ok_rows_loser, 0], losses[all_ok_rows_loser, COL]
# df_red = pd.DataFrame(temp, columns=['ELO', 'time'])
# ax_red = ax0.scatter(df_red['ELO'], df_red['time'], c='red', s=100, alpha=0.01)
#
# # res = stats.goodness_of_fit(stats.norm, A, statistic='ks', random_state=np.random.default_rng())
# slope_blue, intercept, r_value, p_value_blue, std_err = stats.linregress(wins[all_ok_rows_winner, 0], wins[all_ok_rows_winner, COL])
# slope_red, intercept, r_value, p_value_red, std_err = stats.linregress(losses[all_ok_rows_loser, 0], losses[all_ok_rows_loser, COL])
# print("slope_blue: " + str(slope_blue) + "  p_value blue: " + str(p_value_blue))
# print("slope_loser: " + str(slope_red) + "  p_value red: " + str(p_value_red))
#
# ax_blue_reg = sns.regplot(x=df_blue['ELO'], y=df_blue['time'], ci=True, line_kws={'color': 'blue', 'alpha': 0.3}, scatter=False)
# ax_red_reg = sns.regplot(x=df_red['ELO'], y=df_red['time'], ci=True, line_kws={'color': 'red', 'alpha': 0.3}, scatter=False)


'''violin'''
# to_match = np.linspace(1000, 2900, 10, dtype=int)
to_match = list(range(1000, 2900, 200))

elo_cats = np.zeros(shape=(len(D), 1), dtype=float)

for i in range(0, len(to_match) - 1):
	matches = np.where((D[:, 0] >= to_match[i]) & (D[:, 0] < to_match[i + 1]))[0]
	elo_cats[matches, 0] = to_match[i] + 100

D = np.concatenate((D, elo_cats), axis=1)
D = D[np.where(D[:, -1] > 10)[0], :]
# D = D[0:500, :]

# temp = D[:, [1, 12, COL]].astype(int)
# df = pd.DataFrame(temp, columns=['won_lost', 'elo_cats', 'COL'])
df = pd.DataFrame({'won_lost': pd.Series(D[:, 1], dtype='bool'),
                   'elo_cats': pd.Series(D[:, -1], dtype='int'),
                   'COL': pd.Series(D[:, COL], dtype='float')})

sns.violinplot(data=df, x="COL", y="elo_cats", hue="won_lost", split=True, orient='h',
               # hue_order=[True, False],
               palette={True: 'blue', False: 'red'},
               cut=0
               # sharey=False
               )

plt.gca().invert_yaxis()


plt.show()
adf = 5