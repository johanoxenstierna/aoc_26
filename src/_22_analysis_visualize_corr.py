
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# sns.reset_orig()
import pandas as pd

from src.analysis_utils import *


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

D = np.load('./data_proc/D_diffs.npy')  # OBS DONT LOOK AT LEN HERE
# D = D[np.where((D[:, 1] > 0) & (D[:, 7] > 0))[0], :]

# # TIME_CUT = 0.5
D = D[np.where(D[:, 7] > 0.55)[0], :]

# TITLE = 'Initiative through time'
TYPE = 1
if TYPE == 0:
    TITLE = 'Initiative and Elo (pvp difference)'
    D[:, 1] = np.abs(D[:, 1])
    # XLABEL = 'Time (T)'
    XLABEL = 'Elo (normalized difference)'
    YLABEL = 'Initiative'

    D[:, 2] = min_max_normalization(D[:, 2], y_range=[-1, 1])
    D[:, 3] = min_max_normalization(D[:, 3], y_range=[-1, 1])
    D[:, 4] = min_max_normalization(D[:, 4], y_range=[-1, 1])
    D[:, 5] = min_max_normalization(D[:, 5], y_range=[-1, 1])

    D_COL = D[:, 2] + D[:, 3] + D[:, 4] + D[:, 5]
    # D_COL = D[:, 2] + D[:, 3] + D[:, 4] + D[:, 5]
    D_COL = min_max_normalization(D_COL, y_range=[-1, 1])
elif TYPE == 1:
    TITLE = 'delta_S and |S|'

# COL = 6  # THIS IS AN INDEX!!!

# win_rows = np.where(D[:, 0] > 0.5)[0]
# loss_rows = np.where(D[:, 0] < 0.5)[0]


# '''some stats'''
# won_and_COL = np.mean(D[win_rows, COL])
# loss_and_COL = np.mean(D[loss_rows, COL])
#
# print("won_and_COL: " + str(won_and_COL))
# print("lost_and_COL: " + str(loss_and_COL))

# won_and_ELO = np.mean(wins[:, 0])
# lost_and_ELO = np.mean(losses[:, 0])
# gradient = 1 - lost_and_ELO / won_and_ELO
# print("gradient ELO diff: " + str(gradient))


''' cols using time'''
# D = wins

# wins_avg_time = np.mean(wins[wins_and_first, COL])
# losses_avg_time = np.mean(losses[losses_and_sec, COL])

# fig, ax0 = plt.subplots(figsize=(12, 12))

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

# elos = np.zeros(shape=(len(D),), dtype=np.float32)
# elos[win_rows] = abs(D[win_rows, 1])
# elos[loss_rows] = abs(D[loss_rows, 1])

# to_match = np.linspace(1000, 2900, 10, dtype=int)
# to_match = list(range(900, 3100, 300))
# to_match = [-1, -0.05, -0.02, 0, 0.02, 0.05, 1]
# to_match_elos = [0, 0.03, 0.06, 0.125, 0.25, 0.5, 1.0]  # NOT NEEDED FOR TIMES
# to_match = [0, 0.01, 0.02, 0.05, 0.1, 1]
# to_match = [-1, 0.02, 0.05, 1]
# avg_elos = np.zeros(shape=(len(D),), dtype=float)
#
# for i in range(0, len(to_match) - 1):
#     _matches = np.where((elos >= to_match[i]) & (elos < to_match[i + 1]))[0]
#     elos[_matches] = int(0.5 * (to_match[i] + to_match[i + 1]) * 100)

# for i in range(0, len(to_match_elos) - 1):
#     _matches_elos = np.where((elos >= to_match_elos[i]) & (elos < to_match_elos[i + 1]))[0]
#     elos[_matches_elos] = int(0.5 * (to_match_elos[i] + to_match_elos[i + 1]) * 100)

# df = pd.DataFrame({
#     'won?': pd.Series(D[:, 0], dtype='bool'),
#     # 'elo_diff_percentage': pd.Series(elos, dtype='int'),
#     'ELO_diff': pd.Series(D[:, 1], dtype='float'),
#     # 'ini_actions_prop_diff': pd.Series(D[:, 2], dtype='float'),
#     # 'ini_objs_diff': pd.Series(D[:, 3], dtype='float'),
#     'ini_objs_prop_diff': pd.Series(D[:, 4], dtype='float'),
#     'ini_targets_prop_diff': pd.Series(D[:, 5], dtype='float'),
# })

if TYPE == 0:
    fig, ax0 = plt.subplots(figsize=(7, 5))
    df = pd.DataFrame({'Winner?': pd.Series(D[:, 0], dtype='bool'),
                       XLABEL: pd.Series(D[:, 1], dtype='float'),
                       # XLABEL: pd.Series(times, dtype='int'),
                       YLABEL: pd.Series(D_COL, dtype='float')})

    # sns.pairplot(df, hue='Winner?', hue_order=[True, False], kind='hist')
    sns.scatterplot(data=df, x=XLABEL, y=YLABEL, size=0.1)
    slope_blue, intercept, r_value, p_value_blue, std_err = stats.linregress(x=D[:, 1], y=D_COL)
    R2 = r_value ** 2
    print("R2: " + str(R2))

    # ax = sns.regplot(data=df, x=XLABEL, y=YLABEL)

    plt.title(TITLE, fontsize=15)
    plt.xlabel(XLABEL, fontsize=15)
    plt.ylabel(YLABEL, fontsize=15)

elif TYPE == 1:
    fig, ax0 = plt.subplots(figsize=(6, 5))
    df = pd.DataFrame({
        'Winner?': pd.Series(D[:, 0], dtype='bool'),
        # 'elo_diff': pd.Series(D[:, 1], dtype='float'),  # TODO ELO DIFFERENCE
        'actions': pd.Series(D[:, 2], dtype='float'),
        'ini_objs_diff': pd.Series(D[:, 3], dtype='float'),
        'subjects': pd.Series(D[:, 4], dtype='float'),
        'objects': pd.Series(D[:, 5], dtype='float'),
        # 'time_cut': pd.Series(D[:, 7])
        # 'elo_avg': pd.Series(D[:, 10])
    })

    # sns.pairplot(df, kind='scatter', diag_kind=None, corner=True)  # kind : {'scatter', 'kde', 'hist', 'reg'}

    sns.scatterplot(df, x='ini_objs_diff', y='objects',
                    hue='Winner?', hue_order=[True, False])
    # plt.title(TITLE, fontsize=15)
    # plt.legend(loc='upper left', title='Winner?', title_fontsize=14, fontsize=14)

plt.show()
adf = 5