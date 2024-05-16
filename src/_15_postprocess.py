


"""

D_out[:, 0] = won_lost
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

TO

D_out[:, 0] = D_flat[:, 0]
D_out[:, 1] = ELO_diff
D_out[:, 2] = ini_actions_prop_diff
D_out[:, 3] = ini_objs_diff
D_out[:, 4] = ini_objs_prop_diff
D_out[:, 5] = ini_targets_prop_diff
D_out[:, 6] = ini_group_size_avg_diff
D_out[:, 7] = time_cut
D_out[:, 8] = t0_ratio
D_out[:, 9] = t_end
D_out[:, 10] = ELO_avg

"""

import numpy as np

from src.analysis_utils import *

PATH_IN = './data_proc/D_comb.npy'
PATH_OUT = './data_proc/D_diffs.npy'

D = np.load(PATH_IN)
D = D[np.where((D[:, 1] > 0) & (D[:, 7] > 0))[0], :]  # remove bad rows

# [ELO, ini_actions_prop, ini_objs, ini_objs_prop, ini_targets_prop, ini_group_size_avg]
WIN_COLS = [1, 2, 3, 4, 5, 6]
LOSS_COLS = [7, 8, 9, 10, 11, 12]
COLS = WIN_COLS[1:] + LOSS_COLS[1:]

D_ = weighted_means(D, COLS)  # this function doesnt care about winner-loser
# np.save('./data_proc/D_comb_weighed.npy', D_)  # TEMP

D_flat = flatten_winner_loser(D, TIME_CUT=1.0)

ELO_diff = D_flat[:, 1] - D_flat[:, 7]
ELO_avg = (D_flat[:, 1] + D_flat[:, 7]) / 2
# NEW!!!! DONt do this # ELO_diff = min_max_normalization(ELO_diff, y_range=[-1, 1])  # TODO: Use standardization instead
ini_actions_prop_diff = D_flat[:, 2] - D_flat[:, 8]
ini_objs_diff = D_flat[:, 3] - D_flat[:, 9]
ini_objs_prop_diff = D_flat[:, 4] - D_flat[:, 10]
ini_targets_prop_diff = D_flat[:, 5] - D_flat[:, 11]
ini_group_size_avg_diff = D_flat[:, 6] - D_flat[:, 12]

D_out = np.zeros(shape=(len(D_flat), 11))
D_out[:, 0] = D_flat[:, 0]
D_out[:, 1] = ELO_diff
D_out[:, 2] = ini_actions_prop_diff
D_out[:, 3] = ini_objs_diff
D_out[:, 4] = ini_objs_prop_diff
D_out[:, 5] = ini_targets_prop_diff
D_out[:, 6] = ini_group_size_avg_diff
D_out[:, 7] = D_flat[:, 13]
D_out[:, 8] = D_flat[:, 16]
D_out[:, 9] = D_flat[:, 17]
D_out[:, 10] = ELO_avg
# OBS extend np.zeros above

np.save(PATH_OUT, D_out)


adf = 4


