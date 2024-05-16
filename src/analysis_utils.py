import copy

import numpy as np
import random

def min_max_normalization(X, y_range):

	new_min = y_range[0]
	new_max = y_range[1]
	Y = np.zeros(X.shape)

	_min = np.min(X)
	_max = np.max(X)

	for i, x in enumerate(X):
		Y[i] = ((x - _min) / (_max - _min)) * (new_max - new_min) + new_min

	return Y


def convert_to_single_row(DD):
	"""temp: until process_recordings does this

	0       1          2                  3              4                5                6
	ELO,  won Y/N, ini_times_avg_rat, ini_objs_tot, ini_targets, ini_group_size_avg,   profile_id

	toa

	0         1          2                  3            4                5                6        7                   8             9                 10                11
	winner,  ELO0 , ini_times_avg_rat0, ini_objs_tot0, ini_targets0, ini_group_size_avg0, ELO1, ini_times_avg_rat1, ini_objs_tot1, ini_targets1, ini_group_size_avg1, profile_id

	"""

	'''
	first the rows with time > 0.3333 need to be removed. THEY ARE INVALID -> the assumption is that 
	we are only aware of what happens at the first third of aggression
	'''

	D = np.zeros(shape=(len(DD), 12), dtype=float)

	i = 0  # D
	i0 = 0  # DD
	i1 = 1  # DD
	while i1 < len(DD):
		row0 = DD[i0, :]
		row1 = DD[i1, :]

		if (row0[6] != row1[6]):
			print("no pair game")
			i0 += 1
			i1 += 1
			continue

		if (row0[0] < 10 or row1[0] < 10):
			print("player missing elo")
			i0 += 2
			i1 += 2
			continue

		if row0[1] > 0.5 and row1[1] < 0.5:
			D[i, 0] = 0
		elif row0[1] < 0.5 and row1[1] > 0.5:
			D[i, 0] = 1
		else:
			raise Exception("wrong winner thing")

		D[i, 1] = row0[0]  # elo
		D[i, 6] = row1[0]  # elo

		if row0[2] < 0.34:  # ini_times_avg_rat
			D[i, [2, 3, 4, 5]] = [row0[2], row0[3], row0[4], row0[5]]
		else:  # restore defaults
			D[i, [2, 3, 4, 5]] = [1, 0, 0, 0]

		if row1[2] < 0.34:  # ini_times_avg_rat
			D[i, [7, 8, 9, 10]] = [row1[2], row1[3], row1[4], row1[5]]
		else:
			D[i, [7, 8, 9, 10]] = [1, 0, 0, 0]

		D[i, 11] = row0[6]

		i += 1  # D
		i0 += 2  # DD
		i1 += 2  # DD

	D = D[np.where(D[:, 1] > 0)[0], :] # needed to remove the extra rows

	np.save('./data_proc/D3_6000.npy', D)


def flatten_winner_loser(DD, TIME_CUT):

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

	"""

	'''
	need to randomly select whether the loser or the winner appears first
	'''

	# rows = np.where((DD[:, 13] > (0)) & (DD[:, 13] < (TIME_CUT + 0.05)))[0]
	rows = np.where(DD[:, 13] < (TIME_CUT + 0.05))[0]  # cut away unseen data
	D = DD[rows, :]

	# D_out = np.zeros(shape=(len(D) * 2, 14), dtype=np.float32)  # the input to the violin plot
	D_out = np.zeros(shape=D.shape, dtype=np.float32)  # the input to the violin plot
	# win_rows = np.arange(0, len(D))
	# loss_rows = np.arange(len(D), len(D) * 2)

	# D_out[win_rows, 0] = 1

	for row in range(0, len(D)):
		if random.random() < 0.5:
			D_out[row, 0] = 1  # winner. Everything else remains
			D_out[row, 1:] = D[row, 1:]
		else:
			D_out[row, 0] = 0
			'''loser. Swaps position. OBS DANGEROUS AS IT MAKES SAME DATA REAPPEAR
			MORE IMPORTANTLY, no need to discard 50% of data!!! Just use noise, at least for training set
			'''
			D_out[row, 1] = D[row, 7]
			D_out[row, 2] = D[row, 8]
			D_out[row, 3] = D[row, 9]
			D_out[row, 4] = D[row, 10]
			D_out[row, 5] = D[row, 11]
			D_out[row, 6] = D[row, 12]

			D_out[row, 7] = D[row, 1]
			D_out[row, 8] = D[row, 2]
			D_out[row, 9] = D[row, 3]
			D_out[row, 10] = D[row, 4]
			D_out[row, 11] = D[row, 5]
			D_out[row, 12] = D[row, 6]

		D_out[row, 13] = D[row, 13]
		D_out[row, 14] = D[row, 14]
		D_out[row, 15] = D[row, 15]
		D_out[row, 16] = D[row, 16]
		D_out[row, 17] = D[row, 17]

	return D_out


def weighted_means(D, COLS):

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

	"""

	if D[0, 13] > 0.15:
		raise Exception("first row time_cut is not 0.1")
	if D[-1, 13] < 0.95:
		raise Exception("last row time_cut is not 1.0")
	if len(D) % 10 != 0:
		raise Exception("Matches not stored in 10ths")

	# rows = np.where(D[:, 13] < (TIME_CUT + 0.05))[0]  # cut away unseen data
	rows_max = np.where(D[:, 13] > 0.95)[0]

	D_ = copy.deepcopy(D)

	"""Assumes that matches are sorted according to TIME_CUTS"""

	for i, row_max in enumerate(rows_max):
		rows_m = list(range(row_max - 9, row_max + 1))
		m = D[rows_m, :]
		m_ = copy.deepcopy(m) # Not using zeros here cuz ELO should not be weighted

		if m[0, 13] > 0.15 or m[-1, 13] < 0.95 or len(m) != 10:
			raise Exception("incorrect data")

		'''Take all the data available until a TIME_CUT, i.e., 13'''
		for row in range(1, 10):  # first one is ALREADY THE MEAN
			w = m[0:row + 1, 13]  # weights (i.e. TIME_CUT)
			for COL in COLS:
				x = m[0:row + 1, COL] # GETTIGN IS UBE, BUT NOT SETTING
				m_[row, COL] = np.average(x, weights=w)

		D_[rows_m, :] = m_

		if i % 100 == 0:
			print(row_max)

	return D_
