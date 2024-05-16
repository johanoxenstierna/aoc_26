

import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
from src.analysis_utils import *

"""

Default =============
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


DIFFS =============
D_out[:, 0] = won_lost
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
PATH_OUT = './src/results/'  # result_table
COMB_DIFFS = 0   # 0 is COMB

if COMB_DIFFS == 0:
	D = np.load('./data_proc/D_comb.npy')
	# D = np.load('./data_proc/D_comb_weighed.npy')  # TEMP

	D = flatten_winner_loser(D, TIME_CUT=1.0)  # TIME_CUT always 1 here bcs the cutting is done below now
	COL_time_cut = 13
else:
	D = np.load('./data_proc/D_diffs.npy')
	# D = np.load('./data_proc/D_diffs_weighed.npy')
	COL_time_cut = 7

# '''Keep matches where diff in ELO is low'''
rows_to_keep = []
for i in range(0, len(D)):
	# diff_elo = abs(D[i, 1] - D[i, 7])
	# if COMB_DIFFS == 0:
	if COMB_DIFFS == 0:
		diff_elo = abs(D[i, 1] - D[i, 7])
		if diff_elo < 10000:
			rows_to_keep.append(i)

		# if D[i, 17] > 2000 and D[i, 17] < 3000:
		# 	rows_to_keep.append(i)

	if COMB_DIFFS == 1:
		diff_elo = D[i, 1]
		# if diff_elo < 99999.09:
		# 	rows_to_keep.append(i)

		if D[i, 9] > 0 and D[i, 9] < 5000:
			rows_to_keep.append(i)

		# diff_elo_avg = D[i, 10]
		# if diff_elo_avg > 1400:
		# 	rows_to_keep.append(i)

print("Rows before: " + str(len(D)) + "  Rows aft: " + str(len(rows_to_keep)))
D = D[rows_to_keep, :]   # break here to see how many were kept

time_cut_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# time_cut_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# time_cut_ratios = [0.5]  # OBS TEMP TIME_CUT - 0.3 used as frame
result_table = np.zeros(shape=(len(time_cut_ratios), 12), dtype=float)

for i in range(len(time_cut_ratios)):

	TIME_CUT = time_cut_ratios[i]

	'''OBS. This is where the next paper will change things. Here, only the exact time step is used, 
	or (if modded), all the previous ones. The RF is not given data in the form of a match. 
	So what is going to change is that each match is loaded as a minibatch with time steps and then RNN 
	trains on that.'''

	rows = np.where((D[:, COL_time_cut] > (TIME_CUT - 0.05)) & (D[:, COL_time_cut] < (TIME_CUT + 0.05)))[0]
	# rows = np.where((D[:, COL_time_cut] > (TIME_CUT - 0.5)) & (D[:, COL_time_cut] < (TIME_CUT + 0.05)))[0]
	D_t = D[rows, :]

	y = pd.DataFrame({'win0': pd.Series(D_t[:, 0], dtype='bool')})

	if COMB_DIFFS == 0:
		'''
		COMB. 
		Prevent ELO from reappearing in train/test
		NOT NEEDED
		'''
		D_t[:, 1] += np.random.uniform(low=-20, high=20, size=len(D_t[:, 1]))
		D_t[:, 7] += np.random.uniform(low=-20, high=20, size=len(D_t[:, 7]))

		X = pd.DataFrame({
			# 'elo0': pd.Series(D_t[:, 1], dtype='float'),  # TODO ELO DIFFERENCE
			# 'ini_actions_prop0': pd.Series(D_t[:, 2], dtype='float'),
			'ini_objs0': pd.Series(D_t[:, 3], dtype='float'),
			# 'ini_objs_prop0': pd.Series(D_t[:, 4], dtype='float'),
			# 'ini_targets_prop0': pd.Series(D_t[:, 5], dtype='float'),
			# 'elo1': pd.Series(D_t[:, 7], dtype='float'),
			# 'ini_actions_prop1': pd.Series(D_t[:, 8], dtype='float'),
			'ini_objs1': pd.Series(D_t[:, 9], dtype='float'),
			# 'ini_objs_prop1': pd.Series(D_t[:, 10], dtype='float'),
			# 'ini_targets_prop1': pd.Series(D_t[:, 11], dtype='float'),
			# 'time_cut': pd.Series(D_t[:, COL_time_cut])
			}
		)
	else:
		'''Diffs'''
		# D_t[:, 1] += np.random.uniform(low=-0.005, high=0.005, size=len(D_t[:, 1]))  # NOT NEEDED AS LONG AS 1 ROW USED
		X = pd.DataFrame({
			'elo_diff': pd.Series(D_t[:, 1], dtype='float'),  # TODO ELO DIFFERENCE
			'actions': pd.Series(D_t[:, 2], dtype='float'),
			'ini_objs_diff': pd.Series(D_t[:, 3], dtype='float'),
			'subjects': pd.Series(D_t[:, 4], dtype='float'),
			'objects': pd.Series(D_t[:, 5], dtype='float'),
			'time_cut': pd.Series(D_t[:, 7]),  # completely useless for this, as it should be
			'elo_avg': pd.Series(D_t[:, 10])
		})

	# fn = X.columns.values
	feature_names = list(X.columns)
	cn = ['won', 'lost']

	m = RandomForestClassifier(n_estimators=150, max_depth=10, oob_score=True)

	'''No cv'''
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.999, random_state=2, shuffle=False)
	m.fit(X_train, y_train.values.ravel())
	y_pred = m.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print("TIME_CUT: " + str(TIME_CUT) + " Mean accuracy: " + str(accuracy))

	'''Visualize'''
	# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), dpi=800)
	# aa = tree.plot_tree(m.estimators_[0], feature_names=feature_names, class_names=cn, filled=True,
	#                     fontsize=4, precision=1, rounded=True)
	# fig.savefig('rf_individualtree.png')

	'''cv'''
	# cv_results = cross_validate(m, X, y.values.ravel(), cv=10, verbose=1)
	# print("TIME_CUT: " + str(TIME_CUT) + " Mean accuracy: " + str(np.mean(cv_results['test_score'])))

	'''Feature importance
	IDEA: These can be plotted with std (but probably not that interesting)'''
	importances = m.feature_importances_  # following feature_names
	std = np.std([tree.feature_importances_ for tree in m.estimators_], axis=0)

	result_table[i, 0] = np.rint(TIME_CUT * 10)
	result_table[i, 1] = accuracy

	# 0 diff onlyIni
	# result_table[i, 2] = importances[0]  # actions
	# result_table[i, 3] = importances[1]  # ini_objs_prop_diff
	# result_table[i, 4] = importances[2]  # subjects
	# result_table[i, 5] = importances[3]  # objects

	# # 1: diff With Elo
	# result_table[i, 2] = importances[0]  # elo_diff
	# result_table[i, 3] = importances[1]  # actions
	# result_table[i, 4] = importances[2]  # ini_objs_diff
	# result_table[i, 5] = importances[3]  # subjects
	# result_table[i, 6] = importances[4]  # objects
	# result_table[i, 7] = importances[5]  # elo_avg

	# # 2: comb onlyIni
	# result_table[i, 2] = importances[0]  # ini_actions_prop0
	# result_table[i, 3] = importances[1]  # ini_objs0
	# result_table[i, 4] = importances[2]  # ini_objs_prop0
	# result_table[i, 5] = importances[3]  # ini_targets_prop0
	# result_table[i, 6] = importances[4]  # ini_actions_prop1
	# result_table[i, 7] = importances[5]  # ini_objs1
	# result_table[i, 8] = importances[6]  # ini_objs_prop1
	# result_table[i, 9] = importances[7]  # ini_targets_prop1

	# # 3: comb with Elo
	# result_table[i, 2] = importances[0]  # elo0
	# result_table[i, 3] = importances[1]  # ini_actions_prop0
	# result_table[i, 4] = importances[2]  # ini_objs0
	# result_table[i, 5] = importances[3]  # ini_objs_prop0
	# result_table[i, 6] = importances[4]  # ini_targets_prop0
	# result_table[i, 7] = importances[5]  # elo1
	# result_table[i, 8] = importances[6]  # ini_actions_prop1
	# result_table[i, 9] = importances[7]  # ini_objs1
	# result_table[i, 10] = importances[8]  # ini_objs_prop1
	# result_table[i, 11] = importances[9]  # ini_targets_prop1

	# # 4: comb p pOnlyIni
	# result_table[i, 2] = importances[0]  # ini_actions_prop0
	# result_table[i, 3] = importances[1]  # ini_objs0
	# result_table[i, 4] = importances[2]  # ini_objs_prop0
	# result_table[i, 5] = importances[3]  # ini_targets_prop0

	# 5: comb pIniElo
	# result_table[i, 2] = importances[0]  # elo0
	# result_table[i, 3] = importances[1]  # ini_actions_prop0
	# result_table[i, 4] = importances[2]  # ini_objs0
	# result_table[i, 5] = importances[3]  # ini_objs_prop0
	# result_table[i, 6] = importances[4]  # ini_targets_prop0
	# result_table[i, 7] = importances[5]  # elo1

	# OBS extend zeros

	# feature_names = list(X.columns)
	# forest_importances = pd.Series(importances, index=feature_names)
	# fig, ax = plt.subplots()
	# forest_importances.plot.bar(yerr=std, ax=ax)
	# ax.set_title("Feature importances using MDI")
	# ax.set_ylabel("Mean decrease in impurity")
	# fig.tight_layout()
	# plt.show()
	# break

# np.save(PATH_OUT + 'r.npy', result_table)
# np.savetxt(PATH_OUT + 'temp.csv', result_table, delimiter=",")

