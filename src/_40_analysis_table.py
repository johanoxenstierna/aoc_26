
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
"""

# D = np.load('./data_proc/D_comb.npy')  # OBS DONT LOOK AT LEN HERE
# D = D[np.where((D[:, 1] > 10) & (D[:, 7] > 10))[0], :]
#
# TIME_CUT = 1.0
# D = D[np.where(D[:, 13] < TIME_CUT)[0], :]

r = np.load('./src/results/r.npy')
# r = np.load('./src/results/_0_only_ini.npy')


# TITLE = '_0_onlyIni'  # OOOOOOOOOOOOOOOOBSSSSSS
# TITLE = '_1_iniElo'  # OOOOOOOOOOOOOOOOBSSSSSS
# TITLE = '_2_onlyIniComb'  # OOOOOOOOOOOOOOOOBSSSSSS
TITLE = '_5_pIniComb'  # OOOOOOOOOOOOOOOOBSSSSSS
XLABEL = 'T_i'
YLABEL = 'Accuracy'

# # 0  # blue
# df = pd.DataFrame({
#     XLABEL: pd.Series(r[:, 0], dtype='float'),
#     'Accuracy': pd.Series(r[:, 1], dtype='float'),
#     'Actions': pd.Series(r[:, 2], dtype='float'),
#     'Subjects': pd.Series(r[:, 4], dtype='float'),
#     'Objects': pd.Series(r[:, 5], dtype='float')
# })

# # 1  black
# df = pd.DataFrame({
#     XLABEL: pd.Series(r[:, 0], dtype='float'),
#     'Accuracy': pd.Series(r[:, 1], dtype='float'),
#     'Elo_diff': pd.Series(r[:, 2], dtype='float'),
#     'Actions': pd.Series(r[:, 3], dtype='float'),
#     'Subjects': pd.Series(r[:, 5], dtype='float'),
#     'Objects': pd.Series(r[:, 6], dtype='float'),
#     'Elo_avg': pd.Series(r[:, 7], dtype='float')
# })

# # 2 orange
# df = pd.DataFrame({
#     XLABEL: pd.Series(r[:, 0], dtype='float'),
#     'Accuracy': pd.Series(r[:, 1], dtype='float'),
#     'Actions0': pd.Series(r[:, 2], dtype='float'),
#     'ini_objs0': pd.Series(r[:, 3], dtype='float'),
#     'Subjects0': pd.Series(r[:, 4], dtype='float'),
#     'Objects0': pd.Series(r[:, 5], dtype='float'),
#     'Actions1': pd.Series(r[:, 6], dtype='float'),
#     'ini_objs1': pd.Series(r[:, 7], dtype='float'),
#     'Subjects1': pd.Series(r[:, 8], dtype='float'),
#     'Objects1': pd.Series(r[:, 9], dtype='float')
# })

# # 3 green
# df = pd.DataFrame({
#     XLABEL: pd.Series(r[:, 0], dtype='float'),
#     'Accuracy': pd.Series(r[:, 1], dtype='float'),
#     'Elo0': pd.Series(r[:, 2], dtype='float'),
#     'Actions0': pd.Series(r[:, 3], dtype='float'),
#     'ini_objs0': pd.Series(r[:, 4], dtype='float'),
#     'Subjects0': pd.Series(r[:, 5], dtype='float'),
#     'Objects0': pd.Series(r[:, 6], dtype='float'),
#     'Elo1': pd.Series(r[:, 7], dtype='float'),
#     'Actions1': pd.Series(r[:, 8], dtype='float'),
#     'ini_objs1': pd.Series(r[:, 9], dtype='float'),
#     'Subjects1': pd.Series(r[:, 10], dtype='float'),
#     'Objects1': pd.Series(r[:, 11], dtype='float')
# })

# # 4 magenta
# df = pd.DataFrame({
#     XLABEL: pd.Series(r[:, 0], dtype='float'),
#     'Accuracy': pd.Series(r[:, 1], dtype='float'),
#     'Actions0': pd.Series(r[:, 2], dtype='float'),
#     'ini_objs0': pd.Series(r[:, 3], dtype='float'),
#     'Subjects0': pd.Series(r[:, 4], dtype='float'),
#     'Objects0': pd.Series(r[:, 5], dtype='float'),
# })

# 5 brown
df = pd.DataFrame({
    XLABEL: pd.Series(r[:, 0], dtype='float'),
    'Accuracy': pd.Series(r[:, 1], dtype='float'),
    'Elo0': pd.Series(r[:, 2], dtype='float'),
    'Actions0': pd.Series(r[:, 3], dtype='float'),
    'ini_objs0': pd.Series(r[:, 4], dtype='float'),
    'Subjects0': pd.Series(r[:, 5], dtype='float'),
    'Objects0': pd.Series(r[:, 6], dtype='float'),
    'Elo1': pd.Series(r[:, 7], dtype='float'),
})

fig, ax0 = plt.subplots(figsize=(7, 5))
ax_acc = sns.lineplot(data=df, x='T_i', y=YLABEL, color='brown')

ax_acc.set_ylim([0.5, 0.85])
plt.title(TITLE, fontsize=15)
plt.xlabel(XLABEL, fontsize=15)
plt.ylabel(YLABEL, fontsize=15)
plt.legend(loc='upper left', title=TITLE, title_fontsize=14, fontsize=14)


np.save('./src/results/' + TITLE + '.npy', r)
np.savetxt('./src/results/' + TITLE + '.csv', r, delimiter=",")
plt.savefig('./src/results/' + TITLE + '_acc')

plt.show()

YLABEL = 'Importance'

fig, ax0 = plt.subplots(figsize=(7, 5))
# ax_importances = sns.lineplot(data=df[['Actions', 'Subjects', 'Objects']])  # 0
# ax_importances = sns.lineplot(data=df[['Actions', 'Subjects', 'Objects', 'Elo_diff', 'Elo_avg']])  # 1
# ax_importances = sns.lineplot(data=df[['Actions0', 'Subjects0', 'Objects0',
#                                        'Actions1', 'Subjects1', 'Objects1']])  # 2
# ax_importances = sns.lineplot(data=df[['Elo0', 'Actions0', 'Subjects0', 'Objects0',
#                                        'Elo1', 'Actions1', 'Subjects1', 'Objects1']])  # 3
# ax_importances = sns.lineplot(data=df[['Actions0', 'Subjects0', 'Objects0']])  # 4
ax_importances = sns.lineplot(data=df[['Elo0', 'Actions0', 'Subjects0', 'Objects0', 'Elo1']])  # 4

plt.title(TITLE, fontsize=15)
plt.xlabel(XLABEL, fontsize=15)
plt.ylabel(YLABEL, fontsize=15)
plt.legend(loc='upper right', title=TITLE, title_fontsize=14, fontsize=14)

plt.savefig('./src/results/' + TITLE + '_imp')

plt.show()
