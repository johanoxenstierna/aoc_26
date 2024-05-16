
import numpy as np
import matplotlib.pyplot as plt
import pandas
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

D = np.load('./data_proc/D_comb.npy')
D = D[np.where(D[:, 13] > 0.95)[0], :]  # ONLY NEED 1 per match
aa = np.max(D[:, 17])
df = pandas.DataFrame(D)

fig, ax0 = plt.subplots(figsize=(6, 5))

ax = sns.histplot(data=df, x=df[17])

plt.show()

