
import numpy as np

PATH_IN = './data_proc/D50.npy'
PATH_OUT = './data_proc/D51.npy'
D = np.load(PATH_IN)
D = D[np.where(D[:, 1] > 0)[0], :]

# 1 minus on all cols CUZ AGGR ACTIONS WERE WRONG - ONLY CLOSE TO OWN TC
D[:, 2] = 1 - D[:, 2]
D[:, 3] = 1 - D[:, 3]
D[:, 4] = 1 - D[:, 4]

D[:, 7] = 1 - D[:, 7]
D[:, 8] = 1 - D[:, 8]
D[:, 9] = 1 - D[:, 9]

# logs
# D[:, 3] = np.log(D[:, 3])
# D[:, 8] = np.log(D[:, 8])

np.save(PATH_OUT, D)


