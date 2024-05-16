
import numpy as np

file_names = [2, 3, 4, 5, 6, 10, 11, 12, 13, 14]

_1 = np.load('./data_proc/D{}0.npy'.format(str(file_names[0])))
_1 = _1[np.where((_1[:, 1] > 0) & (_1[:, 7] > 0))[0], :]

for i in range(1, len(file_names)):

    _2 = np.load('./data_proc/D{}0.npy'.format(str(file_names[i])), allow_pickle=True)
    _2 = _2[np.where((_2[:, 1] > 0) & (_2[:, 7] > 0))[0], :]

    _1 = np.concatenate((_1, _2), axis=0)

    asdf = 5

np.save('./data_proc/D_comb.npy', _1)  #  OBBBBS

# _3 = np.load('./data_proc/D3_12000.npy')

# A = A[np.where(A[:, 0] > 0)[0]]  # already done

# fix_A = np.zeros(shape=(len(A), 1))
# A = np.concatenate((A, fix_A), axis=1)
# A = np.concatenate((_1, _2))
# B = np.concatenate((A, _3))

# A = np.load('./data_proc/D3.npy')
# rows = np.where(A[:, 0] > 0)[0]
# A = A[rows, :]
# np.save('./data_proc/D60_diffs_comb.npy', A)  #  OBBBBS

asdf = 5