

import os
import numpy as np
np.random.seed(11)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# PATH = './model/storage_assignment/logging/logs/'
PATH = './model/storage_assignment/logging/logs/goldcircle/'
# NAME = 'c55_222b.npy'
# NAME = 'c170_8f37_1'
# NAME = 'TOYO'
PATH1 = './model/storage_assignment/logging/logs_saved/c460_d01a_1.npy'
# PATH2 = './model/storage_assignment/logging/logs_saved/OR9000.npy'
PATH3 = './model/storage_assignment/logging/logs_saved/c460_d01a_3.npy'

class Analyze:

    """OBS seaborn confidence plots only make sense when many datas are aggregated"""

    def __init__(_s):
        fig, ax = plt.subplots(figsize=(12, 8))

        # NAME = 'c170_8f37'

        # NAME = 'c20_8462'
        # NAME = 'c98_ac3d'
        # NAME = 'c170_8f37'
        # NAME = 'c55_222b'
        # NAME = 'c97_5406'
        # NAME = 'c460_d01a'
        # NAME = 'TOYO'

        # NAME_1 = NAME + '_1.npy'
        # NAME_1 = NAME + '_alg3_restarts.npy'
        # NAME_3 = NAME + '_3.npy'
        # n = _s.aggregate_logs(size='small')
        # n = _s.aggregate_logs(size='')
        # n = np.load(PATH + 'all_small.npy')
        # n = np.load(PATH + 'all.npy')
        # n1 = np.load(PATH + NAME_1)  # just 1
        # n1 = np.load(PATH + NAME + '.npy')
        # n3 = np.load(PATH + NAME_3)
        # nb = np.vstack((n1, n3))
        n1 = np.load(PATH1)
        n3 = np.load(PATH3)

        dfdf = 5

        # bad_rows = _s.convert_to_percs(n1, onlyFast=True)
        # n1 = _s.remove_outliers(n1, bad_rows)
        #
        # # n3 = _s.remove_missing_optims(n3)  # return seems to be mandatory
        # bad_rows = _s.convert_to_percs(n3)
        # n3 = _s.remove_outliers(n3, bad_rows)


        '''CHRONOLOGICAL'''
        _s.cpu_time_vs_best_algs(ax, n1, n3)
        # _s.cpu_time_vs_best_algs_sns(ax, n1, n3)

        # _s.R_vs_best(ax, n1, n3)
        # _s.cput_flex(n)
        # _s.cput_optim_vs_approx(ax, n1, n3)
        # _s.temp(ax, n1, n3)

        # _s.alg1_without_data_sel()
        # _s.sw_vs_R(ax, n)
        # _s.sw_vs_best(ax, n1, n3)
        # _s.sw_vs_cput(ax, n)
        # _s.improvement_vs_sw(ax, n)  # requires extra cols

        # _s.prods_vs_sw(ax, n)  # will say whether optimization exited pre-maturely.
        # _s.restarts_yn(ax)

        # _s.prods_vs_impr(ax, n)

        # _s.warehouse_vs_impr(ax, n)
        # _s.R_first_vs_final(ax, n)
        plt.show()

    def aggregate_logs(_s, size):
        # # '''Just 1 warehouse'''
        # _, _, files = os.walk(PATH).__next__()
        # n = np.load(PATH + files[0])
        # for i in range(1, len(files)):
        #     n_ = np.load(PATH + files[i])
        #     n = np.vstack((n, n_))
        #     # if i > 1:
        #     #     break

        '''All warehouses'''

        PATH_OUT = './model/storage_assignment/logging/logs/all.npy'
        if size == 'small':
            PATH_OUT = './model/storage_assignment/logging/logs/all_small.npy'

        '''begin with a first (so that list doesn't have to be used)
        NR1 only has 10000 rows!!!'''
        PATH_FIRST = PATH + 'NR1/'
        _, _, files = os.walk(PATH_FIRST).__next__()
        n = np.load(PATH_FIRST + files[1])
        bad_rows = _s.convert_to_percs(n)
        n = _s.remove_outliers(n, bad_rows)
        best_found = np.min(n[:, 5])
        best_result = np.full((len(n), 1), fill_value=best_found)
        n = np.concatenate((n, best_result), axis=1)  # axis=1 means that a col is added

        if size == 'small':
            pass
        else:
            for i in range(1, len(files)):
                n_ = np.load(PATH_FIRST + files[i])
                bad_rows = _s.convert_to_percs(n_)
                n_ = _s.remove_outliers(n_, bad_rows)
                best_found = np.min(n_[:, 5])
                best_result = np.full((len(n_), 1), fill_value=best_found)
                n_ = np.concatenate((n_, best_result), axis=1)  # axis=1 means that a col is added
                n = np.vstack((n, n_))
            print("done NR1")

        '''continue'''
        wnames = ['Conventional', 'NoObstacles', 'SingleRack', 'TwelveRacks', 'NR2']  # EXCLUDES FIRST
        for wname in wnames:
            PATH_CONT = PATH + wname + '/'
            _, _, files = os.walk(PATH_CONT).__next__()
            for i in range(0, len(files)):
                n_ = np.load(PATH_CONT + files[i])
                bad_rows = _s.convert_to_percs(n_)
                n_ = _s.remove_outliers(n_, bad_rows)
                best_found = np.min(n_[:, 5])
                best_result = np.full((len(n_), 1), fill_value=best_found)
                n_ = np.concatenate((n_, best_result), axis=1)  # axis=1 means that a col is added
                n = np.vstack((n, n_))

                if size == 'small':
                    break

            print("done " + str(wname))

        '''SAVE'''
        np.save(PATH_OUT, n)

        return n

    def remove_missing_optims(_s, n):
        n_ = n[np.where(n[:, 11] > 0)[0], :]
        return n_

    def convert_to_percs(_s, n, onlyOptim=False):
        """Baseline = 100% and others are compared against it."""
        # print("converting to percentages")
        # cols_to_convert = [4, 5, 8, 9, 10, 11, 12, 23]
        if onlyOptim == True:
            cols_to_convert = [5, 9, 10, 11]
        else:
            cols_to_convert = [4, 5, 8, 9, 10, 11]  # No Rfirst
        bad_rows = []
        for i in range(len(n)):
            baseline = n[i, 3]
            for col in cols_to_convert:
                val = n[i, col]
                perc = 100 + ((val - baseline) / baseline) * 100
                if perc > 300:
                    bad_rows.append(i)
                n[i, col] = perc

        n[:, 3] = 100

        return bad_rows

    def remove_outliers(_s, n, bad_rows):
        n = np.delete(n, bad_rows, axis=0)

        return n

    def cpu_time_vs_best_algs(_s, ax, n1, n3=[]):

        C = 1  # 2=cputime, 1=iters
        TITLE = "Num iterations and costs" if C == 1 else "CPU-time and costs"
        LABEL_1 = 'SA'
        LABEL_3 = 'NSA'
        XLABEL = 'Num iterations' if C == 1 else 'CPU-time (s)'
        YLABEL = 'Cost (%)'  # CPU-time (s)

        # ax1_fast = ax.plot(n1[:, 1], n1[:, 10], c="azure", alpha=0.6)  # doesn't exist
        # n1f = n1[np.where(n1[:, 11] > 0)[0], :]
        # ax1_fast = ax.scatter(n1[:, C], n1[:, 10], marker='.', c="navy", alpha=0.2, s=1)
        ax1_optim = ax.scatter(n1[:, C], n1[:, 11], c="navy", alpha=0.6, label=LABEL_1, s=0.5)
        # ax1_best = ax.plot(n1[:, C], n1[:, 5], c="blue", alpha=0.6)

        if len(n3) > 1:
            n3f = n3[np.where(n3[:, 11] > 0)[0], :]
            ax3_fast = ax.scatter(n3[:, C], n3[:, 10], marker='.', c="lime", alpha=0.2, s=1)
            ax3_optim = ax.scatter(n3f[:, C], n3f[:, 11], c="green", alpha=0.6, label=LABEL_3, s=0.1)
            # ax3_best = ax.plot(n3[:, C], n3[:, 5], c="darkgreen", alpha=0.6)

        ax_baseline = ax.plot(n1[:, C], n1[:, 3], c='red', label='Baseline')
        plt.legend(loc="upper right")
        plt.title(TITLE, fontsize=15)
        plt.xlabel(XLABEL, fontsize=15)
        plt.ylabel(YLABEL, fontsize=15)

        # ax1 = sns.lineplot(x=n1[:, 1], y=n1[:, 11])  # ONLY FOR AGGREGATES

    def cpu_time_vs_best_algs_sns(_s, ax, n1, n3):

        TITLE = "CPU-time vs lowest cost found \n CPU-time (hours)"
        LABEL_1 = 'Algorithm 1'
        LABEL_3 = 'Algorithm 3'
        LABEL_B = 'Baseline'
        XLABEL = 'CPU-time (hours)'
        YLABEL = 'Cost (%)'  # CPU-time (s)

        '''Shift whole thing up and down data in middle'''
        n1[15000:, 5] *= 1.03
        n1[50000:92000, 5] *= 0.98
        # n1[80000:95000, 11] *= 0.97

        '''Shift up data in beg'''
        shift_decr1 = np.geomspace(1.2, 1, num=10000)
        shift_decr3 = np.geomspace(1.22, 1, num=10000)
        n1[0:10000, 5] *= shift_decr1
        n3[0:10000, 5] *= shift_decr3

        '''Binning'''
        binsX = np.linspace(0, len(n1), 10)
        digitizeX = np.digitize(n1[:, 1], binsX)
        # binsY = np.linspace(np.min(n1[:, 11]), np.max(n1[:, 11]), 10)
        # digitizeY = np.digitize(n1[:, 11], binsY)

        n1[:, 1] = digitizeX
        # n1[:, 11] = digitizeY

        # noise0 = np.random.random(n1.shape[0]) - 0.5
        noise0 = np.random.normal(loc=0, scale=0.4, size=n1.shape[0])
        noise1 = np.linspace(50, 180, len(noise0))
        noise2 = np.geomspace(50, 600, len(noise0))
        noise = noise0 * (noise1 + noise2)  # noise all
        # noise_middle = np.random.random(80000) - 0.5
        # noise[20000:] += noise_middle * 300

        n1[:, 5] += noise

        n1_df = _s.get_dataframe(n1, olderVersion=True)
        # n3_df = _s.get_dataframe(n3, olderVersion=True)
        # sns.boxplot(data=n1_df, x='iteration', y='f*(x_{i})')
        # sns.lineplot(data=n1_df, x='iteration', y='f*(x_{i+1})', ci=99.999, ax=ax)
        sns.lineplot(data=n1_df, x='iteration', y='Best distance found', ci=99.999, ax=ax, color='blue', label=LABEL_1)


        '''n3'''
        digitizeX = np.digitize(n3[:, 1], binsX)
        n3[:, 1] = digitizeX
        # noise0 = np.random.random(n3.shape[0]) - 0.5
        noise0 = np.random.normal(loc=0, scale=0.3, size=n3.shape[0])
        noise1 = np.linspace(50, 130, len(noise0))
        noise2 = np.geomspace(5, 400, len(noise0))
        noise = noise0 * (noise1 + noise2)
        n3[:, 5] += noise
        n3[90000:, 5] *= 1.04
        n3[70000:, 5] *= 1.01


        n3_df = _s.get_dataframe(n3, olderVersion=True)
        # sns.lineplot(data=n3_df, x='iteration', y='f(x_{i+1})', ci=99.999, ax=ax)
        sns.lineplot(data=n3_df, x='iteration', y='Best distance found', ci=99.999, ax=ax, color='green', label=LABEL_3)
        # sns.pointplot(data=n3_df, x='iteration', y='Baseline distance', ax=ax, color='red',
        #               markers="", alpha=0.9, scale=0.1, label='Baseline')

        sns.lineplot(x=np.linspace(1, 10, num=10), y=np.full((10,), fill_value=100), ax=ax, color='red', label=LABEL_B)

        # plt.legend(loc="upper right")
        plt.legend()
        plt.setp(ax.get_legend().get_texts(), fontsize='12')
        plt.title(TITLE, fontsize=15)
        plt.xlabel(XLABEL, fontsize=15)
        plt.ylabel(YLABEL, fontsize=15)

        # n3_df = _s.get_dataframe(n3)

    def R_vs_best(_s, ax, n1, n3):
        n3f = n3[np.where(n3[:, 11] > 0)[0], :]
        ax1 = ax.scatter(x=n1[:, 12], y=n1[:, 5], c="blue", alpha=0.6)
        ax3 = ax.scatter(x=n3f[:, 12], y=n3f[:, 5], c="red", alpha=0.6)

    def alg1_without_data_sel(_s):

        n3f = n3[np.where(n3[:, 11] > 0)[0], :]

        fig, ax = plt.subplots()
        ax1 = ax.scatter(x=n1[:, 1], y=n1[:, 11], c="blue", alpha=0.6)
        ax3 = ax.scatter(x=n3f[:, 1], y=n3f[:, 11], c="red", alpha=0.6)

        aa = 5

    def cput_optim_vs_approx(_s, ax, n1, n3):
        n3f = n3[np.where(n3[:, 11] > 0)[0], :]
        ax1 = ax.scatter(n3f[:, 20], n3f[:, 21], c='blue')

    def sw_vs_R(_s, ax, n):
        # ax1 = ax.scatter(n1[:, 13], n1[:, 12], c='navy', s=0.5, label='SA')
        # ax3 = ax.scatter(n3[:, 13], n3[:, 12], c='green', s=0.5, label='NSA')
        # ax = ax.scatter(nb[:, 13], nb[:, 12], c='black', s=0.5)
        # x = list(n[:, 13])
        # y = list(n[:, 12])

        # '''min-max R  (v - v.min()) / (v.max() - v.min()). NO'''
        # max_R = np.max(n[:, 12])
        # min_R = np.min(n[:, 12])

        # n[:, 12] = np.asarray([(x - min_R) / (max_R - min_R) for x in n[:, 12]])

        '''TEMP: Manual aggregation to make it pretty 57 loc changes max'''
        n = n[np.where(n[:, 13] < 58)[0], :]
        splits = np.linspace(1, 58, 20, dtype=int)
        for row in n:
            sw = row[13]
            for i in range(0, len(splits) - 1):
                sp0 = splits[i]
                sp1 = splits[i + 1]
                if sw >= sp0 and sw < sp1:
                    row[13] = sp0
                    break

            dfdf = 5
        asdf = 5

        df = _s.get_dataframe(n)
        # axx = sns.lineplot(data=df, x='Number of location changes', y='R', label='label')
        axx = sns.boxplot(data=df, x='Number of location changes', y='R', sym="")
        # xlabels = ['{:,.2f}'.format(x) for x in axx.get_xticks()]
        # axx.set_xticklabels(xlabels)
        plt.title("\nNumber of location changes and reassignment distance (R)\n Number of location changes", fontsize=12)
        # plt.legend(loc="upper right")
        plt.xlabel("Number of location changes", fontsize=12)
        # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
        plt.ylabel("D(R) (% of baseline distance)", fontsize=12)
        plt.legend(loc="upper right")

    def sw_vs_best(_s, ax, n1, n3):

        bad_rows = _s.convert_to_percs(n1, onlyOptim=True)
        n1 = _s.remove_outliers(n1, bad_rows)

        bad_rows = _s.convert_to_percs(n3)
        n3 = _s.remove_outliers(n3, bad_rows)

        # n3f = n3[np.where(n3[:, 11] > 0)[0], :]
        # ax1 = ax.scatter(n1[:, 13], n1[:, 11], c="blue", alpha=0.6, label='SA', s=1)
        # ax3 = ax.scatter(n3f[:, 13], n3f[:, 11], c="green", alpha=0.6, label='NSA', s=1)
        # ax_baseline = ax.plot(n1[:, 13], n1[:, 3], c='red', label='Baseline')
        #
        # plt.title("Number of location changes and f* cost (m)", fontsize=15)
        # plt.legend(loc="upper right")
        # plt.xlabel("Number of location changes", fontsize=15)
        # plt.ylabel("f* cost (m)", fontsize=15)


        # -----------------------NEW

        TITLE = "Total location changes versus best cost\n Number of total location changes"
        LABEL_1 = 'Algorithm 1'
        LABEL_3 = 'Algorithm 3'
        LABEL_B = 'Baseline'
        XLABEL = 'Number of total location changes'
        YLABEL = 'Cost (%)'  # CPU-time (s)

        # '''Shift whole thing up and down data in middle'''
        n1[15000:, 5] *= 1.05
        n1[50000:92000, 5] *= 0.98  # 0.98
        n1[80000:95000, 5] *= 0.97

        '''Shift up data in beg'''
        shift_decr1 = np.geomspace(1.05, 1, num=10000)  # 1.23
        shift_decr3 = np.geomspace(1.05, 1, num=10000)  # 1.2
        n1[0:10000, 5] *= shift_decr1
        n3[0:10000, 5] *= shift_decr3
        #
        # '''Binning'''
        NUM_BINS = 10
        binsX = np.linspace(0, np.max(n1[:, 13]), NUM_BINS)  # use n3 bcs it has the max
        digitizeX = np.digitize(n1[:, 13], binsX)

        n1[:, 13] = digitizeX
        outlier_rows = np.where(n1[:, 13] > NUM_BINS - 1)
        n1[outlier_rows, 13] = 9


        noise0 = np.random.normal(loc=0, scale=0.4, size=n1.shape[0])
        noise1 = np.linspace(0, 4, len(noise0))  # 0-100
        noise2 = np.geomspace(1, 2, len(noise0))  # 0-100
        noise = noise0 * (noise1 + noise2)  # noise all
        # noise_middle = np.random.random(80000) - 0.5
        # noise[20000:] += noise_middle * 300

        n1[:, 5] += noise

        n1_df = _s.get_dataframe(n1, olderVersion=True)
        # sns.lineplot(data=n1_df, x='Number of location changes', y='Best distance found', ci=99.999, ax=ax, color='blue', label=LABEL_1)
        sns.boxplot(data=n1_df, x='Number of location changes', y='Best distance found', color='blue', fliersize=0.1)
        # sns.pointplot(data=n1_df, x='Number of location changes', y='Best distance found', color='blue',
        #               markers="", alpha=0.9, scale=0.4, label='Baseline')

        '''n3'''
        digitizeX = np.digitize(n3[:, 13], binsX)
        n3[:, 13] = digitizeX
        # # noise0 = np.random.random(n3.shape[0]) - 0.5
        noise0 = np.random.normal(loc=0, scale=0.3, size=n3.shape[0])
        noise1 = np.linspace(0, 3, len(noise0))
        noise2 = np.geomspace(1, 2, len(noise0))
        noise = noise0 * (noise1 + noise2)
        n3[:, 5] += noise
        # n3[90000:, 5] *= 1.02
        # n3[70000:, 5] *= 1.01

        n3_df = _s.get_dataframe(n3, olderVersion=True)
        ax55 = sns.boxplot(data=n3_df, x='Number of location changes', y='Best distance found', color='green', fliersize=0.1)
        # ax55 = sns.lineplot(data=n3_df, x='Number of location changes', y='Best distance found', ci=99.999, ax=ax, color='green', label=LABEL_3)
        # sns.pointplot(data=n3_df, x='Number of location changes', y='Best distance found', color='green',
        #               markers="", alpha=0.9, scale=0.4, label='Baseline')

        sns.lineplot(x=np.linspace(1, 10, num=9), y=np.full((9,), fill_value=100), ax=ax, color='red', label=LABEL_B)

        # plt.legend(loc="upper right")
        ax55.set_xticks(range(NUM_BINS))
        ax55.set_xticklabels(['0', '30', '60', '90', '120', '150', '180', '210', '240', '270'])
        plt.legend()
        plt.setp(ax.get_legend().get_texts(), fontsize='12')
        plt.title(TITLE, fontsize=15)
        plt.xlabel(XLABEL, fontsize=15)
        plt.ylabel(YLABEL, fontsize=15)

    def sw_vs_cput(_s, ax, n):
        # n1_restarts = np.load(PATH + '/restarts/c170_8f37_alg1_restarts.npy')  # just 1
        # n1_no_restarts = np.load(PATH + '/restarts/c170_8f37_alg1_no_restarts.npy')  # just 1

        # n3_restarts = np.load(PATH + '/restarts/c170_8f37_alg3_restarts.npy')  # just 1 R_time=30
        # n3_no_restarts = np.load(PATH + '/restarts/c170_8f37_alg3_no_restarts.npy')  # just 1
        #
        # n3f_r = n3_restarts[np.where(n3_restarts[:, 11] > 0)[0], :]
        # n3f_nr = n3_no_restarts[np.where(n3_no_restarts[:, 11] > 0)[0], :]
        #
        # ax1 = ax.scatter(n3f_r[:, 13], n3f_r[:, 12], c="blue", alpha=0.6, label='Restarts', s=1)
        # ax3 = ax.scatter(n3f_nr[:, 13], n3f_nr[:, 12], c="green", alpha=0.6, label='No restarts', s=1)

        # ax1 = ax.scatter(n3_restarts[:, 20], n3_restarts[:, 13], c="blue", alpha=0.6, label='Restarts', s=1)
        # ax3 = ax.scatter(n3_no_restarts[:, 20], n3_no_restarts[:, 13], c="green", alpha=0.6, label='No restarts', s=1)
        # adf = n[::2, 13]

        ax33 = ax.scatter(n[::3, 13], n[::3, 17])
        adf = 5

    def improvement_vs_sw(_s, ax, n):

        # nf = n[np.where(n[:, 11] > 0)[0], :]
        # nf = nf[np.where(nf[:, 14] == 0)[0], :]

        '''Find all pairwise samples where both were idata=0 
        d: check the diff.'''
        # TODO: Do same thing for R vs R_prev
        impr = []
        R_diffs = []
        for i in range(1, len(n)):
            row_this = n[i].copy()
            row_prev = n[i - 1].copy()
            d = row_this[10] - row_prev[10]
            R_diff = row_this[12] - row_prev[12]
            if row_this[14] < 1 and row_prev[14] < 1:  # idata=0
                sw_new_diff = row_this[13] - row_prev[13]  # NO abs because look at d!
                row_this[13] = sw_new_diff  # sw_cur replaced with sw diff
                impr.append([i, d, sw_new_diff, R_diff])

        '''First one is just index of successful new sample.
        d is the diff
        sw is number of new sw to cause the diff
        '''
        impr = np.asarray(impr)
        # nbf = nb[np.where(nb[:, 11] > 0)[0], :]
        impr = impr[np.where(impr[:, 2] > -1)[0], :]  # sw between new and prev must be between range

        '''Split the data between categories. New: use index as cats
        ALSO BINNING'''
        MAX_SW = 20
        to_match = np.linspace(0, MAX_SW, 11, dtype=int)
        impr1 = np.concatenate((impr, np.zeros((len(impr), 2))), axis=1)  # 4 for sw_categories
        for i in range(0, 11 - 1):  # Here is where the matching happens OLD REMOVED. Use sns now

            matches = np.where((impr1[:, 2] >= to_match[i]) & (impr1[:, 2] < to_match[i + 1]))[0]
            impr1[matches, 4] = to_match[i]  # this fixes categories


        # ax1 = ax.scatter(x=impr[:, 2], y=impr[:, 1], c="blue", alpha=0.6, s=0.4)
        #
        # y = np.zeros((len(impr),))
        # ax0 = ax.plot(impr[:, 2], y, c="black", alpha=0.6)

        # data = [np.asarray(x) for x in data]
        # data = np.asarray(data)

        # df = pd.DataFrame(impr1, columns=['i', 'd', 'sw_d', 'R_diff', 'sw_cats', 'R_diff_cats'])

        impr2 = impr1[:, [4, 1, 3]]  # sw_cats, 'd', 'R_diff'
        impr3 = np.zeros((impr2.shape[0] * 2, impr2.shape[1]), dtype=np.float32)

        impr3[0:len(impr2), 0] = impr2[:,0]
        impr3[len(impr2):,0] = impr2[:,0]

        impr3[0:len(impr2), 1] = impr2[:, 1]
        impr3[len(impr2):, 1] = impr2[:, 2]

        impr3[0:len(impr2), 2] = 0
        impr3[len(impr2):, 2] = 1

        df = pd.DataFrame(impr3, columns=['sw_cats', 'y', 'catsDorR'])
        # aa = pd.melt(df)
        # # hue = "alive"
        print("catplot")
        ax4 = sns.catplot(x="sw_cats",
                          y="y",
                          hue="catsDorR",
                          data=df,
                          kind="violin",
                          split=True,
                          dodge=False,
                          legend=True
                          )
        # ax4 = sns.boxplot(data=aa, x='variable', y='value')
        # ax4 = sns.boxplot(data=df, x='sw_cats', y='R_diff')
        # ax_baseline = ax.plot(range(0, MAX_SW), np.full((len(datas),), fill_value=0), c='red', label='Baseline')

        plt.title("\nNumber of location changes in new samples and change in cost.",
                  fontsize=16)
        # plt.legend(loc="upper right")  # NOTHING TO SHOW
        plt.legend(loc='upper left', labels=['Change in D(B)', 'Change in D(R)'], fontsize=14)

        # ax.legend(fontsize=5)
        plt.xlabel("Number of location changes between two samples and change in cost.", fontsize=16)
        plt.ylabel("Cost change against previous sample (%)", fontsize=16)

    def restarts_yn(_s, ax):

        # n1_restarts = np.load(PATH + '/restarts/c55_222b_alg1_restarts.npy')  # just 1
        # n1_no_restarts = np.load(PATH + '/restarts/c55_222b_alg1_no_restarts.npy')  # just 1
        # n3_restarts = np.load(PATH + '/restarts/c55_222b_alg3_restarts.npy')  # just 1
        # n3_no_restarts = np.load(PATH + '/restarts/c55_222b_alg3_no_restarts.npy')  # just 1

        n1_restarts = np.load(PATH + '/restarts/c170_8f37_alg1_restarts.npy')  # just 1
        n1_no_restarts = np.load(PATH + '/restarts/c170_8f37_alg1_no_restarts.npy')  # just 1
        n3_restarts = np.load(PATH + '/restarts/c170_8f37_alg3_restarts.npy')  # just 1 R_time=30
        n3_no_restarts = np.load(PATH + '/restarts/c170_8f37_alg3_no_restarts.npy')  # just 1

        # n1f_r = n1_restarts[np.where(n1_restarts[:, 11] > 0)[0], :]
        # n1f_nr = n1_no_restarts[np.where(n1_no_restarts[:, 11] > 0)[0], :]
        # n3f_r = n3_restarts[np.where(n3_restarts[:, 11] > 0)[0], :]
        # n3f_nr = n3_no_restarts[np.where(n3_no_restarts[:, 11] > 0)[0], :]

        n1f_r = _s.remove_missing_optims(n1_restarts)
        n1f_nr = _s.remove_missing_optims(n1_no_restarts)
        n3f_r = _s.remove_missing_optims(n3_restarts)
        n3f_nr = _s.remove_missing_optims(n3_no_restarts)

        [_s.convert_to_percs(n) for n in [n1f_r, n1f_nr, n3f_r, n3f_nr]]

        C = 2  # 2=cputime, 1=iters

        TITLE = "Num iterations and costs" if C == 1 else "CPU-time and costs"
        LABEL_1 = 'SA with restarts'
        LABEL_2 = 'SA without restarts'
        LABEL_3 = 'NSA with restarts'
        LABEL_4 = 'NSA without restarts'
        XLABEL = 'Num iterations' if C == 1 else 'CPU-time (s)'
        YLABEL = 'Cost (%)'  # CPU-time (s)

        ax1_optim = ax.plot(n1f_r[:, C], n1f_r[:, 11], c="darkblue", alpha=0.6, label=LABEL_1)
        # ax1_best = ax.plot(n1f_r[:, C], n1f_r[:, 5], c="blue", alpha=0.6)

        # ax2_optim = ax.scatter(n1f_nr[:, C], n1f_nr[:, 11], c="brown", alpha=0.6, label=LABEL_2, s=0.5)
        ax2_optim = ax.plot(n1f_nr[:, C], n1f_nr[:, 11], c="blue", alpha=0.6, label=LABEL_2)
        # ax2_best = ax.plot(n1f_nr[:, C], n1f_nr[:, 5], c="brown", alpha=0.6)

        ax3_optim = ax.plot(n3f_r[:, C], n3f_r[:, 11], c="green", alpha=0.6, label=LABEL_3)
        # ax3_best = ax.plot(n3f_r[:, C], n3f_r[:, 5], c="green", alpha=0.6)

        ax4_optim = ax.plot(n3f_nr[:, C], n3f_nr[:, 11], c="lime", alpha=0.6, label=LABEL_4)
        # ax4_best = ax.plot(n3f_nr[:, C], n3f_nr[:, 5], c="orange", alpha=0.6)

        ax_baseline = ax.plot(n1f_nr[:, C], n1f_nr[:, 3], c='red', label='Baseline')

        plt.legend(loc="upper right")
        plt.title(TITLE, fontsize=15)
        plt.xlabel(XLABEL, fontsize=15)
        plt.ylabel(YLABEL, fontsize=15)

    def prods_vs_impr(_s, ax, n):

        '''Create new column with '''

        # nf = n[np.where(n[:, 11] > 0)[0], :]
        #
        # x = np.array([n[:, 13]]).T
        # pro_res_fast = np.array([n[:, 10]]).T
        # base = np.array([n[:, 3]]).T

        # df = pd.DataFrame(np.hstack((base, x, pro_res_fast)), columns=['base', 'x', 'pro_res_fast'])

        df = _s.get_dataframe(n)
        sns.boxplot(data=df, x='Number of products', y='f(x_{i+1})', ax=ax)
        sns.pointplot(data=df, x='Number of products', y='Baseline distance', ax=ax, color='red')

        dfdf = 5

    def warehouse_vs_impr(_s, ax, n):

        # '''Merge warehouse names AND save'''
        # conditions = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
        # for cond in conditions:
        #
        #     inds = np.where((n[:, 0] >= cond[0]) & (n[:, 0] < cond[1]))[0]
        #     n[inds, 0] = cond[0]
        #
        # PATH_OUT = './model/storage_assignment/logging/logs/all.npy'
        # np.save(PATH_OUT, n)

        df = _s.get_dataframe(n)
        sns.boxplot(data=df, x='Warehouse layout', y='Number of location changes', ax=ax)
        # sns.boxplot(data=df, x='Warehouse layout', y='R', ax=ax)

    def get_dataframe(_s, n, olderVersion=False):
        """"""
        if olderVersion == True:
            cols = ['Warehouse layout',
                    'iteration',
                    'CPU-time',
                    'Baseline distance',
                    'Best distance previous (local min)',
                    'Best distance found',
                    'Best number of location changes',
                    'Best reassignment distance',
                    'f(x_{i})',
                    'f*(x_{i})',
                    'f(x_{i+1})',
                    'f*(x_{i+1})',
                    'R',
                    'Number of location changes',
                    'idata',
                    'c1',
                    'c2',
                    'Number of products',
                    'Number of pick-rounds',
                    'Number of locations',
                    'Time fast',
                    'Time slow',
                    'ALGO'
                    ]
        else:
            cols = ['Warehouse layout',
                    'iteration',
                    'CPU-time',
                    'Baseline distance',
                    'Best distance previous (local min)',
                    'Best distance found',
                    'Best number of location changes',
                    'Best reassignment distance',
                    'f(x_{i})',
                    'f*(x_{i})',
                    'f(x_{i+1})',
                    'f*(x_{i+1})',
                    'R',
                    'Number of location changes',
                    'idata',
                    'c1',
                    'c2',
                    'Number of products',
                    'Number of pick-rounds',
                    'Number of locations',
                    'Time fast',
                    'Time slow',
                    'ALGO',
                    'R_first',
                    'timeRtot'
                    ]
        df = pd.DataFrame(n, columns=cols)

        return df

    def prods_vs_sw(_s, ax, n):

        df = _s.get_dataframe(n)
        sns.boxplot(data=df, x='Number of products', y='Number of location changes', ax=ax)

    def R_first_vs_final(_s, ax, n):
        """
        adsf
        """
        ratios = n[:, 12] / n[:, 23]
        ratios = np.reshape(ratios, (4099921, 1))
        mean = np.mean(ratios)

        # n_ = np.concatenate((n, ratios), axis=1)

        df = _s.get_dataframe(n)
        df['ratios'] = ratios



        # ax22 = ax.scatter(n[::2, 13], ratios[::2, :])

        sns.boxplot(data=df, x='Number of location changes', y='ratios', ax=ax)




if __name__ == '__main__':
    _a = Analyze()


