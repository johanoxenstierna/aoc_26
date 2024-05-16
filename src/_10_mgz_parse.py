

import os
# import io
# import json
import time
# import zipfile
# import glob
import shutil
# from mgz_aoc_clone.mgz import header, fast
# from mgz_aoc_clone.mgz.summary import Summary
from mgz_aoc_clone.mgz.model import parse_match, serialize

from src.utils import *

R = 14

PATH_UNZIP_FOLDER = './r/unzip_folder_{}/'.format(str(R))
PATH_FILES_TO_BE_REMOVED = './r/files_to_be_removed_{}.json'.format(str(R))
PATH_OUT = './data_proc/D{}0.npy'.format(str(R))

# PATH_IN_ZIP = './r/{}_z/'.format(str(R))
PATH_IN_ZIP = './r_test/2_z/'.format(str(R))
# PATH_IN_ZIP = '/media/johan/0E45-EEA5/{}_z/'.format(str(R))

_, _, file_names_zip = os.walk(PATH_IN_ZIP).__next__()  # THESE ARE ARCHIVES. NOT SORTED!!!!
file_names_zip.sort()  # good to make sure reruns are same

LEN_DATA = len(file_names_zip)
D = np.zeros(shape=(LEN_DATA * 20, 18), dtype=np.float32)  # THIS IS TWICE AS BIG AS SHOULD BE
cur_row = 0
time0 = time.time()  # 4000 s to do 6000 files
time_cut_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # OBS MUST START ON 0.1

with open('./profiles.json', 'r') as f:
    profiles = json.load(f)

tot_time_parsing = 0
tot_time_infering = 0

print("R: " + str(R))

files_to_be_removed = []
D_row = 0
iii = 0  # attempts at saving games
ii = 0  # all the files

for ii in range(0, len(file_names_zip)):

    file_name_zip = file_names_zip[ii]
    print("ii: " + str(ii) + "  " + str(file_name_zip))

    '''The zip record is unzipped and placed in the UNZIP folder, which is rebuilt with a single file each iteration'''
    if os.path.isdir(PATH_UNZIP_FOLDER):
        shutil.rmtree(PATH_UNZIP_FOLDER)

    os.mkdir(PATH_UNZIP_FOLDER)

    file_name_unzip = file_name_zip[:-4]
    os.system('unzip -qq -j ' + PATH_IN_ZIP + file_name_zip + ' -d ' + PATH_UNZIP_FOLDER + file_name_unzip)

    '''THE PARSING AND INFERRING OF THE RECORD'''
    # try:

    '''file_name_s should never fail given that grabs.py file name convention is correct'''
    file_name_s = file_name_unzip.replace('_', ' ')
    file_name_s = file_name_s.replace('.', ' ')
    file_name_s = file_name_s.split()

    if file_name_s[2] != 'aoe2record':
        print("Wrong file ending: " + str(file_name_s[2]))
        files_to_be_removed.append(file_name_zip)  # this is archive name
        continue

    try:  # the file name that will be used
        PROF_ID_SAVE = int(file_name_s[0])
    except:
        PROF_ID_SAVE = 0

    try:
        MATCH_TIME = int(file_name_s[1])  # not used
    except:
        MATCH_TIME = 0

    try:
        t0_parsing = time.time()
        with open(PATH_UNZIP_FOLDER + file_name_unzip + '/' + file_name_unzip, 'rb') as f:
            m, ps = parse_match(f)
    except Exception as e:
        print("Parsing error")
        print(e)
        if str(e) == 'could not find player end':
            files_to_be_removed.append(file_name_zip)
        continue

    t1_parsing = time.time()
    tot_time_parsing += (t1_parsing - t0_parsing)

    if len(m.players) > 2:
        print("!!!More than two PLAYERS")
        files_to_be_removed.append(file_name_zip)
        continue

    flag_prof_not_found = get_profiles(PROF_ID_SAVE, ps, profiles)
    if flag_prof_not_found == True:
        print("Profile not found: " + str(PROF_ID_SAVE) + ", but file kept")
        continue

    if len(m.actions) < 500:
        print("Too few actions: " + str(file_name_zip))
        files_to_be_removed.append(file_name_zip)
        continue

    flag_not_found_loser = get_ps_actions(m.actions, ps)
    if flag_not_found_loser == True:
        print("Could not deduce loser")
        files_to_be_removed.append(file_name_zip)
        continue

    try:
        get_tc_coords(ps)
    except:
        print("get_tc_coords error")
        files_to_be_removed.append(file_name_zip)
        continue

    flag_not_found_aggr = set_aggr_actions(ps)
    if flag_not_found_aggr == True:
        print("No aggr actions found for file_name: " + str(file_name_zip))
        files_to_be_removed.append(file_name_zip)
        continue

    t0_infering = time.time()
    for i, TIME_CUT_R in enumerate(time_cut_ratios):
        t0_ratio, t_end = compute_initiative(ps, TIME_CUT_R)  # this is a row in D
        D_row = infer_and_push_to_D(D_row, D, ps, TIME_CUT_R, PROF_ID_SAVE, MATCH_TIME, t0_ratio, t_end)
    t1_infering = time.time()
    tot_time_infering += (t1_infering - t0_infering)

    iii += 1  # A GAME WAS SAVED. not same as ii

    if iii % 50 == 0:
        print("\nSAVING ===================")
        print("MEAN COL2: " + str(np.mean(D[:iii, 2])) + "  COL8: " + str(np.mean(D[:iii, 8])))
        print("MEAN COL3: " + str(np.mean(D[:iii, 3])) + "  COL9: " + str(np.mean(D[:iii, 9])))
        print("MEAN COL4: " + str(np.mean(D[:iii, 4])) + "  COL10: " + str(np.mean(D[:iii, 10])))
        print("MEAN COL5: " + str(np.mean(D[:iii, 5])) + "  COL11: " + str(np.mean(D[:iii, 11])))
        print("tot_time_parsing: " + str(tot_time_parsing))
        print("tot_time_infering: " + str(tot_time_infering))
        print(str(ii) + " out of " + str(len(file_names_zip)) + " done")

        np.save(PATH_OUT, D)

        with open(PATH_FILES_TO_BE_REMOVED, 'w') as f:
            json.dump(files_to_be_removed, f, indent=2)
        print("==================DONE SAVING\n")

    # except Exception as e:
    #     print("GENERAL FAILURE")
    #     print(e)
    #     files_to_be_removed.append(file_name_zip)  # this is archive name


with open(PATH_FILES_TO_BE_REMOVED, 'w') as f:
    json.dump(files_to_be_removed, f, indent=2)

time1 = time.time() - time0
np.save(PATH_OUT, D)
print("took " + str(time1) + " to do " + str(ii) + " games")
'''TODO: remove missing rows + print how many missing'''

if os.path.isdir(PATH_UNZIP_FOLDER):
    shutil.rmtree(PATH_UNZIP_FOLDER)
    os.mkdir(PATH_UNZIP_FOLDER)

# D = D[np.where(D[:, 0] > 0)[0], :]

sss = 4