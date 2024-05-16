import copy

import numpy as np
import json


def get_profiles(profile_id_save, ps, profiles):
	"""
	If a profile cannot be found then ELO cannot be obtained,
	so match is useless in that case.
	"""
	grabs_match_found = False
	flag_not_found = False
	for p_name, p in ps.items():

		if p_name not in profiles:
			print("Profile not found: " + str(p_name))
			flag_not_found = True
			break

		profile_ = profiles[p_name]

		if profile_['profile_id'] == profile_id_save:
			grabs_match_found = True

		p['profile'] = profile_

		gg = 5

	if grabs_match_found == False:
		print("grabs_match_found = False ")

	return flag_not_found


def get_winner(ps):

	flag_not_found = False


def get_ps_actions(actions, ps):

	losing_player = None
	for p_id, p in ps.items():
		p['actions'] = []
		p['queues'] = []
		p['winner'] = None

	for a in actions:

		try:
			_ = ps[a.player.name]  # just to check that the name can actually be parsed
		except:
			flag_not_found_loser = True
			return flag_not_found_loser

		ps[a.player.name]['actions'].append(a)

		if a.type.name == 'DE_QUEUE':
			ps[a.player.name]['queues'].append(a)

		temp = actions[int(len(actions) * 0.9):]

		if a.type.name == 'RESIGN':
			losing_player = a.player.name

	if losing_player == None:
		flag_not_found_loser = True
		return flag_not_found_loser
	else:
		for p_id, p in ps.items():
			if p_id == losing_player:
				p['winner'] = 0
			else:
				p['winner'] = 1
		flag_not_found_loser = False
		return flag_not_found_loser


def get_tc_coords(ps):

	'''get tc id'''


	for p_id, p_val in ps.items():

		p_val['tc'] = {'position': {}, 'instance_id': None}

		qactions_within_60 = []
		for a in p_val['queues']:
			if a.timestamp.seconds < 60:
				qactions_within_60.append(a)

		'''check always same id'''
		try:
			tc_id = qactions_within_60[0].payload['object_ids'][0]
		except:
			raise Exception("TC problem")

		for a in qactions_within_60:
			if a.payload['object_ids'][0] != tc_id:
				raise Exception("TC mismatch")

		for obj in p_val['objects']:
			if obj['instance_id'] == tc_id:
				p_val['tc']['instance_id'] = tc_id
				p_val['tc']['position'] = obj['position']

		if p_val['tc']['instance_id'] == None:
			raise Exception("TC not found")


def set_aggr_actions(ps):
	"""
	Coordinates are required
	NOTHING TIME BASED HERE except lower bound
	"""

	aggr_error = False
	num_not_found_aggr = 0 # 0, 1 or 2

	for p_id, p in ps.items():

		p['aggr_actions'] = []
		p['has_aggr_actions'] = True
		# p['num_non_aggr_actions'] = 0

		'''get tc of opponent'''
		p_ids = ps.keys()
		p_opponent_id = None
		for _p_id in p_ids:
			if _p_id != p_id:
				p_opponent_id = _p_id
		tc_opp = ps[p_opponent_id]['tc']
		pos_tc_opp = np.asarray([tc_opp['position']['x'], tc_opp['position']['y']])
		pos_tc_own = np.asarray([p['tc']['position']['x'], p['tc']['position']['y']])

		'''just filter on type and num objects'''
		aggr_actions = []
		flag_found_first = False
		for a in p['actions']:

			# Position ALWAYS present in MOVE and ORDER. Need > 1 to avoid scout
			if a.type.name in ['MOVE', 'ORDER', 'BUILD', 'WALL', 'PATROL', 'DE_ATTACK_MOVE'] and \
				len(a.payload['object_ids']) > 1 and \
				a.timestamp.seconds > 60:  # last condition not very important # TODO: should belong to p'

				'''condition on distances to tc'''
				pos_a = np.asarray([a.position.x, a.position.y])

				d_own_tc = np.linalg.norm(pos_a - pos_tc_own) + 0.00001
				d_opp_tc = np.linalg.norm(pos_a - pos_tc_opp) + 0.00001
				ratio_own_opp = d_own_tc / d_opp_tc

				if ratio_own_opp > 1:  # d_own_tc is more than d_opp_tc

					'''the first order on enemy half marks beginning of aggression'''
					if flag_found_first == False and a.type.name == 'ORDER':
						flag_found_first = True

					if flag_found_first == True:
						aggr_actions.append(a)
				# else:
				# 	p['num_non_aggr_actions'] += 1  # ITS EIATHER THIS OR AGGR_ACTIONS +1

		if len(aggr_actions) > 1: # and p['num_non_aggr_actions'] > 1:
			p['aggr_actions'] = aggr_actions
		else:
			p['has_aggr_actions'] = False
			num_not_found_aggr += 1

	if num_not_found_aggr == 2:  # neither of the players were aggressive
		aggr_error = True

	return aggr_error


def compute_initiative(ps, TIME_CUT_R):
	"""
	aggr_actions have NOT been filtered based on TIME_CUT at this point
	THATS WHATS DONE HERE
	OBS this is where ps is set
	"""

	flag_error_ini = False

	'''get first times of aggr'''
	time_firstest = 9999999
	time_lastest = 0

	t_end = 1  # to avoid /0

	for p_id, p in ps.items():

		try:  # SHOULD BE MOVED TO EARLIER FUNCTION
			t_end_ = p['actions'][-1].timestamp.seconds
			if t_end_ > t_end:
				t_end = copy.deepcopy(t_end_)
		except:
			print("timestamp could not be found for one of the last actions")

		if p['has_aggr_actions'] == True:
			_time_first = p['aggr_actions'][0].timestamp.seconds
			_time_last = p['aggr_actions'][-1].timestamp.seconds

			if _time_first < time_firstest:
				time_firstest = _time_first

			if _time_last > time_lastest:
				time_lastest = _time_last

	tot_time_aggr = time_lastest - time_firstest
	T0_RATIO = time_firstest / t_end
	TIME_CUT_UPPER_S = time_firstest + int(TIME_CUT_R * tot_time_aggr)
	TIME_CUT_LOWER_S = time_firstest + int((TIME_CUT_R - 0.1) * tot_time_aggr)

	for p_id, p in ps.items():

		'''Defaults for a player, i.e., no aggression at all
		OBS. Check how many games are won with these. 
		Proportions are here wrt OWN. Against opp is done later
		'''
		# p['ini_t0_ratio'] = T0_RATIO  returned instead!
		p['ini_actions_prop'] = 0  # THE LARGER THE MORE INI
		p['ini_objs'] = 0  # THE LARGER THE MORE INI
		p['ini_objs_prop'] = 0  # THE LARGER THE MORE INI
		p['ini_targets_prop'] = 0  # THE LARGER THE MORE INI
		p['ini_group_size_avg'] = 0  # need to remove later if 0

		if p['has_aggr_actions'] == False:
			# print("p[has_aggr_actions] == False")
			continue

		aggr_actions_t = []
		for a in p['aggr_actions']:
			if (a.timestamp.seconds > TIME_CUT_LOWER_S) and (a.timestamp.seconds < TIME_CUT_UPPER_S):
				aggr_actions_t.append(a)
			if a.timestamp.seconds > TIME_CUT_UPPER_S:
				break

		'''All actions (not just aggr)'''
		all_num_actions_t = 0
		all_unique_object_ids = []
		all_unique_target_ids = []
		for a in p['actions']:
			if (a.timestamp.seconds > TIME_CUT_LOWER_S) and (a.timestamp.seconds < TIME_CUT_UPPER_S):
				all_num_actions_t += 1
				if 'object_ids' in a.payload:
					for obj_id in a.payload['object_ids']:
						if obj_id not in all_unique_object_ids:
							all_unique_object_ids.append(obj_id)

				if 'target_id' in a.payload and a.type.name in ['ORDER', 'SPECIAL']:
					if a.payload['target_id'] not in all_unique_target_ids:
						all_unique_target_ids.append(a.payload['target_id'])

			if a.timestamp.seconds > TIME_CUT_UPPER_S:
				break

		'''np.unique is done on them afterwards'''
		ini_actions = len(aggr_actions_t)
		ini_objs = []
		ini_targets = []
		ini_group_sizes = []

		for a in aggr_actions_t:
			# ini_times.append(a.timestamp.seconds)
			ini_objs.extend(a.payload['object_ids'])
			if 'target_id' in a.payload and a.type.name in ['ORDER', 'SPECIAL']:
				ini_targets.append(a.payload['target_id'])
			ini_group_sizes.append(len(a.payload['object_ids']))

		'''here writings to p'''
		if len(aggr_actions_t) > 0:
			# ini_times_avg = np.mean(ini_times) - time_firstest
			# ini_times_avg_rat = ini_times_avg / tot_time_aggr
			p['ini_actions_prop'] = ini_actions / all_num_actions_t
			assert(p['ini_actions_prop'] < 1.000001)

		if len(ini_objs) > 0:
			ini_unique_objs = np.unique(np.asarray(ini_objs))
			p['ini_objs'] = len(ini_unique_objs)
			p['ini_objs_prop'] = len(ini_unique_objs) / len(all_unique_object_ids)
			assert (p['ini_objs_prop'] < 1.000001)

			'''Now same thing needed to get all the objs that were not close to enemy'''
			# p['ini_objs_prop'] = np.log(len(ini_objs_tot))
		if len(ini_targets) > 0:
			ini_unique_targets = np.unique(np.asarray(ini_targets))
			p['ini_targets_prop'] = len(ini_unique_targets) / len(all_unique_target_ids)
			assert (p['ini_targets_prop'] < 1.000001)
		if len(ini_group_sizes) > 0:
			p['ini_group_size_avg'] = np.mean(ini_group_sizes)

	return T0_RATIO, t_end

	# for p_id, p in ps.items():  #
	# '''if rat = 1 loop over all actions (not just early) and compute data for that.
	# REMOVED. It is cheating to look forward here.
	# '''
	# # if p['ini_times_avg_rat'] == 1:
	# # 	a_times = []
	# # 	for a in p['aggr_actions']:
	# # 		# if a.timestamp.seconds < time_cutoff:
	# # 		a_times.append(a.timestamp.seconds)
	# # 	# a_times_avg = np.mean(a_times) - time_firstest
	# # 	# p['ini_times_avg_rat'] = a_times_avg / tot_time_aggr
	# #
	# #
	# # 	ini_actions = []
	# # 	for a in p['aggr_actions']:
	# # 		ini_actions.append(a)
	# #
	# # 	ini_times = []
	# # 	ini_objs = []
	# # 	ini_targets = []
	# # 	ini_group_sizes = []
	# #
	# # 	for a in ini_actions:
	# # 		ini_times.append(a.timestamp.seconds)
	# # 		ini_objs.extend(a.payload['object_ids'])
	# # 		if 'target_id' in a.payload:
	# # 			ini_targets.append(a.payload['target_id'])
	# # 		ini_group_sizes.append(len(a.payload['object_ids']))
	# #
	# # 	if len(ini_times) > 0:
	# # 		ini_times_avg = np.mean(ini_times) - time_firstest
	# # 		ini_times_avg_rat = ini_times_avg / tot_time_aggr
	# # 		p['ini_times_avg_rat'] = ini_times_avg_rat
	# #
	# # 	if len(ini_objs) > 0:
	# # 		ini_objs_tot = np.unique(np.asarray(ini_objs))
	# # 		p['ini_objs_tot'] = len(ini_objs_tot)
	# # 	if len(ini_targets) > 0:
	# # 		ini_targets = np.unique(np.asarray(ini_targets))
	# # 		p['ini_targets'] = len(ini_targets)
	# # 	if len(ini_group_sizes) > 0:
	# # 		p['ini_group_size_avg'] = np.mean(ini_group_sizes)
	# #
	# # 	if p['ini_times_avg_rat'] > 0.99999:
	# # 		print("liuyliuyi")
	# # 		# raise Exception("Asdfasdfasdf")


def infer_and_push_to_D(D_row, D, ps, TIME_CUT_R, PROF_ID_SAVE, MATCH_TIME, t0_ratio, t_end):

	"""Select first occurences of aggs types

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

	'''check that D_row is valid'''
	if D_row > 0:
		if D[D_row - 1, 1] < 10 or D[D_row - 1, 7] < 10:
			raise Exception("Elo missing prev row")
		if D[D_row, 1] > 0.1 or D[D_row, 7] > 0.1:
			raise Exception("Elo set on D_row when it shouldnt")

	'''This sets 1 row'''
	cols = []
	for p_id, p in ps.items():

		try:
			_ = p['profile']['ELO']
		except:
			raise Exception("cant add ELO from profile")

		'''player specific cols'''
		if p['winner'] > 0.5:  # winner columns
			cols = [1, 2, 3, 4, 5, 6]
		elif p['winner'] <= 0.5:  # loser columns
			cols = [7, 8, 9, 10, 11, 12]

		D[D_row, cols[0]] = p['profile']['ELO']
		D[D_row, cols[1]] = p['ini_actions_prop']
		D[D_row, cols[2]] = p['ini_objs']
		D[D_row, cols[3]] = p['ini_objs_prop']
		D[D_row, cols[4]] = p['ini_targets_prop']
		D[D_row, cols[5]] = p['ini_group_size_avg']

	'''non player specific rows'''
	D[D_row, 13] = TIME_CUT_R
	D[D_row, 14] = PROF_ID_SAVE
	D[D_row, 15] = MATCH_TIME
	D[D_row, 16] = t0_ratio
	D[D_row, 17] = t_end

		# D[D_row, 16] = p['ini_t0_ratio']  # MOVED    SAME FOR BOTH CURRENTLY

	D_row += 1

	# '''PEND DELDEPR First get the ELO, won and times'''
	# for p_id, p in ps.items():
	# 	try:
	# 		D[ii, 0] = p['profile']['ELO']
	# 	except:
	# 		raise Exception("cant add ELO from profile")
	#
	# 	D[ii, 1] = p['winner']
	# 	D[ii, 6] = profile_id_save
	#
	# 	# for a in p['aggr_actions']:
	# 	#
	# 	# 	if len(a.payload['object_ids']) > 2 and D[ii, 2] == 0:
	# 	# 		D[ii, 2] = a.timestamp.seconds
	# 	#
	# 	# 	if len(a.payload['object_ids']) > 4 and D[ii, 3] == 0:
	# 	# 		D[ii, 3] = a.timestamp.seconds
	# 	#
	# 	# 	if len(a.payload['object_ids']) > 8 and D[ii, 4] == 0:
	# 	# 		D[ii, 4] = a.timestamp.seconds
	#
	# 	D[ii, 2] = p['ini_times_avg_rat']
	# 	D[ii, 3] = p['ini_objs_tot']
	# 	D[ii, 4] = p['ini_targets']
	# 	D[ii, 5] = p['ini_group_size_avg']
	#
	# 	ii += 1


	return D_row




