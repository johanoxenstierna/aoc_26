
import io
import zipfile
import requests
import json

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from grabs_utils import *

"""TEMP"""
# profiles_postproc()
""""""

"""OOOOOOBBBBS CUR PAGE ON SITE: 221. 
TODO: Info on opponent profile id not stored. -> Cannot infer how many total profiles were used"""
# [Fs] guischmitt
NEW = False
if NEW == False:
	with open('./profiles.json', 'r') as f:  # 3961 ELOS, lowest one: 1400
		profiles = json.load(f)
else:
	profiles = {}

driver = webdriver.Firefox()
driver.get("https://www.ageofempires.com/stats/ageiide")
time.sleep(2)

for i in range(0, 400):  # pages

	# input("Press Enter to continue...")

	wait = WebDriverWait(driver, timeout=5)
	wait.until(EC.visibility_of_element_located((By.ID, "global_leaderboard")))
	tbody = driver.find_element(by=By.ID, value="global_leaderboard")

	wait = WebDriverWait(driver, timeout=5)
	wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "leaderboard__row")))
	trs = tbody.find_elements(by=By.CLASS_NAME, value="leaderboard__row")

	name = ""
	for tr in trs:
		try:
			p_info = tr.find_elements(by=By.CLASS_NAME, value="leaderboard__cell")
			ELO = int(p_info[1].text)
			name = p_info[2].text

			href = p_info[2].find_element(By.CLASS_NAME, value="leader__link").get_attribute('href')
			href = href.replace('?', ' ')
			href = href.replace('=', ' ')
			href = href.replace('&', ' ')
			href_s = href.split()
			profile_id = href_s[2]

			try:
				profile_id = int(profile_id)
			except:
				print("could not convert to int profile id: " + str(profile_id))

			if name not in profiles:
				# profiles[name] = {'profile_id': profile_id,
				#                         'name': name,
				#                         'ELO': ELO,
				#                         }
				'''TODO: fix ELO_dates and ELO = average'''

			print("profile_id: " + str(profile_id) + " name: " + str(name) + " ELO: " + str(ELO))
		except:
			print("profile could not be added   name: " + str(name))

	print('page: ' + str(i) + ' len profiles: ' + str(len(profiles)))
	# pagination_class = driver.find_element(by=By.CLASS_NAME, value="leaderboard-section__pagination")
	# pagination_ = pagination_class.find_element(by=By.CLASS_NAME, value="pagination")
	pagination = driver.find_element(by=By.CLASS_NAME, value="pagination")
	next_button = pagination.find_element(by=By.CLASS_NAME, value="pagination__control.--right")
	# aaa = pagination.find_element(by=By., value="pagination__control --right")

	with open('./profiles.json', 'w') as f:
		json.dump(profiles, f, indent=2)

	driver.execute_script("arguments[0].click();", next_button)
	time.sleep(2)




# driver.execute_script("arguments[0].click();", button)
# time.sleep(2)
#
# '''TODO get ELO'''
#
# tbody2 = driver.find_elements(by=By.CLASS_NAME, value='icon_matchReplay')
# download_link = tbody2[0].get_attribute('href')
# response = requests.get(download_link, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
#
# try:
# 	replay_zip = zipfile.ZipFile(io.BytesIO(response.content))
# 	replay = replay_zip.read(replay_zip.namelist()[0])
# except Exception as e:
# 	print(e)
# 	continue
#
# with open('./r/test.aoe2record', 'wb') as f:
# 	f.write(replay)
#
