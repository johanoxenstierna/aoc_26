

import requests




from bs4 import BeautifulSoup

import requests

# # r = requests.get("https://aoe2.net/api/download/")
# r  = requests.get("https://aoe2recs.com/history")
# data = r.text
# soup = BeautifulSoup(data)
# adf = 5
# soup = BeautifulSoup(r.content, "html.parser")
# aa = soup.find_all('script')


# import dryscrape
# from bs4 import BeautifulSoup
# session = dryscrape.Session()
# session.visit("https://aoe2recs.com/history")
# response = session.body()
# soup = BeautifulSoup(response)
# soup.find(id="intro-text")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
import time
import re



driver = webdriver.Chrome()
driver.get("https://aoe2recs.com/history")
time.sleep(60)
# driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
# time.sleep(3)
# driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
# time.sleep(3)
htmlSource = driver.page_source

with open('./htmlSource.txt', 'w') as f:
	f.write(htmlSource)

# re.findall(r'"([^"]*)"', htmlSource)  # r = treat backslashes as raw char
# aa = re.findall(r'download', htmlSource)  # r = treat backslashes as raw char

# elements = driver.find_elements(By.XPATH, '//*[@href]')
# elements = driver.find_elements(By.TAG_NAME, 'a')
# all_tags = [el.get_attribute("href") for el in elements]
# print(all_tags)






adf = 5



