import re
from bs4 import BeautifulSoup

# t = '"addftttyy" df'
# a = re.findall(r'addfttt|df', t)
# b = re.findall(r'"([^"]*)"', t)

with open('./htmlSource.txt', 'r') as f:
	t = f.read()

soup = BeautifulSoup(t)
aa = soup.find_all('a', href=True)
ab = aa.get

gg = 5
