# http://berkeleyearth.lbl.gov/country-list/
# SCRAPE 

import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
response = requests.get('http://berkeleyearth.lbl.gov/country-list/')
soup = BeautifulSoup(response.text)
table_rows = soup.find(attrs={'class':'table table-condensed table-hover'}).find_all('tr')
columns = ['Year', 'Month', 'Monthly Anomaly', 'Monthly Uncertainty', 'Annual Anomaly', 'Annual Uncertainty',
           'Five-Year Anomaly', 'Five-Year Uncertainty', 'Ten-Year Anomaly', 'Ten-Year Uncertainty',
           'Twenty-Year Anomaly', 'Twenty-Year Uncertainty']          
FILE = 'all_country_temp_data.csv'
PATH = 'data'
if not FILE in os.listdir(path=PATH):
    df_list = []
    for i in table_rows:
        if i.find('a'):
            url = i.a.get('href')
            name = i.td.text
            print(name)
            country_page = BeautifulSoup(requests.get(url).text)
            txt_file = country_page.find(attrs={'class':'caption text-center'}).find_all('a')[1].get('href')
            df = pd.read_csv(txt_file, sep='\s+', comment='%', names=columns)
            df['Country'] = name
            df_list.append(df) 
    df = pd.concat(df_list)
    df.to_csv(PATH + '/' + FILE)
else:
    df = pd.read_csv(PATH + '/' + FILE, index_col=0)