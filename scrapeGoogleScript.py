#!/usr/bin/env python3

import pandas as pd
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import sqlite3 as lite

key = 'AIzaSyAkP1-7wumxY6mWPFaXdq9VFgCbET9Mp1k'
#local database:
database = 'googDrugData.db'

quantityThreshold = 10
startingEntry = 4045

#We download a database of drugs we're interested in classifying.
#df = pd.read_csv('https://query.data.world/s/3re1owtak1hk4thb4nlfp5g1k') #First 100 entries
df = pd.read_csv('https://query.data.world/s/1jckku39d0p78m8jm1gnvww5t') #Complete data

#with lite.connect(database) as con:
#   df.to_sql(name = 'medicare_data',con = con)


for i,drug in enumerate(df['Generic Name'][startingEntry:]):
    # Create Data frames to contain our search results
    df_classes = pd.DataFrame(columns = ['Generic_Name','url','Drug_Class_Data_raw','Drug_Class_Data_clean'])
    df_drugs = pd.DataFrame(columns = ['Generic_Name','url','Drug_Data_raw','Drug_Data_clean'])
    
    print('-'*30)
    print("%d of %d, scraping for: "%(i,df.shape[0])+drug)
    params = {'key':key,'cx':'001774473651733095884:tgnlbf2ef_8','q':drug}

    # First, we log into google and search for pages relevant to the drug:
    try:
        resp = requests.get('https://www.googleapis.com/customsearch/v1',params=params)
    except RequestException as e:
        print(e)
        
    # Next we scrape each of those pages for information relevant to classifying the drug:
    if 'items' in resp.json():
        for j,item in enumerate(resp.json()['items']):
            if 'link' in item:
                print("Scraping page %d of %d"%(j,len(resp.json()['items'])))
                try:
                    resp2 = requests.get(item['link'])
                    bsObj = BeautifulSoup(resp2.content,'lxml')
                    drugClassList = bsObj.findAll(itemtype='http://schema.org/DrugClass')
                    drugList = bsObj.findAll(itemtype='http://schema.org/Drug')
                    print("Found %d drug class references and %d drug references"%(len(drugClassList),len(drugList)))
                    #Save all drug classes encountered
                    for dClass in drugClassList[:quantityThreshold]:
                        bsCleanObj = BeautifulSoup(dClass.prettify(),'lxml')
                        df_classes.loc[df_classes.shape[0]]=[drug,item['link'],dClass.prettify(),bsCleanObj.text]
                        
                    #Save all drugs encountered
                    for dClass in drugList[:quantityThreshold]:
                        bsCleanObj = BeautifulSoup(dClass.prettify(),'lxml')
                        df_drugs.loc[df_classes.shape[0]]=[drug,item['link'],dClass.prettify(),bsCleanObj.text]
                except Exception as e:
                    print(e)
    if df_classes.shape[0] > 0:
        print('{:.100}'.format(' '.join(df_classes['Drug_Class_Data_clean'][0].replace('\n',' ').split())))
                    
    #For debugging
    #print(df_classes)
    #print(df_drugs)
    
    #Save the Data:
    with lite.connect(database) as con:
        df_classes.to_sql(name = 'goog_drug_class_data', con = con, if_exists = 'append')
        df_drugs.to_sql(name = 'goog_drug_data', con = con, if_exists = 'append')