
# coding: utf-8

# In[1]:


"""
    Ce script va requêter en permanence coinmarketcap et récupère des données sur le bitcoin qu'il écrit dans 'live_bitcoin.csv'
    Each line is : ["price_usd","24h_volume_usd","market_cap_usd","available_supply","total_supply","percent_change_1h","percent_change_24h","percent_change_7d", 
    bkc["USD"]["sell"],bkc["USD"]["buy"],bkc["USD"]["15m"], date]
"""

import requests
import time
import datetime

#f_name = input("dataset name:")
f = open('live_bitcoin.csv',"a")
keys = ["price_usd","24h_volume_usd","market_cap_usd","available_supply","total_supply","percent_change_1h","percent_change_24h","percent_change_7d"]
vals = [0]*len(keys)

while True:
    data = requests.get("https://api.coinmarketcap.com/v1/ticker/bitcoin/").json()[0]
    #bstamp = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/").json() 
    bkc = requests.get("https://blockchain.info/ticker").json()
    for d in data.keys():
        if d in keys:
            indx = keys.index(d)
            vals[indx] = data[d]
    for val in vals:
        f.write(val+",")
      
    #f.write("{},{},".format(bstamp["volume"],bstamp["vwap"]))
    f.write("{},{},{}".format(bkc["USD"]["sell"],bkc["USD"]["buy"],bkc["USD"]["15m"]))
    f.write(","+datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
    f.write("\n")
    f.flush()
    print(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
    time.sleep(60)

