import pandas as pd 
import os
# #base file setup
stocks = pd.read_csv('dataminer.csv')['Ticker']
url = 'http://download.macrotrends.net/assets/php/stock_data_export.php?t='
print(stocks)



#this goes through all 1500+ stocks and adds them to seperate csv files whoes
#names are the tickers. the range must be updated to pull 200 stocks at a time 
#because the site only allows 200 pulls a month
for tic in stocks:
    if not os.path.exists('newhsd\\' + tic + '.csv'):
        url = 'http://download.macrotrends.net/assets/php/stock_data_export.php?t=' + tic
        try:
            data = pd.read_csv(url, skiprows=14)
            data.to_csv('newhsd\\' + tic + '.csv', index=False)
            print(tic)
        except:
            print("stopped at " + tic)
            break
            



# for proxy in proxies:
#     try:
#         print(requests.get('http://google.com',proxy=))
#     except:
#         print('failed')
#         continue

