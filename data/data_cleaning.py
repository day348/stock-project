import pandas as pd
import os
from progressbar import ProgressBar

stock_tickers = pd.read_csv('data/stock_names.csv')['Ticker']
count = 0
pbar = ProgressBar(0)
for tic in stock_tickers:
    skip = False
    try:
        temp = pd.read_csv('data/normalized_data/' + tic + '.csv')
    except:
        print('skipping ' + tic)
        skip = True
        pass
    if skip == False:
        if len(temp.index) <20:
            os.remove('data/normalized_data/' + tic + '.csv')
            os.remove('data/testing/'+ tic + '.csv')
            os.remove('data/training/'+ tic + '.csv')
            print('removed ' + tic)
            count = count + 1
print('removed ', count, ' stocks for sucking')