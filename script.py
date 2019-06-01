import pandas as pd
from dask import dataframe as dd
import time


# Which links do people click on most often in a given article? 
def top_links_pandas(): # Using pandas
    df = pd.read_csv('data/clickstream_data.tsv',
        delimiter='\t',
        names=['coming_from', 'article', 'referrer_type', 'n'],
        dtype={
            'referrer_type': 'category'
            }
    )
    df.dropna(inplace=True)
    df['n'] = df['n'].astype('uint32') # changing float64 to uint32 memory usage: 3.1+ MB --> memory usage: 2.8+ MB
    df = df.iloc[:100000]
    
    top_links = df.loc[
        df['referrer_type'].isin(['link']),
        ['coming_from', 'article', 'n']
    ].groupby(['coming_from', 'article']).sum().sort_values(by='n', ascending=False)
    return top_links


def top_links_dask(): # Using Dask
    dfd = dd.read_csv('data/clickstream_data.tsv',
        delimiter='\t',
        names=['coming_from', 'article', 'referrer_type', 'n'],
        dtype={
            'referrer_type': 'category',
            'n': 'float64'
        },
        blocksize=64000000 # 64 Mb chunks
    )

    top_links_grouped_dask = dfd.loc[
        dfd['referrer_type'].isin(['link']),
        ['coming_from', 'article', 'n']
    ].groupby(['coming_from', 'article'])

    store = pd.HDFStore('./data/clickstream_store.h5')
    top_links_dask = top_links_grouped_dask.sum().nlargest(20, 'n')
    
    store.put('top_links_dask', top_links_dask.compute(), format='table', data_columns=True) 
    # ImportError: HDFStore requires PyTables, "No module named 'tables'" problem importing
    # pip install --upgrade tables
    # https://www.programcreek.com/python/example/101333/pandas.HDFStore
    chuck = store.select('top_links_dask')
    store.close()
    return chuck


start = time.time()
r= top_links_dask()
print(r)
print('Duration {}'.format(time.time()- start))



