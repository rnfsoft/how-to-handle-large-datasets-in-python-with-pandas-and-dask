import pandas as pd
from dask import dataframe as dd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow


# get data - pandas and dask
def get_data_pandas():
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
    return df

def get_data_dask():
    dfd = dd.read_csv('data/clickstream_data.tsv',
        delimiter='\t',
        names=['coming_from', 'article', 'referrer_type', 'n'],
        dtype={
            'referrer_type': 'category',
            'n': 'float64'
        },
        blocksize=64000000 # 64 Mb chunks
    )
    return dfd
# What is the most common source of visits for each article?
def summed_articles_pandas():
    df = get_data_pandas()
    summed_articles = df.groupby(['article', 'coming_from']).sum()
    max_n_filter = summed_articles.reset_index().groupby('article').idxmax() #returns the indices of the grouped column with max values
    summed_articles = summed_articles.iloc[max_n_filter['n']].sort_values(by='n', ascending=False).head(10)
    return summed_articles

def summed_articles_dask():
    dfd = get_data_dask()
    # summed_articles = dfd.groupby(['article', 'coming_from']).sum().reset_index().to_parquet('./data/summed_articles.parquet', engine='pyarrow')


# What percentage of visitors to a given article page have clicked on a link to get there?
def visitors_clicked_link_pandas():
    df = get_data_pandas()
    # df_article = df.loc[df['article'].isin([article])]
    df_article = df.loc[df['article'].isin(['Jehangir_Wadia'])]
    a = df_article['n'].sum()
    l= df_article.loc[df_article['referrer_type'].isin(['link']), 'n'].sum()
    return round(l/a*100, 2)

def visitors_clicked_link_dask():
    dfd = get_data_dask()
    dfd_article = dfd.loc[dfd['article'].isin(['Jehangir_Wadia'])]
    a = dfd_article['n'].sum().compute()
    l = dfd_article.loc[dfd_article['referrer_type'].isin(['link']), 'n'].sum().compute()
    return round(l/a*100, 2)

# What are the most popular articles users access from all the external search engines?
def most_popular_articles_pandas(): # Using pandas
    df = get_data_pandas()
    external_searches = df.loc[
        (df['referrer_type'].isin(['external'])&
        (df['coming_from'].isin(['other-search'])),
        ['article', 'n']
        )
    ]
    most_popular_articles = external_searches.sort_values(by='n', ascending=False).head(40)
    return most_popular_articles

def most_popular_articles_dask():
    dfd = get_data_dask()
    external_searches = dfd.loc[
        (dfd['referrer_type'].isin(['external'])&
        (dfd['coming_from'].isin(['other-search'])),
        ['article', 'n']
        )
    ]

    most_popular_articles = external_searches.nlargest(40, 'n').compute()
    return most_popular_articles


# Which links do people click on most often in a given article? 
def top_links_pandas(): # Using pandas
    df = get_data_pandas()
    
    top_links = df.loc[
        df['referrer_type'].isin(['link']),
        ['coming_from', 'article', 'n']
    ].groupby(['coming_from', 'article']).sum().sort_values(by='n', ascending=False)
    return top_links


def top_links_dask(): # Using Dask
    dfd = get_data_dask()

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
r= summed_articles_dask()
print(r)
# sns.barplot(data=r, y='article', x='n')
# plt.gca().set_ylabel('')
# plt.show()
print('Duration {}'.format(time.time()- start))



