import pandas as pd
import requests

from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

def search_c4(ners_col):
    def process_url(ners):
        query = ' AND '.join(f'"{ner}"' for ner in ners.split('|||'))
        url = f"https://c4-search.apps.allenai.org/?q={query}"
        response = session.get(url)
        
        soup = BeautifulSoup(response.text, 'lxml')
        search_result_texts = soup.select('ul.search-result-list div.search-result-text')

        return 1 if not search_result_texts else 0

    session = requests.Session()
    tags = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_url, ners_col)
        tags.extend(results)

    return tags

df = pd.read_csv('summary_ners.csv')

df['flags'] = search_c4(df['ner'])
df.to_csv('flagged_summary.csv', index=False)