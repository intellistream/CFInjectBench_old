# pylint: disable=import-error
import re
import spacy
import pandas as pd

from tqdm.auto import tqdm

date_keywords = {'today', 'yesterday', 'tomorrow', 'each', 'weekday', 'tonight', 'daily', 'week',
                 'weeks', 'age', 'weekend', 'weekends', 'ago', 'last', 'past', 'prior', 'this',
                 'old', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',  'days',
                 'sunday', 'later', 'day', 'years', 'year', 'annual', 'quarter', 'upcoming', 'a',
                 'decades', '-', 'season', 'months', 'month', 'era', 'eras', 'quarterly', 'late',
                 'night', 'nights', 'birthday', 'summer', 'autumn', 'winter', 'spring', 'previous',
                 'eve', 'hours', 'seasons', 'end', 'weekly', 'next', 'quarters', 'nightly', 'over',
                 'current', 'first', 'monthly'}

num_rx = re.compile(r'^\d{1,2}$')

df = pd.read_csv('processed_data_full.csv')
# df = df.sample(n=2000, random_state=42)

spacy.prefer_gpu(0)
nlp = spacy.load("en_core_web_trf", disable=[
                 "tagger", "parser", "attribute_ruler", "lemmatizer"])

def trim_text(texts, ners_row):
    max_idxs = [max(int(ner.split('-')[1])
                    for ner in row.split('|||')) for row in ners_row]

    with tqdm(total=len(texts), unit='item') as progress_bar:
        for i, (doc, max_idx) in enumerate(zip(nlp.pipe(texts), max_idxs)):
            for sent in doc.sents:
                if sent.end_char > max_idx:
                    texts[i] = texts[i][:sent.end_char]
                    break
            progress_bar.update(1)
    return texts


def add_ner_col(texts):
    ner_col = []
    ner_idx = []
    with tqdm(total=len(texts), unit='item') as progress_bar:
        for doc in nlp.pipe(texts):
            temp_ner = []
            temp_idx = []
            for ent in doc.ents:
                if ent.label_ in ('GPE', 'DATE', 'PERSON', 'ORG', 'LOC'):
                    if ent.label_ == 'GPE' and not ent.text.isalpha():
                        continue

                    if ent.label_ in ('GPE', 'LOC') and ent.start_char == 0 and ent.text.isupper():
                        continue

                    if ent.label_ == 'DATE':
                        if ent.start_char == 0:
                            continue
                        if any(keyword in ent.text.lower() for keyword in ['-old', '-year', '-day', '-week', 'mid']) \
                                or any(keyword in ent.text.lower().split(' ') for keyword in date_keywords) \
                                or num_rx.match(ent.text):
                            continue

                    if ent.text not in temp_ner:
                        temp_ner.append(ent.text)
                        temp_idx.append(f'{ent.start_char} - {ent.end_char}')

            ner_col.append('|||'.join(temp_ner))
            ner_idx.append('|||'.join(temp_idx))
            progress_bar.update(1)
    return ner_col, ner_idx

df['ner'], df['ner_idx'] = add_ner_col(df['text'])

df = df[df['ner'] != '']

nlp.remove_pipe('ner')
nlp.add_pipe('sentencizer')
df['text'] = list(trim_text(df['text'].to_list(), df['ner_idx']))

df.to_csv('ner_data.csv', index=False)
