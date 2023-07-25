import pandas as pd


def filter_ners(txt_col, ners_col):

    ners_col = [ners.split('|||') for ners in ners_col]
    ners_col = [[ner for ner in ners if ner in txt]
                for ners, txt in zip(ners_col, txt_col)]
    return ['|||'.join(ner for ner in ners if not any(
            ner != other_ner and ner in other_ner for other_ner in ners)) for ners in ners_col]


df = pd.read_csv('summary.csv')

df['ner'] = filter_ners(df['summary'], df['ner'])
df = df[df['ner'] != '']

df.to_csv('summary_ners.csv', index=False)
