import pandas as pd


def mask_entities(texts, idx_row):
    new_text = []
    new_inv_text = []
    for text, idx in zip(texts, idx_row):
        ner_bounds = [ner.split('-') for ner in idx.split('|||')]
        inv_text = text
        prev_idx = 0
        for bound in reversed(ner_bounds):
            start_idx, end_idx = int(bound[0]), int(bound[1])
            text = text[:start_idx] + '<extra_id>' + text[end_idx:]
            inv_text = inv_text[:end_idx] + (' <extra_id>' if inv_text[end_idx:] else '') + (
                inv_text[prev_idx-1:] if prev_idx else '')
            prev_idx = start_idx

        if prev_idx:
            inv_text = '<extra_id>' + inv_text[prev_idx-1:]

        new_text.append(text)
        new_inv_text.append(inv_text)
    return new_text, new_inv_text


df = pd.read_csv('ner_data.csv')

df['input'], df['output'] = mask_entities(df['text'], df['ner_idx'])

# df['text_len'] = df['text'].str.split().str.len()
# df = df[df['text_len'] < 400]

df.drop(columns=['domain', 'ner', 'ner_idx', 'text_len']).to_csv(
    'train_data.csv', index=False)
