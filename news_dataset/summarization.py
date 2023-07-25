import pandas as pd

from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm.auto import tqdm

device = 'cuda:1'

model = BartForConditionalGeneration.from_pretrained(
    'Yale-LILY/brio-cnndm-cased').to(device)
tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-cased')

batch_size = 96

df = pd.read_csv('ner_data.csv')

summaries = []

with tqdm(total=len(df), unit='item') as progress_bar:
    for idx in range(0, len(df), batch_size):
        batch = df.iloc[idx:idx+batch_size]

        inputs = tokenizer(batch['text'].to_list(), max_length=350, padding=True,
                           return_tensors="pt", truncation=True).to(device)
        summary_ids = model.generate(inputs["input_ids"])
        summaries += tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        progress_bar.update(batch_size)


df['summary'] = summaries

df = df.drop(columns=['domain', 'text', 'ner_idx'])
df.to_csv('summaries.csv', index=False)
