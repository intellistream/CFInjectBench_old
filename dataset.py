# pylint: disable=import-error

import spacy
import pandas as pd

from torch.utils.data import Dataset

nlp = spacy.load('en_core_web_sm', disable=['ner'])


class CKLDataset(Dataset):
    def __init__(self, data, type_path, tokenizer, args, mix=False):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path

        self.dataset = pd.DataFrame(data)

        self.input_length = args.max_input_length
        self.output_length = args.max_output_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch):
        if self.args.dataset == 'wiki':
            if self.type_path == 'train':
                doc = nlp(example_batch['corpus'])
                input_ = ''

                for idx, token in enumerate(doc):
                    if (token.dep_ == 'ROOT' and doc[idx+1].dep_ in ('prep', 'agent', 'dep', 'det')) or (token.dep_ in ('dobj', 'attr') and doc[idx+1].dep_ == 'prep') or (token.dep_ == 'pobj' and doc[idx+1].dep_ == 'ROOT'):
                        input_ = " ".join(
                            [token.text for token in doc[:doc[idx+2].i]])
                        # output_text = " ".join([token.text for token in doc[doc[idx+2].i:]]) # mlm
                        break

                    if (token.dep_ == 'ROOT' and doc[idx+1].dep_ == 'compound') or (token.dep_ == 'prep' and doc[idx+1].dep_ in ('pobj', 'compound')) or (token.dep_ in ('ROOT', 'prep') and doc[idx+1].dep_ == 'nmod'):
                        input_ = " ".join(
                            [token.text for token in doc[:doc[idx+1].i]])
                        # output_text = " ".join([token.text for token in doc[doc[idx+1].i:]]) # mlm
                        break

                    if token.dep_ == 'ROOT':
                        input_ = " ".join(
                            [token.text for token in doc[:doc[idx+1].i]])
                        # output_text = " ".join([token.text for token in doc[doc[idx+1].i:]]) # mlm
                        break

                if input_ == '':
                    input_ = doc.text[:-2]

                target_ = example_batch['corpus']
                # target_ = '<extra_id_0> ' + output_text # mlm
                # input_ = input_ + ' <extra_id_0>'
            else:
                input_ = example_batch['query'].split('_X_')
                if len(input_[0]) == 0:
                    input_ = input_[1].strip()
                else:
                    input_ = input_[0].strip()
                target_ = input_ + ' ' + example_batch['answer'] + '.'

                if len(input_) == 0:
                    print(input_)

                # input_ = example_batch['query'].replace('_X_', '<extra_id_0>') # mlm
                # target_ = '<extra_id_0> ' + example_batch['answer'] + '.'

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                  padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                   padding='max_length', truncation=True, return_tensors="pt")
        date = example_batch["date"]
        return source, targets, date

    def __getitem__(self, index):
        source, targets, date = self.convert_to_features(self.dataset.iloc[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "date": date}
