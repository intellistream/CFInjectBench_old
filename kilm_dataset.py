# pylint: disable=import-error

import spacy
import pandas as pd
import re
import random
import torch
import math

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
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            source["input_ids"]
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        inputs = torch.tensor(source["input_ids"])
        attention_mask = torch.tensor(source["attention_mask"])

        # determine how many tokens we need to mask in total
        is_token = ~(inputs == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.float().sum() * 0.3))

        # generate a sufficient number of span lengths
        poisson_distribution = torch.distributions.Poisson(rate=3.0)
        lengths = poisson_distribution.sample(sample_shape=(num_to_mask,))
        while torch.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = torch.cat([lengths, poisson_distribution.sample(sample_shape=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = torch.argmin(torch.abs(torch.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[:idx + 1]

        # select span start indices
        token_indices = is_token.nonzero(as_tuple=False)
        span_starts = torch.randperm(token_indices.shape[0])[:lengths.shape[0]]

        # prepare mask
        masked_indices = token_indices[span_starts]
        mask = torch.full_like(inputs, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = inputs.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the extra_id tokens
        extra_ids = [self.tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)]
        extra_id_idx = 0
        inputs_with_extra_ids = inputs.clone()
        for i, row in enumerate(mask):
            span_count = 0
            for j, val in enumerate(row):
                if val:
                    inputs_with_extra_ids[i, j] = extra_ids[extra_id_idx]
                    span_count += 1
                    if span_count == lengths[extra_id_idx]:
                        extra_id_idx += 1
                        span_count = 0
                        if extra_id_idx >= len(extra_ids):
                            break

        # remove mask tokens that are not starts of spans
        to_remove = mask.bool() & mask.bool().roll(1, 1)
        new_inputs = torch.full_like(inputs_with_extra_ids, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(torch.split(inputs_with_extra_ids, split_size_or_sections=1, dim=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0:new_example.shape[0]] = new_example

        new_attention_mask = torch.full_like(attention_mask, fill_value=0)
        for i, example in enumerate(torch.split(attention_mask, split_size_or_sections=1, dim=0)):
            new_example = example[0][~to_remove[i]]
            new_attention_mask[i, 0:new_example.shape[0]] = new_example

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                   padding='max_length', truncation=True, return_tensors="pt")
        date = example_batch["date"]
        return new_inputs, new_attention_mask, targets, date

    def __getitem__(self, index):
        source_inputs, source_mask, targets, date = self.convert_to_features(self.dataset.iloc[index])

        source_ids = source_inputs.squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source_mask.squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "date": date}
