# pylint: disable=import-error

import pandas as pd

from torch.utils.data import Dataset


class CKLDataset(Dataset):
    def __init__(self, mini_batch, type_path, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path

        if 't5' in args.model_name_or_path:
            self.model_type = 'T5'
        # elif 'gpt2' in args.model_name_or_path:
        #     self.model_type = 'GPT2'

        self.dataset = pd.DataFrame(mini_batch)

        self.input_length = args.max_input_length
        self.output_length = args.max_output_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
        if self.args.dataset == 'recentnews':
            if self.model_type == 'GPT2':
                input_ = example_batch['original']
                target_ = example_batch['original']
            elif self.model_type == 'T5':
                input_ = example_batch['input'] if isinstance(
                    example_batch['input'], str) else ''
                target_ = example_batch['output'] if isinstance(
                    example_batch['output'], str) else ''

        # elif self.args.dataset == 'wikitext103':
        #     input_ = example_batch['original']
        #     target_ = example_batch['original']
        # # evaluation
        else:
            if self.args.dataset == 'invariantlama':
                if self.model_type == 'GPT2':
                    input_pre = example_batch['input']
                    for index, word in enumerate(input_pre.split()):
                        if word == '<extra_id_0>':
                            input_pre = ' '.join(input_pre.split()[:index])
                            break
                    if self.type_path == 'train':
                        input_ = input_pre + ' ' + \
                            example_batch['output'] + '.'
                        target_ = input_pre + ' ' + \
                            example_batch['output'] + '.'
                    else:
                        input_ = input_pre
                        ground_truth_ = example_batch['output']
                        target_ = input_pre + ' ' + \
                            example_batch['output'] + '.'
                elif self.model_type == 'T5':
                    input_ = example_batch['input']
                    target_ = example_batch['output']
            elif self.args.dataset == 'updatedlama':
                input_ = example_batch['statement']
                target_ = example_batch['new_answer']
            elif self.args.dataset in ['newlama', 'newlama_easy']:
                input_ = example_batch['statement']
                target_ = example_batch['answer']
            else:
                raise Exception('Select the correct dataset!')

        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length,
                                                  padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length,
                                                   padding='max_length', truncation=True, return_tensors="pt")
        if self.type_path == 'validation' and self.model_type == 'GPT2':
            ground_truth = self.tokenizer.batch_encode_plus([str(ground_truth_)], max_length=self.output_length,
                                                            padding='max_length', truncation=True, return_tensors="pt")
        else:
            ground_truth = None

        if self.args.dataset in ['newlama', 'updatedlama', 'newlama_easy']:
            labels = example_batch['unique_id']
        else:
            labels = None

        return source, targets, labels, ground_truth

    def __getitem__(self, index):
        source, targets, labels, ground_truth = self.convert_to_features(
            self.dataset.iloc[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if labels is not None:
            label_ids = labels
        else:
            label_ids = -1

        if ground_truth is not None:
            ground_truth_ids = ground_truth["input_ids"].squeeze()
        else:
            ground_truth_ids = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "label_ids": label_ids, "ground_truth_ids": ground_truth_ids}
