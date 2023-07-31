import os
import pandas as pd
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import numpy as np
import torch
from processor import random_spans_noise_mask, filter_input_ids, create_sentinel_ids

# Main data processing function that will concatenate all texts from our dataset and generate chunks of expanded_inputs_length.
def group_texts(examples, type_path = 'train'):
    # Concatenate all texts.
    concatenated_examples = []
    total_length = 0
    if type_path == 'pretrain':
        for idx in range(len(examples)):
            concatenated_examples.append(examples.iloc[idx]['original'])
    else:
        for e in examples:
            s_ = ''
            for s in e['corpus']:
                s_ += s
                if s != e['corpus'][-1]:
                    s_ += ' '
            concatenated_examples.append(s_)
            total_length += len(s_)
    print("Average length of the data sample is: ", total_length / len(concatenated_examples))
    return concatenated_examples

def date2int(date):
    """Convert (year, month, day) to integer representation.

    Args:
      date: Tuple of (year, month, day).

    Returns:
      an int of year * 1e4 + month * 1e2 + day.
    """
    # dint = date[0] * 1e10 if date[0] else 0
    # dint += date[1] * 1e8 if date[1] else 0
    # dint += date[2] * 1e6 if date[2] else 0
    if len(date) == 3:
        dint = int(date[0]) * 1e4 if int(date[0]) else 0
        dint += int(date[1]) * 1e2 if int(date[1]) else 0
        dint += int(date[2]) if int(date[2]) else 0
    else:
        dint = int(date[0]) * 1e2 if int(date[0]) else 0
        dint += int(date[1]) if int(date[1]) else 0
    return int(dint)

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, length=None):
        self.args = args
        print(f'Load dataset type is {type_path}')
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.ssm = False
        self.dataset_version = self.args.dataset_version
        mask_id = []
        for i in range(101):
            tokens = "<extra_id_{}>".format(i)
            mask_id.append(tokenizer.convert_tokens_to_ids(tokens))
        self.mask_token_id = mask_id
        if 't5' in args.model_name_or_path:
            self.model_type = 'T5'
        elif 'gpt2' in args.model_name_or_path:
            self.model_type = 'GPT2'
        dataset_v = ['small', 'full']
        ids_to_answers = None
        if not self.dataset_version in dataset_v:
            raise Exception(f'Provided the correct dataset version among {dataset_v}')
        # dataset for continual training
        if self.args.dataset == 'wikidata':
            # for mixreview pretraining corpus
            if type_path == 'pretrain':
                import random
                dataset = pd.read_csv('/home/users/tongjun_shi/data_with_date/wikipedia/data/wikipedia_pretrain_small.csv',
                                      usecols=['input', 'output', 'original'])
                # if self.dataset_version == 'small':
                #     total_line = 802776
                #     skip = sorted(random.sample(range(1, total_line + 1), total_line - length))
                #     dataset = pd.read_csv('data/wikipedia_pretrain_small.csv', usecols=['input', 'output', 'original'],
                #                                skiprows=skip)
                # elif self.dataset_version == 'full':
                #     total_line = 8021155
                #     skip = sorted(random.sample(range(1, total_line + 1), total_line - length))
                #     dataset = pd.read_csv('data/wikipedia_pretrain_full.csv', usecols=['input', 'output'],
                #                                skiprows=skip)
            else:
                dataset = []
                file_path = os.path.join(args.root_path, "dataset_from_2019_to_2023")
                g_name = "dataset_from_2019-1-1_to_2023-5-31_per_" + args.granularity
                filename = os.path.join(file_path, g_name, "datesorted_{}.jsonl".format(type_path))
                num_file = sum([1 for i in open(filename, "r")])
                date = ["2019-1"]#, "2019-2"]
                idx = 0
                with open(filename, 'r') as f:
                    for line in f:#, total=num_file):
                        idx += 1
                        a = json.loads(line)
                        if a["date"] in date:
                            dataset.append(a)
                        else:
                            continue

            # self.dataset = pd.DataFrame(dataset)
            print(f'Length of dataset retrieving is.. {len(dataset)}')

        self.input_length = input_length
        self.output_length = output_length
        self.ids_to_answers = ids_to_answers

        # =================================================================
        # if self.type_path == "split" or self.type_path == "train":
        #     self.dataset = group_texts(dataset)
        if self.type_path == 'pretrain':
            self.dataset = group_texts(dataset, 'pretrain')
        else:
            # self.dataset = sorted(dataset, key=lambda x: int(date2int(x['date'].split('-'))))
            self.dataset = dataset

        # =================================================================

    def __len__(self):
        return len(self.dataset)

    def mask_sentence(self, examples):
        tokenized_sample = self.tokenizer(examples, add_special_tokens=False, return_tensors="pt").input_ids  # we don't want </s>
        mask = np.asarray([random_spans_noise_mask(tokenized_sample.shape[1])])
        label_mask = ~mask
        input_ids_sentinel = create_sentinel_ids(self.tokenizer, mask.astype(np.int8))
        labels_sentinel = create_sentinel_ids(self.tokenizer, label_mask.astype(np.int8))
        input_ids = filter_input_ids(tokenized_sample, input_ids_sentinel)
        labels = filter_input_ids(tokenized_sample, labels_sentinel)
        model_input = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
        label = self.tokenizer.batch_decode(labels, skip_special_tokens=False)[0]

        # if self.model_type == 'T5':
        #     mask_token = "<extra_id_0>"
        # else:
        #     mask_token = "<mask>"
        # e_s = examples.split()
        # effective_length = len(e_s)
        # words = np.array(e_s)
        #
        # mask = np.zeros(effective_length)
        # num_to_replace = int(effective_length * 0.4)
        # replace_indices = np.random.choice(effective_length, num_to_replace, replace=False)
        #
        # mask[replace_indices] = 1
        #
        # words_masked = np.where(mask == 1, mask_token, words)
        # model_input = ' '.join(words_masked)
        #
        # label_mask = ~mask.astype(bool)
        # label_masked = np.where(label_mask, mask_token, words)
        # label = ' '.join(label_masked)

        # print("example:", examples)
        # print("input:", model_input)
        # print("label:", label)
        return model_input, label
    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
        if self.type_path == 'test':
            input_ = example_batch['query'].replace("_X_.", "?")
            ground_truth_ = ''
            for s in example_batch['answer']:
                ground_truth_ += s['name'] + '.'
                if s != example_batch['answer'][-1]:
                    ground_truth_ += ', '
            target_ = input_ + ' ' + ground_truth_
            date = example_batch['date']
        elif self.type_path == 'train':
            corpus = ''
            for s in example_batch['corpus']:
                corpus += s
                if s != example_batch['corpus'][-1]:
                    corpus += ' '
            if self.model_type == "T5":
                input_, target_ = self.mask_sentence(corpus)
            else:
                input_ = corpus
                target_ = corpus
            date = example_batch['date']
        elif self.type_path == 'pretrain':
            if self.model_type == "T5":
                input_, target_ = self.mask_sentence(example_batch)
            else:
                input_ = example_batch
                target_ = example_batch
            date = None

        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length,
                                                  padding='max_length', truncation=True, return_tensors="pt")
        if self.type_path == 'test' and self.model_type == 'GPT2':
            ground_truth = self.tokenizer.batch_encode_plus([str(ground_truth_)], max_length=self.output_length,
                                                            padding='max_length', truncation=True, return_tensors="pt")
        else:
            ground_truth = None

        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length,
                                                   padding='max_length', truncation=True, return_tensors="pt")
        if (
                self.args.dataset == 'invariantlama' or self.args.dataset == 'TriviaQA' or self.args.dataset == 'fever' or self.args.dataset == 'AY2' or self.args.dataset == 'WNED' or self.args.dataset == 'CWEB'
                or self.args.dataset == 'TREX' or self.args.dataset == 'zsRE' or self.args.dataset == 'NQ' or self.args.dataset == 'HotpotQA' or self.args.dataset == 'ELI5' or self.args.dataset == 'WOW'):
            labels = example_batch['id']
        elif (
                self.args.dataset == 'newlama' or self.args.dataset == 'updatedlama' or self.args.dataset == 'newlama_easy' or self.args.dataset == 'newqa_easy'):
            labels = example_batch['unique_id']
        else:
            labels = None
        return source, targets, labels, ground_truth, date

    def __getitem__(self, index):
        if len(self.dataset) != 0:
            source, targets, labels, ground_truth, date = self.convert_to_features(self.dataset[index])
        else:
            source, targets, labels, ground_truth, date = self.convert_to_features(self.dataset.iloc[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        def generate_mask(token_ids):
            mask = []
            for token_id in token_ids:
                if token_id in self.mask_token_id or token_id == 0:
                    mask.append(0)
                else:
                    mask.append(1)
            return torch.tensor(mask)

        if self.model_type == "T5" or self.type_path == "test":
            src_mask = generate_mask(source_ids)
            target_mask = generate_mask(target_ids)
        else:
            src_mask = source["attention_mask"].squeeze()
            target_mask = targets["attention_mask"].squeeze()

        labels, ground_truth = None, None
        if labels is not None:
            label_ids = labels
        else:
            label_ids = -1

        if ground_truth is not None:
            ground_truth_ids = ground_truth["input_ids"].squeeze()
        else:
            ground_truth_ids = -1

        if date is None:
            date = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,
                "label_ids": label_ids, "ground_truth_ids": ground_truth_ids, "date": date}