# pylint: disable=import-error

import json
import random
import pandas as pd
from tqdm import tqdm
import os


def load_dataset(type_path, args, length=None):
    if not args.dataset_version in ['small', 'full']:
        raise Exception('Provide the correct dataset version')

    ids_to_answers = None

    if args.dataset == "wikidata":
        dataset = []
        file_path = os.path.join(args.root_path, "dataset_from_2019_to_2023")
        g_name = "dataset_from_2019-1-1_to_2023-5-31_per_" + args.granularity
        filename = os.path.join(file_path, g_name, "datesorted_{}.jsonl".format(type_path))
        num_file = sum([1 for i in open(filename, "r")])
        with open(filename, 'r') as f:
            for line in tqdm(f, total=num_file):
                dataset.append(json.loads(line))
        return dataset, num_file

    if args.dataset == 'recentnews':
        if type_path == 'train':
            if args.dataset_version == 'small':
                dataset = pd.read_csv('data/recent_news_small.csv')
            elif args.dataset_version == 'full':
                dataset = pd.read_csv('data/recent_news_full.csv')
        elif type_path == 'split':
            if args.split == 1:
                if args.dataset_version == 'small':
                    dataset = pd.read_csv(
                        'data/split/recent_news_small1.csv')
                else:
                    raise Exception(
                        'Not supporting split for full setting.')
            elif args.split == 2:
                if args.dataset_version == 'small':
                    dataset = pd.read_csv(
                        'data/split/recent_news_small2.csv')
                else:
                    raise Exception(
                        'Not supporting split for full setting.')
            else:
                raise Exception('Currently only supporting two splits.')

        # for mixreview pretraining corpus
        elif type_path == 'pretrain':
            # if args.dataset_version == 'small':
            total_line = 802776
            skip = sorted(random.sample(
                range(1, total_line+1), total_line-length))
            dataset = pd.read_csv('data/wikipedia_pretrain_small.csv', usecols=[
                'input', 'output', 'original'], skiprows=skip)

            # elif args.dataset_version == 'full':
            #     total_line = 8021155
            #     skip = sorted(random.sample(
            #         range(1, total_line+1), total_line-length))
            #     dataset = pd.read_csv(
            #         'data/wikipedia_pretrain_small.csv', usecols=['input', 'output'], skiprows=skip)

    # # GPT-2  was  initially  pretrained  on  WebText  (Dec  2019),  which  consists  of  8  million  documents  withWikipedia  pages  excluded.
    # # In  order  to  measure  the  performance  on  INVARIANTLAMA  constructed  from Wikipedia, we continually pretrain GPT-2 on a subset of Wikipedia (May 2020) for 14k global training stepsbefore CKL.
    # elif args.dataset == 'wikitext103':
    #     dataset = pd.read_csv('data/wikipedia_pretrain_1G_final.csv')
    # dataset for evaluation
    else:
        if args.dataset == 'invariantlama':
            # light tuning 5000 instances for GPT2 experiment
            if type_path == 'train':
                dataset = pd.read_csv('data/trex_5000.csv')
            else:
                dataset = pd.read_csv('data/invariantLAMA.csv')
        elif args.dataset == 'updatedlama':
            if args.dataset_version == 'full':
                dataset = pd.read_csv(
                    'data/updatedlama/updatedLAMA.csv')
                with open('data/updatedlama_val_answers.json') as f:
                    ids_to_answers = json.load(f)
            else:
                raise Exception(
                    'Not supporting small setting for updatedLAMA.')

        elif args.dataset == 'newlama':
            if args.dataset_version == 'full':
                dataset = pd.read_csv('data/newlama/newLAMA.csv')
                with open('data/recentlama_h_val_answers.json') as f:
                    ids_to_answers = json.load(f)
            else:
                raise Exception(
                    'Not supporting small setting for newLAMA.')
        elif args.dataset in ['newlama_easy', 'newqa_easy']:
            if args.dataset_version == 'small':
                if args.split:
                    if args.split == 1:
                        rp_dir = 'data/newlama/newLAMA_easy_small_split1.csv'
                    else:
                        rp_dir = 'data/newlama/newLAMA_easy_small_split2.csv'
                else:
                    rp_dir = 'data/newlama/newLAMA_easy_small.csv'
            elif args.dataset_version == 'full':
                rp_dir = 'data/newlama/newLAMA_easy.csv'

            # light tuning 5000 instances for GPT2 experiment
            if type_path == 'train':
                dataset = pd.read_csv(
                    'data/newlama/newLAMA_easy_5000.csv')
            else:
                dataset = pd.read_csv(rp_dir)

            with open('data/recentlama_val_answers.json') as f:
                ids_to_answers = json.load(f)
        # kilt finetuning + evaluation
        # elif args.dataset== 'TriviaQA':
        #     # Get the KILT task datasets
        #     kilt_triviaqa = load_dataset("kilt_tasks", name="triviaqa_support_only")

        #     # Most tasks in KILT already have all required data, but KILT-TriviaQA only provides the question IDs, not the questions themselves.
        #     # Thankfully, we can get the original TriviaQA data with:
        #     trivia_qa = load_dataset('trivia_qa', 'unfiltered.nocontext')

        #     # The KILT IDs can then be mapped to the TriviaQA questions with:
        #     triviaqa_map = {}

        #     def add_missing_data(x, trivia_qa_subset, triviaqa_map):
        #         i = triviaqa_map[x['id']]
        #         x['input'] = trivia_qa_subset[i]['question']
        #         #x['output']['original_answer'] = trivia_qa_subset[i]['answer']['value']
        #         return x

        #     for k in ['train', 'validation', 'test']:
        #         triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])
        #         kilt_triviaqa[k] = kilt_triviaqa[k].filter(lambda x: x['id'] in triviaqa_map)
        #         kilt_triviaqa[k] = kilt_triviaqa[k].map(add_missing_data, fn_kwargs=dict(trivia_qa_subset=trivia_qa[k], triviaqa_map=triviaqa_map))
        #     dataset = kilt_triviaqa[type_path]
        #     with open('data/tqa_val_answers.json') as f:
        #         ids_to_answers = json.load(f)
        # elif args.dataset== 'fever':
        #     kilt_fever = load_dataset("kilt_tasks", name="fever")
        #     dataset = kilt_fever[type_path]
        # elif args.dataset== 'AY2':
        #     kilt_ay2 = load_dataset("kilt_tasks", name='aidayago2')
        #     dataset = kilt_ay2[type_path]
        # elif args.dataset== 'WNED':
        #     kilt_wned = load_dataset("kilt_tasks", name="wned")
        #     dataset = kilt_wned[type_path]
        # elif args.dataset== 'CWEB':
        #     kilt_cweb = load_dataset("kilt_tasks", name="cweb")
        #     dataset = kilt_cweb[type_path]
        # elif args.dataset== 'TREX':
        #     kilt_trex = load_dataset("kilt_tasks", name="trex")
        #     dataset = kilt_trex[type_path]
        #     with open('data/trex_val_answers.json') as f:
        #         ids_to_answers = json.load(f)
        # elif args.dataset== 'zsRE':
        #     kilt_zsre = load_dataset("kilt_tasks", name="structured_zeroshot")
        #     dataset = kilt_zsre[type_path]
        #     with open('data/zsre_val_answers.json') as f:
        #         ids_to_answers = json.load(f)
        # elif args.dataset== 'NQ':
        #     kilt_nq = load_dataset("kilt_tasks", name="nq")
        #     dataset = kilt_nq[type_path]
        #     with open('data/nq_val_answers.json') as f:
        #         ids_to_answers = json.load(f)
        # elif args.dataset== 'HotpotQA':
        #     kilt_hotqa = load_dataset("kilt_tasks", name="hotpotqa")
        #     dataset = kilt_hotqa[type_path]
        #     with open('data/hotpotqa_val_answers.json') as f:
        #         ids_to_answers = json.load(f)
        # elif args.dataset== 'ELI5':
        #     kilt_eli5 = load_dataset("kilt_tasks", name="eli5")
        #     dataset = kilt_eli5[type_path]
        #     with open('data/eli5_val_answers.json') as f:
        #         ids_to_answers = json.load(f)
        # elif args.dataset== 'WOW':
        #     kilt_wow = load_dataset("kilt_tasks", name="wow", ignore_verifications=True)
        #     dataset = kilt_wow[type_path]
        else:
            raise NameError('Select the correct Dataset!')
    return dataset, ids_to_answers
