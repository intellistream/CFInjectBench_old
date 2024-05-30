# pylint: disable=import-error

import pandas as pd


def load_dataset(type_path, args):

    if args.dataset == 'wiki':
        if args.red_flag == 'nored':
            df = pd.read_json(
                f'dataset_from_2019_to_2023/dataset_from_2019-1-1_to_2023-5-31_per_{args.dataset_version}/datesorted_{type_path}_no_redundancy.jsonl', lines=True, convert_dates=False)
        if args.red_flag == 'red':
            df = pd.read_json(
                f'dataset_from_2019_to_2023/dataset_from_2019-1-1_to_2023-5-31_per_{args.dataset_version}/datesorted_{type_path}_redundancy.jsonl', lines=True, convert_dates=False)

        if type_path == 'train':
            df['corpus'] = df['corpus'].apply(lambda x: x[0])

        elif type_path == 'test':
            df['answer'] = df['answer'].apply(lambda x: x[0]['name'])

        df.drop(columns=['id'], inplace=True)

    return df
