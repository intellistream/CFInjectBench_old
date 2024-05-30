import copy
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp


def transform_csv(input_file, output_file, desired_cols):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(',')
            missing_cols = desired_cols - len(parts)
            if missing_cols > 0:
                parts.extend([''] * missing_cols)
            outfile.write(','.join(parts) + '\n')

def plot_knowledge(date, knowledge_results, output_filename):
    # x_column：dict --> the result of world knowledge and each CL method's model knowledge
    # y_column: list --> the name of each knowledge
    name_list = list(knowledge_results.keys())
    plt.figure(figsize=(12, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    linestyles = ['-', '--', '-.']
    for i in range(len(name_list)):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        plt.plot(date, knowledge_results[name_list[i]], label=name_list[i], color=color, linestyle=linestyle, linewidth=3)

    plt.xlabel('Date', fontsize=21)
    plt.ylabel('Number of Facts', fontsize=21)

    plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(ticks=np.arange(len(date))[::int(len(date) / 10)], labels=np.array(date)[::int(len(date) / 10)],
               rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale('log')
    plt.tight_layout()
    print(f'Save figure in: {output_filename}')
    plt.savefig(f'{output_filename}.pdf', format='pdf')
    plt.show()
s
MODE = 't5-base'
LEN = 53
save_for_plot = True
norm_plot = False

if MODE == 't5-base':
    root = 'log/wiki/base'
    method_name = ['initial', 'vanilla', 'recadam', 'mixreview', 'lora', 'kadapter_k=2', 'modular', 'kd']
elif MODE == 'stream':
    root = 'log/wiki/newstream'
    method_name = ['vanilla_stream', 'recadam_stream', 'mixreview_stream', 'lora_stream', 'kadapter_k=2_stream',
                   'modular_stream', 'kd_stream']
elif MODE == 'large':
    root = 'log/wiki/month'
    method_name = ['vanilla_nored_large', 'recadam_nored_large', 'mixreview_nored_large', 'lora_nored_large',
                   'kadapter_k=2_nored_large', 'modular_nored_large', 'kd_nored_large']
elif MODE == 'flan':
    root = 'log/wiki/newflan'
    method_name = ['Flan_T5_baseline_redundancy', 'Flan_T5_xl_mixreview_online', 'Flan_T5_lora', 'Flan_T5_kadapter_k=2',
                   'Flan_T5_modular_small']
elif MODE == 'select':
    root = 'log/wiki/month'
    method_name = ['T5_base_coreset_random_euc_r=0.5_red', 'T5_base_coreset_kcenter_r=0.5_red',
                   'T5_base_coreset_model_euc_r=0.5_red']
elif MODE == 'ratio':
    root = 'log/wiki/month'
    method_name = ['T5_base_coreset_kcenter_euc_r=0.25', 'T5_base_coreset_kcenter_euc_r=0.5',
                   'T5_base_coreset_kcenter_euc_r=0.75', 'vanilla_nored_euc']
elif MODE == 'gpt2':
    root = 'log/wiki/month'
    method_name = ['GPT2_initial', 'GPT2_baseline', 'GPT2_recadam', 'GPT2_mixreview', 'GPT2_lora', 'GPT2_kadapter']

if save_for_plot:
    name_mapping = {'initial': 'Initial', 'vanilla':'Vanilla', 'recadam':'RecAdam', 'mixreview':'Mix-Review',
                    'lora':'LoRA', 'kadapter_k=2':'K-Adapter', 'modular':'Modular', 'kd':'KD'}
    knowledge_results = {}

for name in method_name:
    result_df = pd.read_csv(osp.join(root, name, 'results.csv'))

    transform_csv(osp.join(root, name, 'acc.csv'), osp.join(root, name, 'check_acc.csv'), LEN)
    acc_df = pd.read_csv(osp.join(root, name, 'check_acc.csv'), header=None, skiprows=1).fillna(0)

    token_df = pd.read_csv('kg/nored/nored_tokens.csv')
    samples_df = pd.read_csv('kg/nored/nored_data_number.csv')

    tokens = token_df.loc[:LEN-1, 'Tokens'].values
    samples = samples_df.loc[:LEN-1, 'Samples'].values

    # calculate world knowledge
    world = copy.deepcopy(samples)
    for i in range(LEN):
        if i == 0:
            delta = 0
        else:
            delta = world[i - 1]
        world[i] += delta
    if save_for_plot and 'World' not in knowledge_results:
        if norm_plot:
            factor = world
        else:
            factor = np.ones(LEN)
        knowledge_results['Perfect Injection'] = world / factor
        # knowledge_results['World'] = world / world

    if name == 'initial':
        acc = acc_df.iloc[0].values[:LEN]
        assert len(acc) == len(tokens) == len(samples)
        model = np.zeros(LEN)
        kg = np.zeros(LEN)

        for i in range(LEN):
            model[i] = np.sum(acc[:i+1] * 0.01 * samples[:i+1])
        kg = np.mean((world - model) / world)
        # for i in range(LEN):
        #     kg[i] = ((model[i] - world[i]) ** 2) / (world[i] ** 2)
        # kg = np.sqrt(np.sum(kg))

        if save_for_plot:
            knowledge_results[name_mapping[name]] = model / factor

        kg = np.mean(kg)
        em = np.mean(acc)
        print(f'MODE: {MODE:6}\t Method: {name:<10}\t EM: {em:6.2f}\t BWT: {0:5.2f}\t '
              f'FWT: {0:5.2f}\t KG: {kg:5.5f}\t KAR: {0:7.2f}')
    else:
        acc = acc_df.iloc[:LEN].values
        fwt = result_df.loc[:LEN-1, 'FWT'].values[1:]
        traintime = result_df.loc[:LEN-1, 'Time'].values
        em = np.zeros(LEN)
        bwt = np.zeros(LEN)
        kar = np.zeros(LEN)
        model = np.zeros(LEN)

        # calculate model knowledge
        for i in range(LEN):
            if i == 0:
                em[i] = acc[i][0]
                model[i] = acc[i][0] * samples[i] * 0.01
            else:
                non_zero_len = len(acc[i][acc[i] != 0])
                temp_bwt = acc[i][:non_zero_len] - acc[i-1][:non_zero_len]
                bwt[i] = np.mean(temp_bwt[:-1])
                em[i] = acc[i][:non_zero_len][-1]

                model[i] = np.sum(acc[i] * 0.01 * samples)

        if save_for_plot:
            knowledge_results[name_mapping[name]] = model / factor

        kg = np.mean((world - model) / world)
        # kg = np.zeros(LEN)
        # for i in range(LEN):
        #     kg[i] = ((model[i] - world[i]) ** 2) / (world[i] ** 2)
        # kg = np.sqrt(np.sum(kg))
        em = np.mean(em)
        fwt = np.mean(fwt)
        bwt = np.mean(bwt)
        kar = (bwt + fwt) * np.sum(tokens) / np.sum(traintime)

        print(f'MODE: {MODE:6}\t Method: {name:<10}\t EM: {em:6.2f}\t BWT: {bwt:5.2f}\t FWT: {fwt:5.2f}\t KG: {kg:5.5f}\t KAR: {kar:7.2f}')

if save_for_plot:
    date = samples_df.loc[:LEN-1, 'Month'].values
    plot_knowledge(date=date, knowledge_results=knowledge_results, output_filename='kg/KG_ratio' if norm_plot else 'kg/KG_number')