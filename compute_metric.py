import copy
import sys
import csv

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

def six_month_value(array, gap):
    first_value = array[0]
    remaining_array = array[1:]
    if len(remaining_array) % gap != 0:
        remaining_array = remaining_array[:-(len(remaining_array) % gap)]

    reshaped_array = remaining_array.reshape(-1, gap)

    mean_values = np.mean(reshaped_array, axis=1)
    final_values = np.insert(mean_values, 0, first_value)

    return final_values

def plot_knowledge(date, knowledge_results, output_filename):
    name_list = list(knowledge_results.keys())
    gap = 4
    date = date[::gap]
    plt.figure(figsize=(12, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf','#546C75']
    linestyles = ['-']
    markers = ['o', '^', 's', 'x', 'D', 'p', '*', 'h', 'H', '+']
    for i in range(len(name_list)):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        plt.plot(date, six_month_value(knowledge_results[name_list[i]], gap), label=name_list[i], color=color, marker=marker, linestyle=linestyle, linewidth=3)

    plt.xlabel('Date', fontsize=21)
    plt.ylabel('Number of Facts', fontsize=21)

    plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale('log')
    plt.tight_layout()
    print(f'Save figure in: {output_filename}')
    plt.savefig(f'{output_filename}.pdf', format='pdf')
    plt.show()

MODE = 't5-base'
LEN = 53
save_for_plot = True
norm_plot = False
writefile = open(f'kg/temp_results.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(writefile)
writer.writerow(["Mode", 'Method', "EM", "BWT", "FWT", "KG", 'KAR'])

if MODE == 't5-base':
    root = 'log/wiki/base'
    method_name = ['initial', 'vanilla', 'recadam', 'mixreview', 'lora', 'kadapter_k=2', 'modular', 'kd', 'kilm']
elif MODE == 't5-large':
    root = 'log/wiki/large'
    method_name = ['initial', 'vanilla', 'recadam', 'mixreview', 'lora', 'kadapter_k=2', 'modular', 'kd', 'kilm']
elif MODE == 'flan':
    root = 'log/wiki/flan'
    method_name = ['initial', 'baseline', 'mixreview', 'lora', 'kadapter_k=2', 'modular_small', 'kilm']
elif MODE == 'stream':
    root = 'log/wiki/stream'
    method_name = ['baseline_stream', 'recadam_stream', 'mixreview_stream', 'lora_stream', 'kadapter_k=2_stream',
                   'modular_stream', 'kd_stream', 'kilm_stream']
elif MODE == 'coreset':
    root = 'log/wiki/coreset'
    method_name = ['T5_base_random_r=0.5', 'T5_base_kcenter_r=0.5', 'T5_base_model_r=0.5']
elif MODE == 'ratio':
    root = 'log/wiki/ratio'
    method_name = ['T5_base_kcenter_r=0.25', 'T5_base_kcenter_r=0.75']
elif MODE == 'gpt2':
    root = 'log/wiki/gpt2'
    method_name = ['initial', 'baseline', 'recadam', 'mixreview', 'lora', 'kadapter']

if save_for_plot:
    name_mapping = {'initial': 'Initial', 'vanilla':'Vanilla', 'recadam':'RecAdam', 'mixreview':'MixReview',
                    'lora':'LoRA', 'kadapter_k=2':'Kadapters (k=2)', 'modular':'Modular', 'kd':'KD', 'kilm':'KILM'}
    knowledge_results = {}

for name in method_name:
    result_df = pd.read_csv(osp.join(root, name, 'results.csv'))

    transform_csv(osp.join(root, name, 'acc.csv'), osp.join(root, name, 'check_acc.csv'), LEN)
    acc_df = pd.read_csv(osp.join(root, name, 'check_acc.csv'), header=None, skiprows=1).fillna(0)

    token_df = pd.read_csv('data_statistics/tokens.csv')
    samples_df = pd.read_csv('data_statistics/samples.csv')

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

    if name == 'initial':
        acc = acc_df.iloc[0].values[:LEN]
        assert len(acc) == len(tokens) == len(samples)
        model = np.zeros(LEN)
        kg = np.zeros(LEN)

        for i in range(LEN):
            model[i] = np.sum(acc[:i+1] * 0.01 * samples[:i+1])
        kg = np.mean((world - model) / world)

        if save_for_plot:
            knowledge_results[name_mapping[name]] = model / factor

        kg = np.mean(kg)
        em = np.mean(acc)
        bwt = 0.
        fwt = 0.
        kar = 0.
        traintime = 0.
    else:
        acc = acc_df.iloc[:LEN].values
        fwt = result_df.loc[:LEN-1, 'FWT'].values[1:]
        traintime = result_df.loc[:LEN-1, 'Time'].values
        em = np.zeros(LEN)
        bwt = np.zeros(LEN)
        kar = np.zeros(LEN)
        model = np.zeros(LEN)

        # calculate model knowledge
        current_len = 1
        for i in range(LEN):
            if i == 0:
                em[i] = acc[i][:current_len]
                model[i] = acc[i][0] * samples[i] * 0.01
                current_len += 1
            else:
                temp_bwt = acc[i-1][:current_len-1] - acc[i][:current_len-1]
                bwt[i] = np.mean(temp_bwt)
                em[i] = acc[i][:current_len][-1]
                current_len += 1

                model[i] = np.sum(acc[i] * 0.01 * samples)

        if save_for_plot:
            knowledge_results[name_mapping[name]] = model / factor

        kg = np.mean((world - model) / world)
        em = np.mean(em)
        fwt = np.mean(fwt)
        bwt = np.mean(bwt)
        kar = (fwt - bwt) * np.sum(tokens) / np.sum(traintime) * 0.01

    print(f'MODE: {MODE:6}\t Method: {name:<10}\t EM: {em:6.2f}\t BWT: {bwt:5.2f}\t '
          f'FWT: {fwt:5.2f}\t KG: {kg:5.5f}\t KAR: {kar:7.2f}\t '
          f'Tokens:{np.sum(tokens):<10}\t Time:{np.sum(traintime):6.2f}' )
    writer.writerow([f'{MODE}', f'{name}', f'{em:.2f}', f'{bwt:.2f}', f'{fwt:.2f}', f'{kg:.3f}', f'{kar:.2f}'])
    writefile.flush()

writefile.close()

if save_for_plot:
    date = samples_df.loc[:LEN-1, 'Month'].values
    plot_knowledge(date=date, knowledge_results=knowledge_results, output_filename='data_statistics/KG_ratio' if norm_plot else 'data_statistics/KG_number')
