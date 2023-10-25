import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import collections
import tensorflow as tf
import sys
from absl import logging
import re
from tqdm import tqdm
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle

# # 创建一个随机的2D数据矩阵
# data = np.random.random((10, 10))
#
# # 使用Seaborn创建热力图
# sns.heatmap(data, cmap='coolwarm', annot=True)  # 使用'coolwarm'颜色映射，显示数值标签
# plt.show()

import spacy

nlp = spacy.load("en_core_web_sm")
# from nltk.tokenize import word_tokenize
# 确保已下载punkt分词器模型
def read_templates(path):
    """Loads relation-specific templates from `templates.csv`.

    Returns:
      a dict mapping relation IDs to string templates.
    """
    my_path = path
    template_file = os.path.join(my_path, "templates.csv")
    logging.info("Reading templates from %s", template_file)
    reader = csv.reader(tf.io.gfile.GFile(template_file))
    headers = next(reader, None)
    data = collections.defaultdict(list)
    for row in reader:
        for h, v in zip(headers, row):
            data[h].append(v)
    templates = dict(zip(data["Wikidata ID"], data["Template"]))
    logging.info("\n".join("%s: %s" % (k, v) for k, v in templates.items()))
    return templates

def extract_subj_obj(corpus, template):
    # 使用正则表达式构建匹配模式，将<subject>和<object>替换为相应的正则表达式
    pattern = re.sub(r'<subject>', r'(.*?)', re.escape(template))
    pattern = re.sub(r'<object>', r'(.*?)', pattern)

    # 通过正则表达式查找所有匹配的内容
    matches = re.findall(pattern, corpus)
    subjects = []
    objects = []
    # 提取匹配的subject和object
    for match in matches:
        subj, obj = match
        subjects.append(subj.strip())
        objects.append(obj.strip())
    return subjects[0], objects

def token_change(tk, n_tk):
    tokens1 = [token.text for token in nlp(tk)]
    tokens2 = [token.text for token in nlp(n_tk)]

    # 计算标记差异
    common_tokens = set(tokens1) & set(tokens2)
    token_difference = len(tokens1) + len(tokens2) - 2 * len(common_tokens)
    return token_difference

def date_change(da, n_da):
    # 将日期字符串解析为 datetime 对象
    date1 = datetime.strptime(da, '%Y-%m')
    date2 = datetime.strptime(n_da, '%Y-%m')

    # 计算月份差距
    months_diff = (date2.year - date1.year) * 12 + (date2.month - date1.month)
    return months_diff

def get_cdf(tk_change, d_change, path):
    ecdf = sm.distributions.ECDF(tk_change)
    num_points = int((80 - min(tk_change)) / 1) + 1
    x = np.linspace(min(tk_change), 80, num_points)
    y = ecdf(x)
    print(y)
    print(f"For the token change 7, the percentage is {ecdf(7)}")

    print('*'*50)
    print(f'*** min(token change) is {min(tk_change)}, max(token change) is {max(tk_change)}')
    plt.figure()
    plt.grid(True)
    plt.plot(x, y, linewidth='3', color=(244 / 255, 138 / 255, 80 / 255), linestyle="--")
    plt.xlabel('Token Changes', fontsize=15)
    plt.ylabel('CDF', fontsize=15)
    plt.xlim(0, 80)  # for token change
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(path, "analysis_token.pdf"), format="pdf")
    print(f'>>> Save CDF image in [{path}]')
    print(f'average token change is {sum(tk_change)/len(tk_change)}')

    print(f'average date change is {sum(d_change)/len(d_change)}')
    ecdf = sm.distributions.ECDF(d_change)
    num_points = int((max(d_change) - min(d_change)) / 1) + 1
    x2 = np.linspace(min(d_change), max(d_change), num_points)
    y2 = ecdf(x2)
    print(f"For the date change 20, the percentage is {ecdf(20)}")
    print(f'*** min(date change) is {min(d_change)}, max(date change) is {max(d_change)}')
    print('*'*50)
    plt.figure()
    plt.grid(True)
    plt.plot(x2, y2, linewidth='3', color=(52/255, 168/255, 97/255), linestyle="--")
    plt.xlabel('Date Changes (month)', fontsize=15)
    plt.ylabel('CDF', fontsize=15)
    plt.xlim(0, 52) # for date change
    # plt.ylim(0, 1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(path, "analysis_date.pdf"), format="pdf")
    print(f'>>> Save CDF image in [{path}]')

# 1. 统计每个属性变化的token变化量，对全部属性，全部实体取平均
# 2. 统计每个属性变化的时间间隔，对全部属性和实体取平均
train_stream_df = pd.read_json(
    f'../data_with_date/dataset_from_2019_to_2023/dataset_from_2019-1-1_to_2023-5-31_per_month/train_pre.jsonl',
    lines=True, convert_dates=False)
templates = read_templates("/home/tongjun/temporal-robustness/dynamic_data_collection")

all_subjects = collections.defaultdict(lambda: collections.defaultdict(list))
for idx, row in tqdm(train_stream_df.iterrows(), desc=">>> Rearrange knowledge by subjects", total=len(train_stream_df)):
    relation = row['relation']  # relation
    template = templates[relation]  # template
    corpus = ''
    for c in row['corpus']:
        corpus += c if c == row['corpus'][0] else (' ' + c)
    subjects, objects = extract_subj_obj(corpus, template)
    all_subjects[subjects][relation].append({'corpus': corpus, 'date': row['date']})

change_subj = 0
change_rel = 0
total_subj = 0
total_rel = 0

v_len_text = []
a_len_text = []
text_size = 0
a_len_token = []
v_len_token = []
p_token_change = []
p_date_change = []
time_variant = 0
time_invariant = 0

all_change = collections.defaultdict(lambda: collections.defaultdict(lambda: {'corpus': [], 'date': []}))
for subj in tqdm(all_subjects.keys(), desc='>>> Calculate all corpus and dates changes'):
    total_subj += 1
    for rel in all_subjects[subj].keys():
        total_rel += 1
        k = all_subjects[subj][rel] # knowledge
        if len(k) > 1:
            for i in range(len(k)):
                time_variant += 1
                text_size += 1
                tokens1 = [token.text for token in nlp(k[i]['corpus'])]
                v_len_token.append(len(tokens1))
                a_len_token.append(len(tokens1))
                v_len_text.append(len(k[i]['corpus']))
                a_len_text.append(len(k[i]['corpus']))
                if (i + 1) <= (len(k) - 1): # if next corpus still exists
                    tc = token_change(k[i]['corpus'], k[i+1]['corpus'])
                    p_token_change.append(tc)
                    dc = date_change(k[i]['date'], k[i+1]['date'])
                    p_date_change.append(dc)
                    all_change[subj][rel]['corpus'].append(tc)
                    all_change[subj][rel]['date'].append(dc)
        else:
            for i in range(len(k)):
                tokens1 = [token.text for token in nlp(k[i]['corpus'])]
                a_len_token.append(len(tokens1))
                a_len_text.append(len(k[i]['corpus']))
                time_invariant += 1
                text_size += 1
print('>>> Overall')
print(f"Averaged token length in dataset: {sum(a_len_token) / len(a_len_token)}")
print(f"Averaged token length in dataset: {sum(a_len_text) / len(a_len_text)}")
print('>>> For variant')
print(f"Averaged token length in dataset: {sum(v_len_token) / len(v_len_token)}")
print(f"Averaged token change in dataset: {sum(p_token_change) / len(p_token_change)}")
print(f"Averaged date change in dataset: {sum(p_date_change) / len(p_date_change)}")
print(f"Averaged text length in dataset: {sum(v_len_text) / len(v_len_text)}")
print('>>> For size')
print(f"Whole text size in dataset: {text_size}")
print(f"Variant text size in dataset: {time_variant}({time_variant/text_size})")
print(f"Invariant text size in dataset: {time_invariant}({time_invariant/text_size})")

p_subj = []
p_rel = []
p_mc = []
p_md = []

mean_change = collections.defaultdict(lambda: collections.defaultdict(lambda: {'corpus': None, 'date': None}))
for subj in tqdm(all_change.keys(), desc='>>> Calculate average changes based on property-level'):
    for rel in all_change[subj].keys():
        if len(all_change[subj][rel]['corpus']) > 0:
            m_c = sum(all_change[subj][rel]['corpus']) / len(all_change[subj][rel]['corpus'])
            m_d = sum(all_change[subj][rel]['date']) / len(all_change[subj][rel]['date'])
            mean_change[subj][rel]['corpus'] = m_c
            mean_change[subj][rel]['date'] = m_d
            p_subj.append(subj)
            p_rel.append(rel)
            p_mc.append(m_c)
            p_md.append(m_d)
            change_rel += 1

s_subj = []
s_mc = []
s_md = []
subject_change = collections.defaultdict(lambda: {'corpus': None, 'date': None})
for subj in tqdm(mean_change.keys(), desc='>>> Sum changes of all subjects'):
    rel_cnt = 0
    sum_tk = 0
    sum_d = 0
    for rel in mean_change[subj].keys():
        if not mean_change[subj][rel]['corpus'] is None:
            sum_tk += mean_change[subj][rel]['corpus']
            sum_d += mean_change[subj][rel]['date']
            rel_cnt += 1
    if rel_cnt != 0:
        subject_change[subj]['corpus'] = sum_tk / rel_cnt
        subject_change[subj]['date'] = sum_d / rel_cnt
        s_subj.append(subj)
        s_mc.append(sum_tk / rel_cnt)
        s_md.append(sum_d / rel_cnt)

sum_tk = 0
sum_d = 0

for subj in tqdm(subject_change, desc='>>> Calculate average changes based on subjects-level'):
    change_subj += 1
    sum_tk += subject_change[subj]['corpus']
    sum_d += subject_change[subj]['date']

get_cdf(p_token_change, p_date_change, path='/home/tongjun/OCKL-main/data_analysis')

mean_subject_tk_change = sum_tk / len(subject_change.keys())
mean_subject_d_change = sum_d / len(subject_change.keys())
print('=' * 50)
print('>>> Overall (changed / total)')
print(f'For subjects: {change_subj}/{total_subj}')
print(f'For property: {change_rel}/{total_rel}')
print('-' * 20)
print('>>> For all subjects')
print(f'*** Total number of subjects: {len(subject_change.keys())}')
print(f"*** The average token changes for all subjects: {mean_subject_tk_change}")
print(f"*** The average date changes for all subjects: {mean_subject_d_change}")
print('-' * 20)
print('>>> For single property')
for i in range(5):
    print(f"*** For {p_subj[i]}'s {p_rel[i]}, the average token and date change: {p_mc[i]}, {p_md[i]}")
# print('-' * 20)
# print('>>> For single subject')
# for i in range(5):
#     print(f"*** For {s_subj[i]}, the average token and date change: {s_mc[i]}, {s_md[i]}")
print('=' * 50)

'''
all_subjects={'subject_1': {'relation_1':[[], [], ..., []],
                            'relation_2':[[], [], ..., []],
                            ...,
                           'relation_n':[[], [], ..., []]},
              'subject_2': {'relation_1':[[], [], ..., []],
                            'relation_2':[[], [], ..., []],
                            ...,
                            'relation_n':[[], [], ..., []]},
              ...,
              'subject_n': {'relation_1':[[], [], ..., []],
                            'relation_2':[[], [], ..., []],
                            ...,
                            'relation_n':[[], [], ..., []]}
}

all_subjects={'subject_1': {'relation_1':{'corpus':[[], [], ..., []], 'date':[[], [], ..., []]}},
                            'relation_2':{'corpus':[[], [], ..., []], 'date':[[], [], ..., []]}},
                            ...,
                           'relation_n':{'corpus':[[], [], ..., []], 'date':[[], [], ..., []]}},
              'subject_2': {'relation_1':{'corpus':[[], [], ..., []], 'date':[[], [], ..., []]}},
                            'relation_2':{'corpus':[[], [], ..., []], 'date':[[], [], ..., []]}},
                            ...,
                            'relation_n':{'corpus':[[], [], ..., []], 'date':[[], [], ..., []]}},
              ...,
              'subject_n': {'relation_1':{'corpus':[[], [], ..., []], 'date':[[], [], ..., []]}},
                            'relation_2':{'corpus':[[], [], ..., []], 'date':[[], [], ..., []]}},
                            ...,
                            'relation_n':{'corpus':[[], [], ..., []], 'date':[[], [], ..., []]}},
}

all_subjects={'subject_1': {'relation_1':{'corpus':int, 'date':int}},
                            'relation_2':{'corpus':int, 'date':int}},
                            ...,
                           'relation_n':{'corpus':int, 'date':int}},
              'subject_2': {'relation_1':{'corpus':int, 'date':int}},
                            'relation_2':{'corpus':int, 'date':int}},
                            ...,
                            'relation_n':{'corpus':int, 'date':int}},
              ...,
              'subject_n': {'relation_1':{'corpus':int, 'date':int}},
                            'relation_2':{'corpus':int, 'date':int}},
                            ...,
                            'relation_n':{'corpus':int, 'date':int}},
}
all_subjects={'subject_1': {'corpus':float, 'date':float},
              'subject_2': {'corpus':float, 'date':float},
              ...,
              'subject_n': {'corpus':float, 'date':float}
}
'''
