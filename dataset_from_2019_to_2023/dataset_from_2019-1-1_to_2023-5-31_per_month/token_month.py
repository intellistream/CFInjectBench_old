
import json

from nltk.tokenize import word_tokenize

# 确保已下载punkt分词器模型
def count_tokens_in_jsonl(file_path):
    """
    计算给定.jsonl文件的token数
    :param file_path: .jsonl文件的路径
    :return: token的总数
    """
    total_tokens = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            text = data.get("corpus", "")
            text = str(text)
            tokens = word_tokenize(text)
            total_tokens += len(tokens)


    return total_tokens

# 使用函数计算token数
file_path = '/yuhao/OCKL/dataset_from_2019_to_2023/dataset_from_2019-1-1_to_2023-5-31_per_month/datesorted_train_1.jsonl'
total = count_tokens_in_jsonl(file_path)
print(f"Total tokens in dataset: {total}")
