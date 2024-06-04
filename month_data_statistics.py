import json
import nltk
import csv

nltk.download('punkt')

from nltk.tokenize import word_tokenize
def count_tokens_in_jsonl(file_path):
    all_tokens = 0
    all_samples = 0
    tokens_results = {}
    samples_results = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            all_samples += 1
            data = json.loads(line)
            date = data.get("date", "")
            text = data.get("corpus", "")
            text = str(text)
            tokens = word_tokenize(text)
            if date not in tokens_results:
                tokens_results[date] = len(tokens)
            else:
                tokens_results[date] += len(tokens)
            if date not in samples_results:
                samples_results[date] = 1
            else:
                samples_results[date] += 1
            all_tokens += len(tokens)

    return all_tokens, all_samples, tokens_results, samples_results

file_path = 'dataset_from_2019_to_2023/dataset_from_2019-1-1_to_2023-5-31_per_month/datesorted_train_no_redundancy.jsonl'
all_tokens, all_samples, tokens_results, samples_results = count_tokens_in_jsonl(file_path)
print(f"Total number of tokens: {all_tokens}")
csv_file_path = 'data_statistics/tokens.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Month', 'Tokens'])
    for month, count in tokens_results.items():
        writer.writerow([month, count])
print(f"Saved in: {csv_file_path}")
print(f"Total number of samples: {all_samples}")
csv_file_path = 'data_statistics/samples.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Month', 'Samples'])
    for month, count in samples_results.items():
        writer.writerow([month, count])
print(f"Saved in: {csv_file_path}")
