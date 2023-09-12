import json
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def gather_corpus_from_jsonl(file_path):
    """
    从给定的.jsonl文件中收集corpus
    :param file_path: .jsonl文件的路径
    :return: 按月分类的corpus列表
    """
    monthly_corpus = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            date = data.get("date", "")
            text = data.get("corpus", "")
            text = str(text)
            if date not in monthly_corpus:
                monthly_corpus[date] = []
            monthly_corpus[date].append(text)

    return {date: ' '.join(texts) for date, texts in monthly_corpus.items()}


def calculate_similarity(corpus_by_month):
    """
    计算每月与整个语料库的TF-IDF相似度
    :param corpus_by_month: 按月分类的corpus字典
    :return: 相似度列表
    """
    all_texts = list(corpus_by_month.values())
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    # 计算相似度
    similarity = (tfidf_matrix * tfidf_matrix.T).A
    return similarity


def plot_heatmap(similarity, dates):
    """
    绘制热力图
    :param similarity: 相似度矩阵
    :param dates: 日期列表
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, annot=False, xticklabels=dates, yticklabels=dates, cmap="YlGnBu")
    plt.title("Monthly Corpus Similarity Heatmap")
    plt.savefig('similarity_heatmap.png')
    plt.show()


file_path = '/yuhao/OCKL/dataset_from_2019_to_2023/dataset_from_2019-1-1_to_2023-5-31_per_month/datesorted_train_1.jsonl'
corpus_by_month = gather_corpus_from_jsonl(file_path)
# 取前5个月
first_five_months = dict(list(corpus_by_month.items())[:5])
similarity = calculate_similarity(first_five_months)
dates = list(first_five_months.keys())
plot_heatmap(similarity, dates)
