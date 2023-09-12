import json
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def gather_corpus_by_quarter(file_path):
    """
    从给定的.jsonl文件中按季度收集corpus
    :param file_path: .jsonl文件的路径
    :return: 按季度分类的corpus列表
    """
    monthly_corpus = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            date = data.get("date", "")
            year, month = date.split("-")
            quarter = (int(month)-1) // 3 + 1
            quarter_key = f"{year}-Q{quarter}"
            text = data.get("corpus", "")
            text = str(text)
            if quarter_key not in monthly_corpus:
                monthly_corpus[quarter_key] = []
            monthly_corpus[quarter_key].append(text)

    # 仅返回文本数量超过3的季度
    return {date: ' '.join(texts) for date, texts in monthly_corpus.items() if len(texts) >= 3}


def calculate_similarity(corpus_by_quarter):
    """
    计算每个季度与整个语料库的TF-IDF相似度
    :param corpus_by_quarter: 按季度分类的corpus字典
    :return: 相似度列表
    """
    all_texts = list(corpus_by_quarter.values())
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    # 计算相似度（这里我们简单地使用TF-IDF的乘积作为相似度度量）
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
    plt.title("Quarterly Corpus Similarity Heatmap")
    plt.savefig('similarity_heatmap.png')
    plt.show()


file_path = '/yuhao/OCKL/dataset_from_2019_to_2023/dataset_from_2019-1-1_to_2023-5-31_per_month/datesorted_train_1.jsonl'
corpus_by_quarter = gather_corpus_by_quarter(file_path)
similarity = calculate_similarity(corpus_by_quarter)
dates = list(corpus_by_quarter.keys())
plot_heatmap(similarity, dates)
