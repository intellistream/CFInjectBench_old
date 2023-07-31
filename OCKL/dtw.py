import numpy as np


def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)

    # 创建一个二维的距离矩阵，用于存储每个对应元素之间的距离
    distance_matrix = np.zeros((n, m))

    # 计算对应元素之间的距离（欧氏距离，可以根据需要选择其他距离度量）
    for i in range(n):
        for j in range(m):
            distance_matrix[i, j] = abs(s1[i] - s2[j])

    # 创建一个二维的累积距离矩阵，用于存储累积的最小距离
    cumulative_distance = np.zeros((n, m))

    # 初始化累积距离矩阵的第一行和第一列
    cumulative_distance[0, 0] = distance_matrix[0, 0]
    for i in range(1, n):
        cumulative_distance[i, 0] = distance_matrix[i, 0] + cumulative_distance[i - 1, 0]
    for j in range(1, m):
        cumulative_distance[0, j] = distance_matrix[0, j] + cumulative_distance[0, j - 1]

    # 计算累积距离矩阵中的其他元素
    for i in range(1, n):
        for j in range(1, m):
            cumulative_distance[i, j] = distance_matrix[i, j] + min(
                cumulative_distance[i - 1, j],  # 向上的距离
                cumulative_distance[i, j - 1],  # 向左的距离
                cumulative_distance[i - 1, j - 1]  # 向左上的距离
            )

    # 最终的DTW距离是累积距离矩阵右下角的元素
    dtw_distance = cumulative_distance[n - 1, m - 1]

    return dtw_distance
