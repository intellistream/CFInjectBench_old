def print_first_lines(file_path, num_lines=5):
    """
    打印.jsonl文件的前num_lines行。

    参数:
        file_path (str): .jsonl文件的路径。
        num_lines (int): 要打印的行数，默认为5。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for _ in range(num_lines):
            line = file.readline()
            if not line:
                break
            print(line)


# 使用
file_path = 'datesorted_train_1.jsonl'
print_first_lines(file_path, num_lines=3)
