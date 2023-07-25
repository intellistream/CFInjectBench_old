import pandas as pd

def replace_repeating_ids(input_string):
    count = 0
    while "<extra_id>" in input_string:
        input_string = input_string.replace("<extra_id>", f"<extra_id_{count}>", 1)
        count += 1
    return input_string

df = pd.read_csv('train_data_month.csv', nrows=100000)

df['input'] = df['input'].apply(replace_repeating_ids)
df['output'] = df['output'].apply(replace_repeating_ids)

df.to_csv('temp.csv', index=False)