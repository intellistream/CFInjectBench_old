import pandas as pd
import matplotlib.pyplot as plt

# Data provided by the user
data = {
    "dataset": ["Wic", "Rte", "Copa", "hotpot-acc"],
    "base": [71.15987461, 79.85611511, 54, 46.94119641],
    "2020": [59.87460815, 51.79856115, 44, 43.68751059],
    "2021": [57.99, 56.834532, 48, 41.484494],
    "2022": [59.56112, 53.95, 48, 41.75563464],
    "2023": [56.42633229, 52.17391, 58, 40.3829]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Setting the 'dataset' column as the index
df.set_index('dataset', inplace=True)

# Transposing the DataFrame to have years as the x-axis and datasets as lines
df = df.transpose()

# Plotting the data
plt.figure(figsize=(10, 5))
print(df)
plt.plot(df.index, df[0], marker='o', label=column, color="salmon")
plt.plot(df.index, df[1], marker='s', label=column, color="khaki")
plt.plot(df.index, df[2], marker='v', label=column, color="lightblue")
plt.plot(df.index, df[HotpotQA], marker='D', label=column, color="lightgreen")

plt.title('Performance over Years')
plt.xlabel('Year')
plt.ylabel('Score')
plt.legend(title='Dataset')
plt.grid(True)
plt.tight_layout()
plt.savefig()
plt.show()



