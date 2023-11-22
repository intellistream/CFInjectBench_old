import pandas as pd
import matplotlib.pyplot as plt

# Data provided by the user
data = {
    "dataset": ["Wic", "Rte", "Copa", "hotpot-acc"],
    "2019(base)": [71.15987461, 79.85611511, 54, 46.94119641],
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
plt.figure(figsize=(8, 5))
print(df)
plt.plot(df.index, df['Wic'], marker='o', label='Wic', color="salmon")

# Plot for Rte
plt.plot(df.index, df['Rte'], marker='s', label='Rte', color="khaki")

# Plot for Copa
plt.plot(df.index, df['Copa'], marker='v', label='Copa', color="lightblue")

# Plot for hotpot-acc (HotpotQA)
plt.plot(df.index, df['hotpot-acc'], marker='D', label='HotpotQA', color="lightgreen")


plt.title('Performance over Years')
plt.xlabel('Year')
plt.ylabel('Score')
plt.legend(title='Dataset')

plt.tight_layout()
plt.savefig('./performance_chart.png')
plt.show()



