import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/dataset_train.csv')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Astronomy'], y=df['Defense Against the Dark Arts'])
plt.title('Scatter Plot of Astronomy vs Defense Against the Dark Arts')
plt.xlabel('Astronomy')
plt.ylabel('Defense Against the Dark Arts')
plt.grid(True)
plt.savefig('scatter_plot.jpeg')
