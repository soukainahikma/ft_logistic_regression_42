import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/dataset_train.csv')
value1 = 'Astronomy'
value2 = 'Defense Against the Dark Arts'
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[value1], y=df[value2], hue= df['Hogwarts House'])
plt.title(f'Scatter Plot of {value1} vs {value2}')
plt.xlabel(value1)
plt.ylabel(value2)
plt.grid(True)
plt.savefig('scatter_plot.jpeg')
