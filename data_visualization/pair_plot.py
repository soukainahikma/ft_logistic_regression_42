import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('../data/dataset_train.csv')
columns = df.select_dtypes(include=[np.number])
columns = df.copy()
columns.dropna(how='all', axis=1, inplace=True)
sns.pairplot(columns, diag_kind='hist', hue='Hogwarts House')
plt.savefig('pair_plot.jpeg')
