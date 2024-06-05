import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df = pd.read_csv(sys.argv[1])

# plt.scatter(data['Astronomy'], data['Defense Against the Dark Arts'])
# plt.scatter(data['Hogwarts House'], data[['Muggle Studies']],
#                 c=data['Hogwarts House'].astype('category').cat.codes,
#                 cmap='viridis')

# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x = df['Astronomy'], y = df['Defense Against the Dark Arts'])
plt.title(f'Scatter Plot of Astronomy vs Defense Against the Dark Arts')
plt.xlabel('Astronomy')
plt.ylabel('Defense Against the Dark Arts')
plt.grid(True)
plt.show()
