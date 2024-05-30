import pandas as pd
import matplotlib.pyplot as plt
import sys


data = pd.read_csv(sys.argv[1])

# plt.scatter(data['Astronomy'], data['Defense Against the Dark Arts'])
plt.scatter(data['Hogwarts House'], data[['Muggle Studies']],
                c=data['Hogwarts House'].astype('category').cat.codes,
                cmap='viridis')

plt.tight_layout()
plt.show()
