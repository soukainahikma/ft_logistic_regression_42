import seaborn as sns
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv(sys.argv[1])
columns = df.select_dtypes(include=[np.number])
columns.dropna(how='all', axis=1, inplace=True)

# if drop_index and 'Index' in df.columns:
columns.drop('Index', axis=1, inplace=True)

sns.pairplot(columns, diag_kind='hist', kind='reg',
                 
                #   hue='Hogwarts House', 
                vars=columns.columns,
                #  kind='reg',
                #  height=10,
                #  diag_kind='hist'
                 )
# g.fig.set_size_inches(15,15)
plt.show()