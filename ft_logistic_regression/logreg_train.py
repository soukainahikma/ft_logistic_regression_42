from sklearn.model_selection import train_test_split
import pandas as pd  # type: ignore
from logreg_predict import LogisticRegression
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

import numpy as np
df_train = pd.read_csv('../data/dataset_train.csv')
df_test = pd.read_csv('../data/dataset_test.csv')

features = ['Arithmancy',
            # 'Astronomy',
            # 'Herbology',
            # 'Defense Against the Dark Arts',
            # 'Divination',
            # 'Muggle Studies',
            'Ancient Runes',
            'History of Magic',
            'Transfiguration',
            'Potions',
            'Care of Magical Creatures',
            'Charms',
            'Flying'
            ]

lable = 'Hogwarts House'

# print(df_train[features]())
df_train = df_train.dropna()
X_train = df_train[features].values
y_train = df_train[lable].values
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(df_test.head(1).info())
X_test = df_test[features].dropna().values

X_test_scaled = scaler.transform(X_test)

reg = LogisticRegression()
reg.fit(X_train_scaled, y_train)

pred = reg.predict(X_train_scaled)
print(reg.score(X_train_scaled, y_train))
print(sum(pred == y_train))

pred_test = reg.predict(X_test_scaled)
print(pd.DataFrame(pred_test, columns=['house']))
