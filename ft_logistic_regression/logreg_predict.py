import pandas as pd
from ft_logistic_regression.dslr import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys

df_test = pd.read_csv(sys.argv[1])

features = [
            'Astronomy',
            'Herbology',
            'Divination',
            'Muggle Studies',
            'Ancient Runes',
            'Potions',
            'Charms',
            'Flying'
]

lable = 'Hogwarts House'

X = df_test[features].values

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X)

reg = LogisticRegression()

pred = reg.predict(X_test_scaled)

houses = pd.DataFrame(pred, columns=['Hogwarts House'])

houses.index.names = ['Index']
houses.to_csv('./house.csv')
