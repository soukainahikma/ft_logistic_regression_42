from sklearn.model_selection import train_test_split
import pandas as pd
from ft_logistic_regression.dslr import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.metrics import accuracy_score

df_train = pd.read_csv(sys.argv[1])

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

df_train = df_train.dropna(subset=features)
X = df_train[features].values
y = df_train[lable].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=4)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LogisticRegression()
reg.gd(X_train_scaled, y_train)

pred = reg.predict(X_test_scaled)

print(f'Accuracy score: {accuracy_score(y_test, pred, normalize=True)}')
