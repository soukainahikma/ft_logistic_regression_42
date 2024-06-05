from sklearn.model_selection import train_test_split
import pandas as pd
from logreg_predict import LogisticRegression
from sklearn.preprocessing import StandardScaler
df_train = pd.read_csv('../data/dataset_train.csv')
df_test = pd.read_csv('../data/dataset_test.csv')

features = [
            # 'Arithmancy',
            'Astronomy',
            'Herbology',
            # 'Defense Against the Dark Arts',
            'Divination',
            'Muggle Studies',
            'Ancient Runes',
            # 'History of Magic',
            # 'Transfiguration',
            'Potions',
            # 'Care of Magical Creatures',
            'Charms',
            'Flying'
]


lable = 'Hogwarts House'


# split data into test train

df_train = df_train.dropna(subset=features)
X = df_train[features].values
y = df_train[lable].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=4)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LogisticRegression()
reg.sgd(X_train_scaled, y_train)

pred = reg.predict(X_test_scaled)


print('------------------ score and prediction in testing  ---------------')
print(f'{reg.score(X_test_scaled, y_test):.2f}')
print('Misslabeled Data : ', sum(pred != y_test))
print(pd.DataFrame(pred, columns=['Hogwarts House']))

final_test = df_test[features].values
final_test_scaled = scaler.transform(final_test)


final_pred = reg.predict(final_test_scaled)
print('------------------prediction in final testing  ---------------')
houses = pd.DataFrame(final_pred, columns=['Hogwarts House'])

houses.index.names = ['Index']
houses.to_csv('./house.csv')
