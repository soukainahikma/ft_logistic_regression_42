from sklearn.model_selection import train_test_split
import pandas as pd  # type: ignore
from logreg_predict import LogisticRegression
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('../data/dataset_train.csv')
df_test = pd.read_csv('../data/dataset_test.csv')

features = [
            'Arithmancy',
            'Astronomy',
            # 'Herbology',
            # 'Defense Against the Dark Arts',
            # 'Divination',
            # 'Muggle Studies',
            'Ancient Runes',
            'History of Magic',
            'Transfiguration',
            'Potions',
            'Care of Magical Creatures',
            # 'Charms',
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
X_test_scaled = scaler.fit_transform(X_test)


X_test_scaled = scaler.transform(X_test)

reg = LogisticRegression()
reg.fit(X_train_scaled, y_train)

pred = reg.predict(X_test_scaled)

y_pred = reg.predict(X_test_scaled)

print('------------------ score and prediction in testing  ---------------')
print(reg.score(X_test_scaled, y_test))
print('Misslabeled Data : ', sum(pred != y_test))
print(pd.DataFrame(y_pred, columns=['house']))

final_test = df_test[features].dropna().values
final_test_scaled = scaler.transform(final_test)


final_pred = reg.predict(final_test_scaled)
print('------------------prediction in final testing  ---------------')
print(pd.DataFrame(final_pred, columns=['house']))
