import pandas as pd
from dslr import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys


def predict():
    if (len(sys.argv) != 3):
        sys.exit('Enter <test data> <model.pkl>')

    try:

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

        X = df_test[features].values

        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X)

        reg = LogisticRegression()

        pred = reg.predict(X_test_scaled, sys.argv[2])

        houses = pd.DataFrame(pred, columns=['Hogwarts House'])

        houses.index.names = ['Index']
        houses.to_csv('./house.csv')

    except Exception as error:
        sys.exit(error)


if __name__ == "__main__":
    predict()
