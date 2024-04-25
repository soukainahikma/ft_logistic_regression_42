import pandas as pd
import sys
from describe import describe

if __name__ == "__main__":

    data = pd.read_csv(sys.argv[1])
    data = pd.DataFrame(data)
    # print(data[['Arithmancy', 'Herbology']])
    print(describe(data))
    # print(data[['Hogwarts House']].describe())
    # print(data.describe())
    # print(data.info())
