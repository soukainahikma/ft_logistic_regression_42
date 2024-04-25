import pandas as pd
import sys
from data_analysis.describe import describe

if __name__ == "__main__":

    data = pd.read_csv(sys.argv[1])
    data = pd.DataFrame(data)
    # print(data[['Arithmancy', 'Herbology']])
    print(describe(data))
    print(data.describe())
