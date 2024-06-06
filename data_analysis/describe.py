import pandas as pd
from math import sqrt
import numpy as np
import sys


def get_count_mean_std(data):
    sums = {}
    count = {}
    means = {}
    std = {}

    for key in data.keys():
        sums[key] = 0
        count[key] = float(0.0)
        means[key] = float('NaN')
        std[key] = float('NaN')

    for key, column in data.items():
        for i, value in column.items():
            if value == value:
                sums[key] += float(value)
                count[key] += 1

        if count[key] != 0:
            means[key] = f'{(sums[key]/count[key]):6f}'
            var = sum([(x - float(means[key])) ** 2 for i, x in column.items()
                       if x == x])
            std[key] = sqrt(var/(float(count[key])-1))
            std[key] = f'{std[key]:6f}'
            count[key] = f'{count[key]:6f}'
    count_df = pd.DataFrame(list(count.items()))
    count_df.columns = ['', 'count']
    mean_df = pd.DataFrame(list(means.items()))
    mean_df.columns = ['', 'mean']
    std_df = pd.DataFrame(list(std.items()))
    std_df.columns = ['', 'std']

    return (count_df, mean_df, std_df)


def get_quantiles(data, count, x):

    index = int(count) * x
    if (int(index) == index):
        return f'{data[int(index)]:6f}'
    else:
        fraction = index - int(index)
        left = int(index)
        right = left + 1
        i, j = data[left], data[right]
        return f'{i + (j - i) * fraction:6f}'


def quartiles(data):
    _25_percent = {}
    _50_percent = {}
    _75_percent = {}
    max = {}
    min = {}

    for key in data.keys():
        _25_percent[key] = np.nan
        _50_percent[key] = np.nan
        _75_percent[key] = np.nan
        max[key] = np.nan
        min[key] = np.nan

    for key, column in data.items():
        column = list(sorted(column.dropna()))
        if len(column) != 0:
            max[key] = column[-1]
            min[key] = column[0]
            max[key] = f'{max[key]:6f}'
            min[key] = f'{min[key]:6f}'
            _25_percent[key] = get_quantiles(column, len(column) - 1, 0.25)
            _50_percent[key] = get_quantiles(column, len(column) - 1, 0.50)
            _75_percent[key] = get_quantiles(column, len(column) - 1, 0.75)

    _25_percent_df = pd.DataFrame(list(_25_percent.items()))
    _25_percent_df.columns = ['', '25%']
    _50_percent_df = pd.DataFrame(list(_50_percent.items()))
    _50_percent_df.columns = ['', '50%']
    _75_percent_df = pd.DataFrame(list(_75_percent.items()))
    _75_percent_df.columns = ['', '75%']
    max_df = pd.DataFrame(list(max.items()))
    max_df.columns = ['', 'max']
    min_df = pd.DataFrame(list(min.items()))
    min_df.columns = ['', 'min']

    return (_25_percent_df, _50_percent_df, _75_percent_df, max_df, min_df)


def get_more_fields(data):
    nunique = {}
    n_missing = {}
    var = {}
    for key, column in data.items():
        nunique[key] = len(set(column.dropna()))
        n_missing[key] = len(set(column)) - nunique[key]
        var[key] = float('NaN')
        if (not pd.isnull(column).all()):
            m = sum(column.dropna())/len(column.dropna()-1)
            var[key] = sum((xi - float(m)) ** 2 for xi in column if xi == xi)\
                / (len(column.dropna()) - 1)
            var[key] = "{:.6e}".format(var[key])

        if nunique[key] != 0:
            nunique[key] = f'{nunique[key]:6f}'
        n_missing[key] = f'{n_missing[key]:6f}'

    nunique_df = pd.DataFrame(list(nunique.items()))
    nunique_df.columns = ['', 'nunique']
    n_missing_df = pd.DataFrame(list(n_missing.items()))
    n_missing_df.columns = ['', 'n_missing']
    var_df = pd.DataFrame(list(var.items()))
    var_df.columns = ['', 'var']
    return (nunique_df, n_missing_df, var_df)


def describe(data):

    df = data.select_dtypes(include='number')
    count_df, mean_df, std_df = get_count_mean_std(df)
    _25_percent_df, _50_percent_df, _75_percent_df, max_df, min_df = quartiles(
                                                                df)
    nunique, n_missing, var = get_more_fields(df)
    result_df = pd.concat([count_df,
                           mean_df['mean'],
                           std_df['std'],
                           min_df['min'],
                           _25_percent_df['25%'],
                           _50_percent_df['50%'],
                           _75_percent_df['75%'],
                           max_df['max'],
                           nunique['nunique'],
                           n_missing['n_missing'],
                           var['var']
                           ], axis=1)
    return (result_df.set_index('').T)


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        sys.exit('Enter file name')
    try:
        data = pd.read_csv(sys.argv[1])
        print(describe(data))
        # print(data.describe())
    except Exception as error:
        sys.exit(error)
