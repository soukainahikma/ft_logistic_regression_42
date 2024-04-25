import pandas as pd
from math import sqrt
import numpy as np


def get_count_mean(data):
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

        print(count)
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


def get_median(data, count, x):

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
            _25_percent[key] = get_median(column, len(column) - 1, 0.25)
            _50_percent[key] = get_median(column, len(column) - 1, 0.50)
            _75_percent[key] = get_median(column, len(column) - 1, 0.75)

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


def describe(data):
    df = data._get_numeric_data()
    print(df)
    count_df, mean_df, std_df = get_count_mean(df)
    _25_percent_df, _50_percent_df, _75_percent_df, max_df, min_df = quartiles(
                                                                df)

    result_df = pd.concat([count_df,
                           mean_df['mean'],
                           std_df['std'],
                           min_df['min'],
                           _25_percent_df['25%'],
                           _50_percent_df['50%'],
                           _75_percent_df['75%'],
                           max_df['max']
                           ], axis=1)
    return (result_df.set_index('').T)
