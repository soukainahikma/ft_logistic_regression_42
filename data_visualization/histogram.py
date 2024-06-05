import pandas as pd
import matplotlib.pyplot as plt
import sys


data = pd.read_csv(sys.argv[1])

subjects = [
    'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
    'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
    'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
    'Flying'
]

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 9))
axes = axes.flatten()

for idx, subject in enumerate(subjects):
    for house in df['Hogwarts House'].unique():
        house_data = df[df['Hogwarts House'] == house][subject].dropna()
        axes[idx].hist(house_data, alpha=0.5, label=house, edgecolor='black')

    axes[idx].set_title(subject)
    axes[idx].legend()

plt.tight_layout()
plt.show()