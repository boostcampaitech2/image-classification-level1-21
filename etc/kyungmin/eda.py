import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

PATH = '../train/train.csv'
df = pd.read_csv(PATH)
df = df.drop('race', axis=1)
df = df.drop('path', axis=1)
df = df.drop('id', axis=1)

male_group = df.loc[df.gender=='male', 'age'].value_counts().sort_index()
female_group = df.loc[df.gender=='female', 'age'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(15,7))

axes[0].bar(male_group.index, male_group.values, color='royalblue')
axes[0].set_xticks(np.arange(9,31)*2)
axes[0].set_title('male')
axes[1].bar(female_group.index, female_group.values, color='tomato')
axes[1].set_xticks(np.arange(9,31)*2)
axes[1].set_title('female')
plt.show()

PATH = './train/new_images/'

label_list = sorted(os.listdir(PATH))

images_of_label = list()
for label in label_list:
    images_of_label.append(int(os.popen('ls ' + PATH + label + ' | wc -l').read()[0:-1]))

fig, axes = plt.subplots(1, 1, figsize=(15, 7))
axes.bar(np.arange(18), images_of_label, color='royalblue')
axes.set_xticks(np.arange(18))

plt.show()