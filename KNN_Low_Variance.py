import math
import numpy as np
import pandas as pd
from collections import Counter
import random
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt


def KNN(data, predict, k):
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


x = pd.read_excel('LSVT_voice_rehabilitation.xlsx','Data')
X = x.astype(float).values.tolist()
energy = []
power = []
curve_length = []
nonlin_energy = []
dataset = []
s1 = 0
s2 = 0
s3 = 0
s4 = 0
X = np.array(X)
sel = VarianceThreshold(threshold=(.8 * (1-.8)))
X = sel.fit_transform(X)
print(len(X[0]))
for i in range(len(X[0])):
    dataset.append(X[:,i])
Dataset = []
for i in zip(*dataset):
    Dataset.append((list(i)))
Dataset = np.array(Dataset)

test_size = 0.1
train_set = {1:[], 2:[]}
test_set = {1:[], 2:[]}
train_x = Dataset[:-int(test_size*len(Dataset))]
test_x = Dataset[-int(test_size*len(Dataset)):]

for i in train_x:
    train_set[i[-1]].append(i[:-1])

for i in test_x:
    test_set[i[-1]].append(i[:-1])


correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote = KNN(train_set, data, k=21)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy =', correct/total)

