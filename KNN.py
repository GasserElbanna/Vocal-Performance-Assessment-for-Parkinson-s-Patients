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
X = np.array(X)


test_size = 0.2
train_set = {1:[], 2:[]}
test_set = {1:[], 2:[]}
train_x = X[:-int(test_size*len(X))]
test_x = X[-int(test_size*len(X)):]

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

