import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

dataset = pd.read_excel('LSVT_voice_rehabilitation.xlsx')

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=51, p=2, metric='euclidean')
rfe = RFE(clf, 3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print('Accuracy = ', accuracy_score(y_test, y_pred))
