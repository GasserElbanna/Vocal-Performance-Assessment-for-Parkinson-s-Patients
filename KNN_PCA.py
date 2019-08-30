import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

dataset = pd.read_excel('LSVT_voice_rehabilitation.xlsx')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = scaler = StandardScaler()
scaler.fit(x_train)
x_train =scaler.transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=15)
fit = pca.fit(x_train)
print(fit.components_)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

clf = KNeighborsClassifier(n_neighbors=19, p=2, metric='euclidean')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print('Accuracy = ', accuracy_score(y_test, y_pred))


