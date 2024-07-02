import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")['Labels']
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)

sample_per_class = 5
fig = plt.figure(figsize = (nclasses*2,(1+sample_per_class*2)))
index_class = 0
for cls in classes:
    samples = np.flatnonzero(y == cls)
    samples = np.random.choice(samples,sample_per_class,replace = False)
    i = 0
    for sample in samples:
       plt_idx = i*nclasses + index_class + 1
       p = plt.subplot(sample_per_class,nclasses,plt_idx);
       p = sns.heatmap(np.array(X.loc[sample]).reshape((28,28)),cmap = plt.cm.gray,xticklabels = False,yticklabels = False)
       p = plt.axis("off")
       i += 1
    index_class += 1


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size = 7500, test_size = 2500)

x_train_scaled = X_train/255.0
x_test_scaled = X_test/255.0


y_pred = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

cm = pd.crosstab(y_test,y_pred,rownames = ["Actual"],colnames = ["Predicted"])
p = plt.figure(figsize = (10,10))
p = sns.heatmap(cm,annot = True,fmt = "d", cbar = False)