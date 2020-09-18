import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

features = np.load(r'C:\Users\Data Science\Desktop\PR_final\features.npy')
gender_labels = np.load(r'C:\Users\Data Science\Desktop\PR_final\binary_gender_labels.npy')
age_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\age_labels.npy'))
id_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\id_labels.npy'))
flatten_features = np.zeros([2859, 37520])
for i in range(len(features)):
    flatten_features[i] = features[i].flatten()
pca = PCA()
pca.fit(flatten_features)
np.save(r'C:\Users\Data Science\Desktop\PR_final\pca_features.npy', flatten_features)
kf = KFold(n_splits=10)
Accuracy = 0
conf = np.zeros([2, 2])
for train_index, test_index in kf.split(flatten_features):
    X_train, X_test = flatten_features[train_index], flatten_features[test_index]
    y_train, y_test = gender_labels[train_index], gender_labels[test_index]
    gmm = GaussianMixture(n_components=2, covariance_type='diag').fit(X_train)
    labels = gmm.predict(X_test)
    # np.save(r'C:\Users\Data Science\Desktop\PR_final\gmm_pred_labels.npy', labels)
    c = [(i == j) for i, j in zip(list(labels), y_test)]
    print("Accuracy: (percent)")
    print(max((sum(c)*100) / len(c), 100 - (sum(c)*100) / len(c)))
    Accuracy += max((sum(c)*100) / len(c), 100 - (sum(c)*100) / len(c))
    if (sum(c)*100) / len(c) > 100 - (sum(c)*100) / len(c):
        confusion = confusion_matrix(y_test, list(labels))
    else:
        not_labels = [int(not m) for m in list(labels)]
        confusion = confusion_matrix(y_test, list(not_labels))
    print("Confusion Matrix:")
    print(confusion)
    conf += confusion
print("10-fold average accuracy: ", Accuracy / 10)
print("10-fold average confusion matrix: ")
print(conf)