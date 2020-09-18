import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from math import sqrt
from random import shuffle
import matplotlib.pyplot as plt

features = np.load(r'C:\Users\Data Science\Desktop\PR_final\features.npy')
gender_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\binary_gender_labels.npy'))
age_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\age_labels.npy'))
id_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\id_labels.npy'))
flatten_features = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\pca_features.npy'))
inds = [i for i in range(len(gender_labels))]
shuffle(inds)
flatten_features = np.array([flatten_features[j] for j in inds])
gender_labels = np.array([gender_labels[j] for j in inds])
age_labels = np.array([age_labels[j] for j in inds])
id_labels = np.array([id_labels[j] for j in inds])

ard_id = []
ami_id = []
h_id = []
c_id = []
v_id = []
fm_id = []

ard_age = []
ami_age = []
h_age = []
c_age = []
v_age = []
fm_age = []

ard_g = []
ami_g = []
h_g = []
c_g = []
v_g = []
fm_g = []

comps = [2, 5, 10, 50, 100, 200, 292, 400]
for c in comps:
    kmeans = KMeans(c, random_state=0)
    labels = kmeans.fit(flatten_features).predict(flatten_features)
    cluster_avg_bd = []
    mses = 0
    for i in range(c):
        cluster = list(np.where(labels == i)[0])
        bds = [age_labels[i] for i in cluster]
        if len(bds) != 0:
            cabd = sum(bds) / len(bds)
            cluster_avg_bd.append(cabd)
            mse = sqrt(metrics.mean_squared_error(bds, len(bds)*[cabd]))
            # print(mse)
            mses += mse
    # print(cluster_avg_bd)
    print(mses / c)

    ard = metrics.adjusted_rand_score(id_labels, labels)
    ami = metrics.adjusted_mutual_info_score(id_labels, labels)
    hcv = metrics.homogeneity_completeness_v_measure(id_labels, labels)
    fm = metrics.fowlkes_mallows_score(id_labels, labels)
    
    ard_id.append(ard)
    ami_id.append(ami)
    h_id.append(hcv[0])
    c_id.append(hcv[1])
    v_id.append(hcv[2])
    fm_id.append(fm)

    print("Identity: \n")
    print("adjusted_rand_score: ", ard, "\n")
    print("adjusted_mutual_info_score", ami, "\n")
    print("homogeneity_completeness_v_measure: ", hcv, "\n")
    print("fowlkes_mallows_score", fm, "\n")

    ard = metrics.adjusted_rand_score(age_labels, labels)
    ami = metrics.adjusted_mutual_info_score(age_labels, labels)
    hcv = metrics.homogeneity_completeness_v_measure(age_labels, labels)
    fm = metrics.fowlkes_mallows_score(age_labels, labels)

    ard_age.append(ard)
    ami_age.append(ami)
    h_age.append(hcv[0])
    c_age.append(hcv[1])
    v_age.append(hcv[2])
    fm_age.append(fm)

    print("Age: \n")
    print("adjusted_rand_score: ", ard, "\n")
    print("adjusted_mutual_info_score", ami, "\n")
    print("homogeneity_completeness_v_measure: ", hcv, "\n")
    print("fowlkes_mallows_score", fm, "\n")

    ard = metrics.adjusted_rand_score(gender_labels, labels)
    ami = metrics.adjusted_mutual_info_score(gender_labels, labels)
    hcv = metrics.homogeneity_completeness_v_measure(gender_labels, labels)
    fm = metrics.fowlkes_mallows_score(gender_labels, labels)

    ard_g.append(ard)
    ami_g.append(ami)
    h_g.append(hcv[0])
    c_g.append(hcv[1])
    v_g.append(hcv[2])
    fm_g.append(fm)

    print("Gender: \n")
    print("adjusted_rand_score: ", ard, "\n")
    print("adjusted_mutual_info_score", ami, "\n")
    print("homogeneity_completeness_v_measure: ", hcv, "\n")
    print("fowlkes_mallows_score", fm, "\n")

ard1 = plt.plot(comps, ard_id, label = 'Adjusted rand index')
ami1 = plt.plot(comps, ami_id, label = 'Adjusted mutual information')
h1 = plt.plot(comps, h_id, label = 'Homogeneity')
c1 = plt.plot(comps, c_id, label = 'Completeness')
v1 = plt.plot(comps, v_id, label = 'V-measure')
fm1 = plt.plot(comps, fm_id, label = 'Fawlkes Mallows score')
plt.legend()
plt.show()
ard2 = plt.plot(comps, ard_age, label = 'Adjusted rand index')
ami2 = plt.plot(comps, ami_age, label = 'Adjusted mutual information')
h2 = plt.plot(comps, h_age, label = 'Homogeneity')
c2 = plt.plot(comps, c_age, label = 'Completeness')
v2 = plt.plot(comps, v_age, label = 'V-measure')
fm2 = plt.plot(comps, fm_age, label = 'Fawlkes Mallows score')
plt.legend()
plt.show()
ard3 = plt.plot(comps, ard_id, label = 'Adjusted rand index')
ami3 = plt.plot(comps, ami_id, label = 'Adjusted mutual information')
h3 = plt.plot(comps, h_id, label = 'Homogeneity')
c3 = plt.plot(comps, c_id, label = 'Completeness')
v3 = plt.plot(comps, v_id, label = 'V-measure')
fm3 = plt.plot(comps, fm_id, label = 'Fawlkes Mallows score')
plt.legend()
plt.show()