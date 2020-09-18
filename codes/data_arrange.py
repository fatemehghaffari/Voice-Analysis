import os
import librosa as rs
from pathlib import Path
import soundfile as sf
import numpy as np
import pandas as pd

# current directory
cwd = os.getcwd()
dir = os.path.join(cwd,"Data_Upload")

# loading saved features after an error
# features = np.load(r'C:\Users\Data Science\Desktop\PR_final\features.npy')
# gender_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\gender_labels.npy'))
# age_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\age_labels.npy'))
# id_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\id_labels.npy'))

# initializing
features = np.zeros([2859, 40, 938])
gender_labels = []
age_labels = []
age_of_persons = []
num_of_audios = []
id_labels = []
itera = 0
summ = 0
index = 0
identity = -1

# walk through all the folders and files
for subdir, dirs, files in os.walk(dir):
    x = 0
    itera += 1
    if itera == 1:
        number_of_sets = len(dirs)
    print("subdir: ", subdir)
    print("dirs", dirs)
    print("files", files)
    for file in files:
        typ = os.path.join(subdir, file).split('.')[-1]
        if typ == 'csv':
            df = pd.read_csv(os.path.join(subdir, file))
            diff_persons_in_set = list(df['g'])
            diff_persons_in_set_bd = list(df['bd'])
        else:
            number = int(os.path.join(subdir, file).split('\\')[len(os.path.join(subdir, file).split('\\')) - 2])
            if x == 0:
                identity += 1
                x = 1
                age_of_persons.append(diff_persons_in_set_bd[number - 1])
                num_of_audios.append(len(files))
            id_labels.append(identity)
            set_num = int(os.path.join(subdir, file).split('\\')[len(os.path.join(subdir, file).split('\\')) - 3])
            # summ = summ + 1  #executed once to get the number of samples
            gender_labels.append(diff_persons_in_set[number - 1])
            age_labels.append(diff_persons_in_set_bd[number - 1])
            y, sr = rs.load(os.path.join(subdir, file), sr = 8000)
            # reduce the duration to one minute
            if len(y) < 60*8000:
                z = np.zeros(60*8000 - len(y))
                y = np.array(list(y.T) + list(z)).T
            M = rs.feature.mfcc(y[:60*8000], n_mfcc=40)
            features[index] = M
            index += 1
            print("index", index)
            print("label index:", len(gender_labels))
            print(os.path.join(subdir, file))
    np.save(r'C:\Users\Data Science\Desktop\PR_final\features.npy', features)
    np.save(r'C:\Users\Data Science\Desktop\PR_final\gender_labels.npy', gender_labels)
    np.save(r'C:\Users\Data Science\Desktop\PR_final\age_labels.npy', age_labels)
    np.save(r'C:\Users\Data Science\Desktop\PR_final\id_labels.npy', id_labels)
np.save(r'C:\Users\Data Science\Desktop\PR_final\age_of_persons.npy', age_of_persons)
np.save(r'C:\Users\Data Science\Desktop\PR_final\num_of_audios.npy', num_of_audios)
np.save(r'C:\Users\Data Science\Desktop\PR_final\gender_labels.npy', gender_labels)
np.save(r'C:\Users\Data Science\Desktop\PR_final\features.npy', features)
np.save(r'C:\Users\Data Science\Desktop\PR_final\age_labels.npy', age_labels)
print(age_of_persons)
print(num_of_audios)

# relabelling
# male -> 0     female -> 1
new_gender_labels = []
for i in range(len(gender_labels)):
    gl = gender_labels[i].strip()
    if (gl == 'M') | (gl == 'Male') | (gl == 'male') | (gl == 'm'):
        gl = 0
    elif (gl == 'F') | (gl == 'Female') | (gl == 'female') | (gl == 'f') | (gl == 'W'):
        gl = 1
    else:
        print(gl)
    new_gender_labels.append(gl)
np.save(r'C:\Users\Data Science\Desktop\PR_final\binary_gender_labels.npy', new_gender_labels)