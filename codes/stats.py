import numpy as np
import matplotlib.pyplot as plt

features = np.load(r'C:\Users\Data Science\Desktop\PR_final\features.npy')
gender_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\binary_gender_labels.npy'))
age_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\age_labels.npy'))
id_labels = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\id_labels.npy'))
age_of_persons = list(np.load(r'C:\Users\Data Science\Desktop\PR_final\age_of_persons.npy'))
num_of_audios = np.load(r'C:\Users\Data Science\Desktop\PR_final\num_of_audios.npy')
print("Average Age: ", 1399 - (sum(age_of_persons) / len(age_of_persons)), "\n")
print("Min Age: ", 1399 - max(age_of_persons), "\n")
print("Max Age: ", 1399 - min(age_of_persons), "\n")
print("Male percentage: ", 100 - ((sum(gender_labels) * 100) / len(gender_labels)), "% \n")
print("Male percentage: ", ((sum(gender_labels) * 100) / len(gender_labels)), "% \n")
print("Average number of audio files per person: ", sum(num_of_audios) / len(num_of_audios))
bd_rng = list(range(min(age_of_persons), max(age_of_persons)))
age_rng = 1399 - np.array(bd_rng)
real_age = 1399 - np.array(age_of_persons)
ages = []
freqs = []
for aop in age_of_persons:
    if aop not in ages:
        ages.append(aop)
        freqs.append(1)
    else:
        freqs[ages.index(aop)] += 1
zipped_lists = zip(ages, freqs)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
ages, freqs = [ list(tuple) for tuple in  tuples]
plt.plot(1399 - np.array(ages), freqs)
plt.xlabel("Age")
plt.ylabel("Number of persons")
plt.title("Age distribution")
plt.show()