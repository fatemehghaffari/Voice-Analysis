# Voice-Analysis
This is the entire process of implementing a voice recognition project, from feature extraction and selection to classification and evaluation. Regarding the lack of proper data, only age recognition and clustering were performed. However, other analyses can be implemented the same way.
## Conceptual Design
Train:

![train](/img/train.png)

Test:

![train](/img/test.png)

Clustering:

![train](/img/cluster.png)
## Data Statistical Analysis
Gender Distribution:

![gender](/img/gender.png)

Age Distribution:

![age](/img/age.png)

## Feature Extraction
In this project the first 40 MFCC features are used. Please consider that the first 20 MFCC features, which may be the default ones, are used for linguistic applications and are independant from personal features of the voice such as age and gender. Therefore, they can be ignored completely. However, it won't hurt to include them. 
## Classification
The GMM classifier is well-suited for the MFCC features. The covariance type is selected as "diag" to avoid memory error. 
Classification Results:


## Clustering
