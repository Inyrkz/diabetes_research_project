# diabetes_research_project
This research focuses on using some machine learning algorithms to diagnose if a patient has diabetes.


## Methodology
The  Pima Indians Diabetes Dataset was used for this project. It has two classes which are normal patients and diabetic patients.
There are 500 records of healthy patients and 268 records of diabetic patients. This makes this dataset imbalanced. The proportion of the classes are shown in the pie chart below.


![alt text](image.jpg)
*Pie Chart of the Classes in the Dataset*


### Dataset Features
The features of the dataset are: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
Standardization was applied to the dataset to keep all the features on the same scale. It also helps to speed up the training process.


### Training Model
The dataset was spitted into the training set and the validation set in the ratio of 75:25.

The machine learning algorithms used for training are `Logistic Regression`, `K Nearest Neighbors (KNN)`, `Support Vector Machine (SVM)`, and `Random Forest.`

The metrics used for evaluation are `accuracy`, `recall`, `precision` and `f1-score.`


| ML Algorithms | Hyper-parameters used |
| ----------- | ----------- |
| Logistic Regression | C: 0.5, penalty: 'l2' |
| SVM | kernel: 'rbf’ |
| KNN | n_neighbors=9, metric='minkowski', p=2 |
| Random Forest | criterion: 'entropy', min_samples_split: 30, n_estimators: 110 |

*Table 1: Hyperparameters settings *


## RESULTS

| Models | Accuracy | Precision | Recall | F1-Score |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Logistic Regression | 80.73 | 79.57 | 76.89 | 77.87 |
| SVM | 79.17 | 77.29 | 76.04 | 76.56 |
| KNN | 80.73 | 79.17 | 77.58 | 78.23 |
| Random Forest | 81.25 | 79.86 | 77.98 | 78.73 |

* Table 1: Validation  Accuracy, Precision, Recall and F1-scores *




**CONFUSION MATRIX**


![alt text](image.jpg)
Figure 4.0: Logistic Regression Confusion Matrix


![alt text](image.jpg)
Figure 4.1: SVM Confusion Matrix


![alt text](image.jpg)
Figure 4.2: KNN Confusion Matrix


![alt text](image.jpg)
Figure 4.3: Random Forest Confusion Matrix


| Models | True Positives | False Positives | True Negatives | False Negatives |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Logistic Regression | 43 | 13 | 112 | 24 |
| SVM | 44 | 17 | 108 | 23 |
| KNN | 45 | 15 | 110 | 22 |
| Random Forest | 45 | 14 | 111 | 22 |

** Table 3: True Positives, False Positives, True Negatives, False Negatives **


In healthcare problems, the goal is to reduce the number of false negatives. The false negatives are the patients our model predicted to be healthy, but the fact is that these patients have diabetes. This is why the recall metric is highly considered in the healthcare industry. The higher the recall, the lower the false negatives. The machine learning models that gave us the lowest number of false negatives are the KNN and the Random Forest. 



![alt text](image.jpg)
Figure 4.4: Accuracy chart for each ML model.



![alt text](image.jpg)
Figure 4.5: Precision chart for each ML model.



![alt text](image.jpg)
Figure 4.6: Recall chart for each ML model.


![alt text](image.jpg)
Figure 4.7: F1-Scorel chart for each ML model.


![alt text](image.jpg)
Figure 4.8: Summary of validation accuracy, precision, recall, f1-score  for each ML model.

