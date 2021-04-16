# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:17:13 2021

@author: f
"""

import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('C:\stroke_dataset\healthcare-dataset-stroke-data.csv')

dataset

dataset.info()

dataset.isnull().sum()


dataset["bmi"].replace(to_replace=np.nan, value = dataset.bmi.mean(), inplace=True)

#%% 데이터 시각화 
dataset.isnull().sum()

## 데이터셋 요약본 보기
dataset.describe()

## feature들 간의 상관계수
dataset.corr()


## Heat map으로 상관행렬 나타내기
corr = dataset.corr()

## 상삼각 행렬
mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11,9))

cmap = sns.diverging_palette(230, 20, as_cmap = True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square = True, linewidths=.5, cbar_kws={"shrink":.5})

## gender
print(dataset.gender.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=dataset, x="gender")

## hypertension
print(dataset.hypertension.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=dataset, x="hypertension")

## married
print(dataset.ever_married.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=dataset, x="ever_married")

## work type
print(dataset.work_type.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=dataset, x="work_type")

## residence type
print(dataset.Residence_type.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=dataset, x="Residence_type")

## smoking status
print(dataset.smoking_status.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=dataset, x="smoking_status")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)

## Stroke
print(dataset.stroke.value_counts())
sns.set_theme(style="darkgrid")
ax = sns.countplot(data=dataset, x="stroke")

#######################################분포확인############################################
### 분포 플랏
fig = plt.figure(figsize=(7,7))
sns.distplot(dataset.avg_glucose_level, color="green", label="avg_glucose_level", kde=True)

## bmi 분포보기
fig = plt.figure(figsize=(7,7))
sns.distplot(dataset.bmi, color="orange", label="bmi", kde=True)

plt.figure(figsize=(12,10))

sns.distplot(dataset[dataset['stroke']==0]["bmi"], color="green")
sns.distplot(dataset[dataset['stroke']==1]["bmi"], color="red")

plt.title('No Stroke vs Stroke by BMI', fontsize=15)
plt.xlim([10,100])

## 포도당 분포에 의한 stroke 분포보기
plt.figure(figsize=(12,10))

sns.distplot(dataset[dataset['stroke']==0]["avg_glucose_level"], color='green')
sns.distplot(dataset[dataset['stroke']==1]["avg_glucose_level"], color="red")

plt.title('No Stroke vs Stroke by Avg. Clucose Level', fontsize=15)
plt.xlim([30,330])


## 나이에 의한 stroke 분포보기
plt.figure(figsize=(12,10))

sns.distplot(dataset[dataset['stroke']==0]["age"], color="green")
sns.distplot(dataset[dataset['stroke']==1]["age"], color="red")
plt.title('No Stroke vvs Stroke by Age', fontsize=15)
plt.xlim([18,100])

## Scatter plot
## Age vs BMI
fig = plt.figure(figsize=(7,7))
graph = sns.scatterplot(data=dataset, x="age", y="bmi", hue="gender")
graph.axhline(y=25, linewidth=4, color="r", linestyle='--')

#  Age vs Avg.Glucose Level
fig = plt.figure(figsize=(7,7))
graph = sns.scatterplot(data=dataset, x="age", y="avg_glucose_level", hue="gender")
graph.axhline(y=150, linewidth=4, color="r", linestyle="--")


## Violin Plot
plt.figure(figsize=(13,13))
sns.set_theme(style="darkgrid")
plt.subplot(2,3,1)
sns.violinplot(x='gender', y='stroke', data=dataset)
plt.subplot(2,3,2)
sns.violinplot('hypertension', y='stroke', data=dataset)
plt.subplot(2,3,3)
sns.violinplot("heart_disease", y="stroke", data=dataset)
plt.subplot(2,3,4)
sns.violinplot(x='ever_married', y="stroke", data=dataset)
plt.subplot(2,3,5)
sns.violinplot(x='work_type', y="stroke", data=dataset)
plt.xticks(fontsize=9, rotation=45)
plt.subplot(2,3,6)
sns.violinplot(x = 'Residence_type', y='stroke', data=dataset)


## Pair plot
fig = plt.figure(figsize=(10,10))
sns.pairplot(dataset)


# %% 데이터 전처리 
## 데이터 전처리
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0,5,9])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))


## 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 15] = le.fit_transform(x[:,15])
x[:, 16] = le.fit_transform(x[:, 16])
x

print('Shape of X: ', x.shape)
print('Shape of Y: ', y.shape)


## training set 과 tset 셋으로 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("Number transactions x_train dataset: ", x_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions x_test dataset: ", x_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

# 변수 스케일링
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

## SMOTE를 이용해 데이터 불균형 손보기
from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1' : {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0' : {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape))
print('After OverSampling, the shape of train_y: {}\n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1' : {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0' : {}".format(sum(y_train_res==0)))
fig = plt.figure(figsize=(10,10))
sns.countplot(y_train_res)


# %% 데이터 모델링
## Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score



## 모델링
models = []
models.append(['Logistic Regressionm', LogisticRegression(random_state=0)])
models.append(['SVM', SVC(random_state=0)])
models.append(['KNeighbors', KNeighborsClassifier()])
models.append(['GaussiianNB', GaussianNB()])
models.append(['BernoulliNB', BernoulliNB()])
models.append(['Decision Tree', DecisionTreeClassifier(random_state=0)])
models.append(['Random Forest', RandomForestClassifier(random_state=0)])
models.append(['XGBoost', XGBClassifier(eval_metric = 'error')])

list_1 = []

for m in range(len(models)):
    list_2 = []
    model = models[m][1]
    model.fit(x_train_res, y_train_res)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred) 
    accuracies = cross_val_score(estimator = model, X = x_train_res, y = y_train_res, cv = 10) ## 교차검증
    roc = roc_auc_score(y_test, y_pred) ## roc커브
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(models[m][0], ':')
    print(cm)
    print('Accuracy Score: ', accuracy_score(y_test, y_pred))
    print('')
    print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print('')
    print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))
    print('')
    print("ROC AUC Score: {:.2f}".format(roc))
    print('')
    print("Precision: {:.2f}".format(precision))
    print('')
    print("Recall: {:.2f}".format(recall))
    print('')
    print("F1: {:.2F}".format(f1))
    print('-------------------------------')
    print('')
    list_2.append(models[m][0]) 
    list_2.append((accuracy_score(y_test, y_pred))*100)
    list_2.append(accuracies.mean()*100)
    list_2.append(accuracies.std()*100)
    list_2.append(roc)
    list_2.append(recall)
    list_2.append(precision)    
    list_2.append(f1)
    list_1.append(list_2)


df = pd.DataFrame(list_1, columns =  ['Model', 'Accuracy', 'K-Fold Mean Accuracy', 'Std.Deviation','ROC AUC','Precision','Recall','F1'])
df.sort_values(by=['Accuracy', 'K-Fold Mean Accuracy'], inplace=True, ascending=False)    
df    

## Tning the models
from sklearn.model_selection import GridSearchCV

grid_models = [(LogisticRegression(), [{'C':[0.25, 0.5, 0.75,1], 'random_state':[0]}]),
               (KNeighborsClassifier(),[{'n_neighbors':[5,7,8,10], 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}]),
               (SVC(), [{'C':[0.25,0.5,0.75,1],'kernel':['linear','rbf'],'random_state':[0]}]),
               (BernoulliNB(), [{'alpha': [0.25, 0.5, 1]}]),
               (DecisionTreeClassifier(), [{'criterion':['gini','entropy'], 'random_state':[0]}]),
               (RandomForestClassifier(),[{'n_estimators':[100,150,200],'criterion':['gini','entropy'], 'random_state':[0]}]),
               (XGBClassifier(), [{'learning_rate': [0.01, 0.05, 0.1], 'eval_metric': ['error']}])]

for i,j in grid_models:
    grid = GridSearchCV(estimator=i, param_grid = j, scoring = 'accuracy', cv=10)
    grid.fit(x_train_res, y_train_res)
    best_accuracy = grid.best_score_
    best_param = grid.best_params_
    print('{}:\nBest Accuracy : {:.2f}%'.format(i, best_accuracy*100))
    print('Best Parameters : ', best_param)
    print('')
    print('-----------')
    print('')
    
    
# %% 모델 하이퍼파라미터 튜닝
## RandomForest 
classifier = RandomForestClassifier(criterion = 'gini', n_estimators = 100, random_state=0)
classifier.fit(x_train_res, y_train_res)
y_pred = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)[:,1]
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
print('ROC AUC score: {0}'.format(roc_auc_score(y_test, y_prob)))
print('Accuracy Score: ', accuracy_score(y_test, y_pred))

# Confusion Matrix 시각화
plt.figure(figsize = (8,5))
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}, 
            yticklabels = ['No stroke', 'Stroke'], xticklabels = ['Predicted no stroke', 'Predicted stroke'])
plt.yticks(rotation = 0)


# Roc AUC Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

sns.set_theme(style = 'white')
plt.figure(figsize = (8,8))
plt.plot(false_positive_rate, true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# %% Xgboost
classifier = XGBClassifier(eval_metric = 'error', learning_rate = 0.1)
classifier.fit(x_train_res, y_train_res)
y_pred = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)[:,1]
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
print('ROC AUC score: {0}'.format(roc_auc_score(y_test, y_prob)))
print('Accuracy Score: ', accuracy_score(y_test, y_pred))

# confusion matrix  시각화
plt.figure(figsize = (8,5))
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths=5, cbar = False, annot_kws = {'fontsize' : 15}, yticklabels=['No Stroke', 'Stroke'],
        xticklabels = ['Predicted no stroke', 'Predicted stroke'])
plt.yticks(rotation = 0)

# Roc Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

sns.set_theme(style = 'white')
plt.figure(figsize = (8,8))
plt.plot(false_positive_rate, true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# %% Keras ANN
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from keras.regularizers import l2
# from sklearn.model_selection import GridSearchCV

# ## ANN 설계
# def ann_classifier():
#     ann = tf.keras.models.Sequential()
#     ann.add(tf.keras.layers.Dense(units= 8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
#     ann.add(tf.keras.layers.Dense(units= 8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
#     tf.keras.layers.Dropout(0.6)
#     ann.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))
#     ann.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
#     return ann

# ann = KerasClassifier(build_fn = ann_classifier, batch_size = 32, epochs = 50)
# ## Cross Validation으로 ANN 평가하기
# accuracies = cross_val_score(estimator = ann, X = x_train_res, y = y_train_res, cv=5)

# mean = accuracies.mean()
# std_deviation = accuracies.std()
# print("Accuracy: {:.2f}%".format(mean*100))
# print('Standard Deviation: {:.2f}%'.format(std_deviation*100))

# ## ANN tuning
# def ann_classifier(optimizer = 'adam'):
#     ann = tf.keras.models.Sequential()
#     ann.add(tf.keras.layers.Dense(units= 8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
#     ann.add(tf.keras.layers.Dense(units= 8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
#     tf.keras.layers.Dropout(0.6)
#     ann.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))
#     ann.compile(optimizer= optimizer, loss= 'binary_crossentropy', metrics= ['accuracy'])
#     return ann

# ann = KerasClassifier(build_fn = ann_classifier, batch_size = 32, epochs = 50)

# parameters = {'batch_size': [25, 32],
#              'epochs': [50, 100, 150],
#              'optimizer': ['adam', 'rmsprop']}

# grid_search = GridSearchCV(estimator = ann, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)

# grid_search.fit(x_train_res, y_train_res)

# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
