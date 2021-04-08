# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:53:52 2021

@author: f
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder ## 레이블 인코더 패키지
from sklearn.linear_model import LogisticRegression ##  logisticRegression 패키지
from sklearn.model_selection import train_test_split ## train_test 셋 나눌때 패키지
from sklearn.preprocessing import StandardScaler ## 정규화 패키지

stroke = pd.read_csv("C:\stroke_dataset\healthcare-dataset-stroke-data.csv")

## 데이터 정보확인
stroke.head()
stroke.info()
len(stroke) ## 행이 5110개
len(stroke.columns) ## 칼럼이 12개

## id값 드랍하기
stroke = stroke.drop('id', axis=1)
stroke


## 데이터 NULL값 유무 확인 
stroke.isnull().sum() ## bmi에 null 값 201개 확인
stroke["bmi"]

## stroke bmi null값을 평균값으로 채워넣기
stroke["bmi"] = stroke["bmi"].fillna(stroke["bmi"].mean())

## 레이블 인코더를 사용해 모델에 사용할 수 있도록 데이터 바꾸기(gender, ever_married, work_type, Residence_type, smoking_status)
le = LabelEncoder()
le.fit(stroke.gender)
le.classes_
stroke.gender = le.transform(stroke.gender) ## Female, Male, Other


le.fit(stroke.ever_married)
le.classes_
stroke.ever_married = le.transform(stroke.ever_married) ## No, Yes


le.fit(stroke.work_type)
le.classes_
stroke.work_type = le.transform(stroke.work_type) ## Govt_job, Never_worked, Private, Self-employed, children

le.fit(stroke.Residence_type)
le.classes_
stroke.Residence_type = le.transform(stroke.Residence_type) ## Rural, Urban

le.fit(stroke.smoking_status)
le.classes_
stroke.smoking_status = le.transform(stroke.smoking_status) ## Unknown, formerly smoked, never smoked,  smokes


## train set과 test 셋으로 나누기
stroke.columns
features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']

X = stroke[features]
y = stroke.stroke

## train_test_split을 이용해 test사이즈는 30퍼로 나누며 random_state를 통해 난수 고정
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50)
X_train, X_test, y_train, y_test

## 기본 파라메터 모델 만들기
model = LogisticRegression()

## 모델에 값 적합하기
model.fit(X_train, y_train)

## 모델 예측하기
y_pred = model.predict(X_test)

print(model.score(X_train, y_train)) ## 94.85 정분류율
print(model.score(X_test, y_test)) ## 95.69 정분류율

## 혼동행렬을 사용한 모델 평가

from sklearn import metrics
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
cf_matrix

## 히트맵을 사용하여 시각화
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names) ## x축 눈금 메기기
plt.yticks(tick_marks, class_names) ## y축 눈금 메기기

## heatmap 시각화 하기
sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix", y=1.1)
plt.yalbel('Actual label')
plt.xlabel('Predicted label')

## 혼동행렬 평가지표
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
