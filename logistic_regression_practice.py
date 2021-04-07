# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:55:36 2021

@author: f
"""

import pandas as pd

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

## 당뇨 데이터 불러오기
pima = pd.read_csv("C:\stroke_dataset\diabetes.csv", header=None, names=col_names)
pima.info()
pima = pima.drop(0,)
pima
## 판다스 데이터 프레임
type(pima)

## 칼럼 이름 확인
pima.columns

feature_names = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree', 'skin']

# 종속 변수와 독립 변수 나누기
X = pima[feature_names] ## 독립 변수
y = pima.label ## 타겟 변수


## 데이터셋을 트레인 셋과 테스트 셋으로 나누기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


## logistic regression 모델 만들기
from sklearn.linear_model import LogisticRegression

## 기본 파라메터 모델 만들기
logreg = LogisticRegression()

## 모델에 값 적합하기
logreg.fit(X_train, y_train)

## 모델 예측하기
y_pred=logreg.predict(X_test)


## 혼동행렬을 사용한 모델 평가

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

## 히트맵을 사용하여 혼동행렬 시각화
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   ## 통계 데이터 시각화 패키지


class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


## create heatmap ##시각화는 한번에 실행하기
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##혼동행렬평가지표
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
