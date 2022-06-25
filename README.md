# Brain_stock_research

## Purpose
- 데이터셋을 이용하여 뇌졸중을 가장 잘 예측할 수 있는 모델 만들기

## Data
- https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

## 변수설명

1) id: unique identifier

2) gender: "Male", "Female" or "Other"

3) age: age of the patient

4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension(고혈압)

5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease(심장 질환)

6) ever_married: "No" or "Yes"

7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"

8) Residence_type: "Rural" or "Urban"

9) avg_glucose_level: average glucose **level** in blood(혈액에 포도당 평균)

10) bmi: body mass index

11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*

12) stroke: 1 if the patient had a stroke or 0 if not

- Note: "Unknown" in smoking_status means that the information is unavailable for this patient

## Feature analysis
- 수치형 변수들 간의 상관분석 시행

## PreProcessing
- bmi 결측치 -> 평균값으로 대체
- 라벨인코딩, 변수 스케일링 사용
- SMOTE 알고리즘을 이용하여 label 불균형 해결

## Model
- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- Gaussian NB

## 

https://fuchsia-runner-4af.notion.site/beb770371b08409b91b5df097e207d69
