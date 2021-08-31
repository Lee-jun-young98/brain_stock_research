# Machine-Learning-Based Approach to Differential Diagnosis in Tuberculous and Viral Meningitis

---

---

링크 : [https://www.icjournal.org/search.php?where=aview&id=10.3947/ic.2020.0104&code=0086IC&vmode=PUBREADER](https://www.icjournal.org/search.php?where=aview&id=10.3947/ic.2020.0104&code=0086IC&vmode=PUBREADER)

## 사용된 의학 용어

- Meningitis(수막염)
- lower serum sodium(혈청 나트륨)
- CSF(cerebrospinal fluid) : 뇌척수액
- CSF glucose : 뇌척수액 포도당
- CSF protein : 뇌척수액 단백질
- CSF adenosine deaminase : 뇌척수액 아데노신 디아미나제(ADA)
- PCR(polymerase chain reaction) : 중합효소 연쇄반응으로 바이러스성 뇌수막염을 진단할 때 사용한다.

## 논문의 목적

### 배경

- 결핵성 수막염(TBM)은 결핵이 가장 심한 형태지만 TBM 진단과 바이러스성 수막염(VM)의 구별은 어렵다. 따라서 TBM과 VM을 차별화하기 위한 머신 러닝 모듈을 개발했다.
- 결핵성 수막염이란?
    - [http://www.amc.seoul.kr/asan/healthinfo/disease/diseaseDetail.do?contentId=31672](http://www.amc.seoul.kr/asan/healthinfo/disease/diseaseDetail.do?contentId=31672)
    - 결핵균이 뇌척수막을 침입하여 염증을 일으키는 질환.
- 바이러스성 수막염이란?
    - [https://ko.wikipedia.org/wiki/바이러스성_수막염](https://ko.wikipedia.org/wiki/%EB%B0%94%EC%9D%B4%EB%9F%AC%EC%8A%A4%EC%84%B1_%EC%88%98%EB%A7%89%EC%97%BC)
    - 바이러스 감염에 의해 발생하는 수막염의 일종으로, 뇌와 척수를 감싸고 있는 뇌척수막에 발생한 염증을 의미.

### 데이터를 얻은 방법

- train 데이터의 경우 2000년 1월부터 2018년 7월까지 국내 5개 교원병원에서 TBM 및 확진 VM 사례가 소급하여 수집되었다. 훈련을 위해 다양한 기계학습 알고리즘이 사용 되었다. 기계학습 알고리즘은 leave-one-out- cross-validation에 의해 test 되었다. 4명의 레지던트와 2명의 감염병 전문가가 요약한 의료정보를 검사했다.

### 결과

- TBM이 확인되거나 있을 가능성이 있는 60명의 환자와 VM이 확인된 143명의 환자로부터 데이터를 구성했다.
- TBM 환자에게는 노년기, 방문 전 증상 지속시간, 혈청나트륨(lower serum sodium), 낮은 뇌척수액 포도당(lower cerebrospinal fluid(CSF) glucose), 높은 CSF 단백질(higher CSF protein), CSF 아데노신 디아미나제(CSF adenosine deaminase)이 발견되었다.
- 다양한 머신러닝 알고리즘 중 matrix completion을 위해 ImperativeImputer와 사용한 ANN의 AUC값이 가장 높은 것으로 나타났다.
- ANN의 모델의 AUC값은 모든 레지던트(0.67~0.72 P = 0.001)와 감염병 전문의 (0.76, P = 0.03)보다 통계적으로 높았다.

### 결론

- 머신 러닝 기술은 TBM과 VM을 구별하는 역할을 할 수 있다. 구체적으로, ANN 모델은 비전문가 임상의보다 더 나은 진단 성능을 가지고 있는 것으로 보인다.

## Introduction(소개)

1. 결핵성 뇌막염(TBM)은 결핵의 가장 심각한 형태이며 뇌막염의 염증을 유발한다. TBM은 모든 결핵 사례의 1%와 면역능력자의 모든 외폐질환의 5%를 차지한다. 전 세계적으로 매년 100,000건 이상의 TBM 사례가 발생한 것으로 추산. TBM 환자는 처음에는 나른함, 두통, 저급열이 나타나며 메스꺼움, 구토, 혼란스러운 정신 상태가 질병이 진행되면서 발생하며 이로 인해 혼수상태, 발작, 신경학적 손상으로 이어진다. 의학의 발전에도 불구하고 치명적인 위험이 높으며 조기진단과 항진균 요법이 매우 중요하다.
2. TBM과 VM은 초기 유사한 임상 징후 때문에 구별하기 어렵다. TBM 및 VM용 진단 툴의 민감도가 매우 낮다. TBM을 진단하기 위한 빠르고 구체적인 방법은 CSF에서 항진균검사를 하는 것이며 최저 30%정도로 낮은 민감도를 가지고 있다.  TBM의 결정적 진단도구로 꼽히는 CSF mycoboacterial culture도 민감도가 낮아 최대 2개월까지 배양해야 한다. 핵산증폭테스트(NAAT)와 Xpert MTB/Rif polymerase chain reaction(PCR) 기반 분석은 매우 구체적이지만 TBM 진단에 민감한 도구는 아니다. 이러한 이유로 TBM과 VM의 차등 진단은 임상의의 판단에 따라 달라진다.
3. 머신러닝 기술은 차별 문제를 해결하는데 유용하며 머신 러닝을 사용하는 진단 모듈의 개발은 의학 분야에서 활발한 연구 주제이다. TBM과 VM의 차별화도 차별의 문제이기 때문에 빠른 판단이 필요한 상황에서 머신러닝 기법이 진단을 하는데에 보조적인 역할을 보인다. 딥러닝은 많은 기존 머신러닝 모델에 비해 우수한 성능(예 : 정확도) 때문에 특히 매력적이다. 그럼에도 불구하고 TBM고 VM을 구별하기 위한 딥러닝 기법의 진단 역할은 존재하지 않는 것으로 보인다.

 4. 따라서 TBM과 VM을 차별화하기 위해 딥러닝 모델을 포함한 다양한 머신러닝 기술을 조사하고 그 결과를 임상의사가 수행한 진단과 비교했다.

## Materials and Methods

### 1. Study design & population

- 국내 5개 교원병원(침대 수 642-2705명)에서 소급코호트 연구 실행
- 2000년 1월부터 2018년 7월까지 4가지의 조건을 충족한 의료기록을 검토함
    1. CSF에서의 결핵증식
    2. M.tuberculosis(결핵) 양성 핵산증식검사
    3. 결핵성 뇌수막염 진단코드(KCDA170)
    4. 단순 뇌수막염 양성 핵산증식 검사 ex) HSV(단순 헤르포진 바이러스), VZV(수두 대상 포진 바이러스,varisella-zoster virus) 또는 엔테로바이러스(Enterovirus)
- 제외 기준은 검토 했을 때 불안전한 의료 기록, CSF 유체 분석 부족 및 18세 미만의 환자
- 순천향대 부천 병원(2018-09-026)의 승인을 받았다. 본 연구의 소급적 성격, 환자 개입 및 추가 검체 수집이 없기 때문에 사전 동의 요건이 면제되었으며, 인간 참여자를 포함하는 모든 철자는 기관 및/또는 국가 연구 위원회의 윤리 표준과 2013년 헬싱키 선언 및 이후 개정 또는 이에 상응하는 윤리 표준에 따라 수행되었다.

### 2. Definition and data collection

- 수막염은 CSF 백혈구 수 > 5 cells/mm3(입방 밀리미터)으로 정의 되었으며, 두 개 이상의 소견(두통, 메스꺼움/구토증, 광포증, 목 뻣뻣함, 발열 > 38도)로 정의되었다.
- HSV, VZV, 장내 바이러스 PCR에 대한 양성 CSF PCR 결과를 나타내는 임상적 표시가 있는 환자들은 바이러스 수막염을 확인했다.
- CNS(중추 신경 계통) 감염을 나타내는 임상적 표시가 있는 환자들은 CSF 시료가 배양이나 PCR 분석에 근거해 M. tuberculosis(결핵)에 양성인 경우 TBM을 확인했다. CNS 감염과 다른 체액 배양에 대한 임상적 표시가 있는 환자들은 M.tuberculosis에 positive였다. 폐결핵은 뇌수막염의 다른 알려진 병리 없이 TBM일 가능성이 있다.
- 사례 수가 충분하지 않았기 때문에, TBM과 VM을 구별하는 데 유용한 특성으로 알려진 것만 포함했다. 이전 연구를 기반으로 나이, 증상 출현 및 징후 발생부터 병원 방문, 구토, 신경 증상 및 징후, 혈청 나트륨, CSF 포도당, CSF 단백질, CSFA(ADA)에 대한 데이터를 기계 학습의 차별적 특징으로 수집하였다. 신경학적 증상과 징후는 무기력증, 혼란증, 뇌신경마비, 혈우병, 환경증, 망상증, 혼수상태, 발작, 반신불수, 마비증 또는 마비증 중 하나에 따라 정의되었다.

### 3. Model development and validation

- 나이브베이즈, 로지스틱 회귀분석,  랜덤포레스트, 서포트벡터머신, ANN 사용
- ANN은 알파 0.0001의 L2 정규화를 사용
- leave-one-out cross-validation(LOO-CV) 사용 203번의 교차검증
- ANN을 제외한 머신러닝 모델은 Weka tool에서 사용
- ANN은 Tensorflow 1.12를 사용
- i7-7700에서  3.6GHz의 8개의 CPU와 2개의 Geforce 1080Ti GPU를 사용
- 머신러닝 모델이외에도 4년차 레지던트와 10년차 이상 경력이 있는 감염병 전문가 2명 총 6명의 임상실험 결과를 수집했다.
- 모델 파라미터
    - 랜덤포레스트
        - Maximum number of trees :
    - 나이브베이즈
        - No kernel estimator,  정규분포 사용
    - 로지스틱 회귀분석
        - Ridge 사용 1.0 × 10−8
        - Training algorithm: Broyden–Fletcher–Goldfarb–Shanno
    - 서포트 벡터 머신
        - Training algorithm: Sequential minimal optimization
        - C 값 1
        - epsilon: 1.0 × 10−12
        - Kernel: PolyKernel (exponent: 1.0)
    - ANN
        - Hidden layers : [20, 5]
        - Activation function : ReLU
        - number of epoch : 250
        - Traning algorithm: Adam(initial learning rate : 0.001)

### 4.  Statistics

- 모든 통계 분석은 SPSS Statistics 버전 25.0과 MedCalc 버전 19.3을 사용하여 계산되었다.
- 범주형 변수는 카이-제곱 검정 또는 피셔의 정확 검정을 사용했다.
- 연속형 변수는  Mann-Whitney U 검정을 사용하여 분석했다.
- A non-significant Little’s missing complexly at random test에서 X^2 = 27.244, df = 22, P = 0.2의 MCAR 패턴을 나타냈다.
- 데이터 수가 적어서 결측값 대체 알고리즘을 사용하여 모든 데이터를 유지했다.
- 결측값을 채우기 위한 알고리즘을 사용할 때 기계학습모델 성능(예 : 정확도)는 어떤 알고리즘을 사용하는지에 따라 달라진다. 그래서 3가지의 알고리즘을 적용했다.
    1. 반복적인 라운드 로빈 스케줄링을 통해 결측값을 산입하는 전략(IterativeImputer)
    2. soft thresholding of Singular Vector Decomposition을 사용하여 반복한다(SoftImputer)
    3. K-NN을 사용한 산입 전략(KnnImputer)
- Delong 방법은 수신기 연산자 특성의 곡선 아래 면적(AUC)를 계산하는데 사용되었다.
- 기계학습 간에 AUC 값의 차이는 통계적으로 미미하지만 인간의 판단과 비교하기 위해 AUC값이 가장 높은 기계학습 모델을 사용
- Bootstrapping은 사람의 판단과과 머신러닝 알고리즘 사이에 AUC값을 비교하기 위해 사용했다.
- Cohen's kappa 통계량은 진단 합치도를 분석하는데 사용 되었으며 모든 검정은 양측검증을 사용했고 P < 0.05 에서 유의했다.

![Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled.png](Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled.png)

- 결핵성 뇌수막염 39명 확진 21명 추측
    - TBM = TP, VM = FN
- 바이러스성 뇌수막염 143명 모두 확진
    - TBM = FP, VM = TN

## Results

- 437개의 환자 중 234명이 제외되었으며 203명의 환자 데이터가 사용됐다.
- TBM일 환자 60명과 VM이 확정된 환자 143명으로 이루어져 있다.
- 결핵 확진 환자 중 39명 중 M.결핵 단지는 5명의 환자만 CSF에서 배양 되었다.
- TBM의 연간 평균 환자는 3명이다.
- 144개의 VM 바이러스 현황
    - HSV : 49명
    - VZV : 65명
    - 엔테로바이러스 : 29명

![Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%201.png](Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%201.png)

- 환자의 평균 연령은 37세, 방문 전 질병 기간의 평균 기간은 5일
- 환자의 80%가 구토를 호소했고 70%가 신경증상과 징후를 보였다.
- TBM 환자에서는 VM 환자에 비해 노년기, 병원 방문 전 질병 지속기간, 잦은  신경증상 및 징후, 낮은 혈청나트륨, 낮은 CSF 포도당, 높은 CSF 단백질, 높은 CSF ADA를 보였다.

![Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%202.png](Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%202.png)

![Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%203.png](Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%203.png)

![Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%204.png](Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%204.png)

- 결측값 수
    - duration of illness : 1개
    - vomiting : 1개
    - serum sodium : 2개
    - CSF glucose : 1개
    - protein : 1개
    - CSF ADA : 16개
- SVM을 제외한 모든 머신러닝 모델은 Iterrative Imputer로 최고의 정확도를 달성
- Soft Imputer를 사용하는 NB의 민감도가 가장 높았고 SVM의 특이도(80)가 가장 높았다.
- KnImputer(2,3)를 사용하는 ANN의 정확도(87.7, 95)가 가장 높았다.
- TBM과 VM을 구별하기 위한 가장 높은 AUC 값은 IterativeImputer가 있는 ANN에서 발견되었다.

![Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%205.png](Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%205.png)

![Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%206.png](Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%206.png)

![Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%207.png](Machine-Learning-Based%20Approach%20to%20Differential%20Di%203e3a4a6f65454f90bd46615d06c63310/Untitled%207.png)

- IterativeImputer 모델이 있는 ANN의 AUC가 가장 높게 나왔으므로 인간과의 비교를 위한 기계학습 모델로 설정됨
- 레지던트들은 ID 전문가들보다 TBM에 덜 민감하게 진단함
- 레지던트는 53.7 전문의는 65 이상의 민감도를 얻음
- ANN 모델의 AUC는 모든 레지던트의 AUC보다 높았다.
- 전문가1보다는 통계적으로 높았고 전문가2랑은 비슷했다.
- 기계학습 전문가와 ID 전문가는 통계적으로 달랐다.

## Discussion

- ANN의 경우 비전문가 임상의보다 진단성능이 정확했으며 전문가의 판단에 필적한다.
- 이전 연구에서는 우수한 성능을 가졌지만 한계점이 있다.
    1. 적은 수의 사례에도 불구하고 너무 많은 변수로 구성되어 있다. → 과적합 발생 우려
    2. 한 센터에서 개발한 모델 성능은 다른 센터에서 보장 불가
    3. TBM에 대한 정의가 낮아 모델 성능이 과대평가 될 수 있다.
- 차별점, TBM에 대한 정의를 바로해 진단 성능이 약하지만 신뢰도를 높였으며, 모델에 포함된 변수는 전자 건강기록에서 획득한 데이터에서 쉽고 자율적으로 접근할 수 있다. → 임상 실무에서 응용프로그램을 개발하는 것이 편하다.
- TBM과 VM 사이의 '분류'를 위한 머신 러닝 모델의 실현가능성을 조사하기 위해 머신러닝 방법 선택
- 데이터 수가 적기 때문에 한계가 있다.
- 2개의 레이어가 있는 ANN이 문제 해결에 좋다는 것을 알고 LOOCV가 기계학습 분야에서 선호되어야 하지만 검증 데이터세트는 분리되지 않았다.
- 결론적으로 머신러닝으로 TBM고 VM을 차별화 할 수 있다. 기계 학습 알고리즘의 성능을 개선하고 실제 임상 실험에서 기계학습 알고리즘의 안전성과 유용성을 평가하기 위해 추가 연구가 수행되어야 한다.