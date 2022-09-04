---
published: true
layout: post
title: Oversampling
categories: [Project]
tags: [subject_project]
math: true
sitemap: false
---

# Project-Oversampling

# Classification for Imabalanced Data

머신러닝 분류(classifier) 문제에서 target 변수의 범주가 불균형(imbalance) 할 경우 

model의 학습 과정에서 major한 class의 데이터의 학습이 주로 이루어지므로 minor한 class의

학습이 제대로 이루어지 않는다.

이러한 imbalanced data의 경우 실제로 자주 마주할 수 있는 데이터라고 할 수 있고,

실질적으로 minor class를 잘 맞추는 것이 현실에서 주요한 머신러닝 과제라고 할 수 있다.

ex) 특정 질병을 예측하는 문제의 경우 질병을 가진 사람의 데이터가 소수 class로 이루어질 가능성이 크고, 이때 정상인지를 예측하는 것보다

질병을 예측하는 것에 focus를 맞추게 된다.

# classification of evaluation

본격적으로 머신러닝 모델에서 불균형 데이터를 처리하는 기법에 대해 설명하기에 앞서 분류 모델에서 사용되는 평가 지표에 대해

알아보고자 한다. 편의상 Binary classification 문제를 가정한다.

## Confusion-matrix(혼동 행렬)

혼동행렬은 머신러닝 분류 평가지표로 많이 활용되는 행렬으로 각 행은 실제 값을 각 열은 분류 모델에서의 예측값을 의미한다

![다운로드](/assets/images/oversampling/다운로드.png)

## Accuracy(정확도)

정확도는 전체 데이터 中 예측에 성공한 데이터의 비율을 의미한다. 

$ Accuracy = (TP+TN)/(TP+FP+TN+FN) $

이 때 class가 고르게 분포되어 있는 경우 정확도를 모델 평가 지표로 사용하는 것이 좋지만 class가 불균형한 형태의 경우

정확도는 해당 모델이 좋은 모델인지에 대한 올바른 판단을 제시해 줄 수 없다.

why? True와 False의 비율이 9:1이라 할 때 단순히 True라고 예측하기만 해도 90%의 정확도를 얻게 된다.

따라서 Imbalanced한 데이터의 경우 다른 지표를 고려해야 한다.

## Precision(정밀도)

예측 결과가 positive일 때 실제로 positive인 경우를 말한다.

$ Precision = TP /(TP + FP) $

## Sensitivity(민감도 = 재현율)

실제로 positive일 때 예측도 positive인 경우를 의미한다.

$ Sensitivity = TP / (TP+FN) $

## Specificity(특이도)

현실이 부정일 때 예측도 부정인 경우를 의미한다.

$ Specificity = TN / (TN+FP) $

## F1-SCORE

정밀도와 재현율은 일반적으로 trade-off 관계를 갖는다. 따라서 이 두 개의 지표를 모두 고려한 지표가 f1-score이고

정밀도와 재현율의 조화평균으로 표현된다.

민감도를 P 재현율을 R이라고 하면

$ f1-score = 2PR / (P+R) $

## ROC-CURVE

roc 커브는 가로축에는 거짓 긍정율 새로축에는 민감도가 배치가 되고 각각의 값들은 [0,1]의 범위를 갖는다.

이 때 cut-off-value(Threshold 라고도 함)의 변화에 따른 거짓 긍정율과 민감도 값을 그래프로 그린 것이 roc-curve이다.

$ y=x $는 임의의 분류 모델 즉 동전 던지기와 같으며 (0,1)의 점에 수렴할 수록 좋은 분류 모델을 의미하게 된다.

이 때 이 roc-curve의 면적을 AUC라고 하며 불균형한 데이터의 분류 모델의 지표로 많이 활용된다.



![다운로드 (1)](/assets/images/oversampling/다운로드 (1).png)

## How to modeling Imbalanced data

일반적으로 불균형 데이터를 모델링하는 방법은 크게 3가지가 존재한다.

## under sampling

이 방법의 경우 다수의 데이터를 소수의 데이터의 개수와 유사하게 맞추기 위해 다수의 데이터를 삭제하는 작업을 말한다.

이 경우 데이터의 개수가 줄어들기 때문에 다량의 loss-information이 발생한다는 단점이 있지만 

데이터의 개수를 줄임에 따라 모델링 시간이 단축된다는 장점도 있다.

## oversampling

이 방법의 경우 소수의 데이터를 특정 알고리즘에 의해 합성 데이터를 생성하여 다수의 데이터와 비슷한 데이터의 개수를 가지게

만드는 것을 의미한다. 이 경우 under-sampling 방법과는 달리 정보의 손실이 일어나지 않는 반면에 데이터를 늘리기 때문에 

모델링에 걸리는 시간이 늘어난다는 단점이 존재한다.

## Loss-function approach

이 방법의 경우 데이터를 늘리거나 줄이지 않고 모델링에서의 loss-function을 조정하는 방법이다.

일반적인 손실 함수의 경우 모든 데이터에 대해 동일한 가중치를 부여하여 계산하는 반면

불균형 데이터의 경우 소수 class 데이터를 예측하는 것에 틀리는 것에 좀 더 가중치를 부여하여 

모델의 소수 class 데이터에 대해 좀 더 잘 예측하도록 만든다.

본 포스팅에서는 oversampling 기법과 이를 python으로 활용하는 방법에 대해 다뤄보고자 한다.


# undersampling 과 oversampling

![다운로드 (2)](/assets/images/oversampling/다운로드 (2).png)

### DATA-SET : Kaggle Loan prediction based on customer behavior

![다운로드 (3)](/assets/images/oversampling/다운로드 (3).png)

# SMOTE

가장 대표적인 oversampling 방법이다. 대다수의 oversampling 기법이

이 SMOTE를 응용한 방법이라고도 할 수 있다. SMOTE에서 합성 데이터를 생성하는 방식은 다음과 같다.

1. 임의의 소수 데이터를 잡는다.

2. 해당 소수 데이터로부터 가장 가까운 k개의 이웃 데이터를 잡는다.

3. 기준이 되는 소수 데이터와 이웃이 되는 데이터 사이의 직선 상에서 합성 데이터를 랜덤하게 생성한다.

이를 그림으로 표현하면 다음과 같다.

![다운로드 (4)](/assets/images/oversampling/다운로드 (4).png)


```python
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
import time


from sklearn.preprocessing import LabelEncoder

df =pd.read_csv('C:/Users/koyounghun/Desktop/찌르레기/Loan_data/Loan_train.csv')
df = df.drop('Id',axis = 1)



labelEncoder = LabelEncoder()

data = df


for e in data.columns:
    if data[e].dtype == 'object':
        labelEncoder.fit(list(data[e].values))
        data[e] = labelEncoder.transform(data[e].values)
        
        # Accommodate the data that has been changed
        df = data
        
y = df.Risk_Flag
X = df.drop('Risk_Flag', axis=1)

import warnings
warnings.filterwarnings(action='ignore')

df['Risk_Flag'].value_counts() # 전체 데이터 중 약 10%만이 1로 코딩 
```




    0    221004
    1     30996
    Name: Risk_Flag, dtype: int64




```python
# smote 적용

start = time.time()


smote = SMOTE(random_state = 101)

X_smote, y_smote = smote.fit_resample(X, y)
end = time.time()

print(f"{end - start:.5f} sec")


np.unique(y_smote, return_counts = True) # 동일한 개수로 oversampling 되었음


```

    3.35296 sec





    (array([0, 1], dtype=int64), array([221004, 221004], dtype=int64))



# ADASYN

SMOTE를 개선한 방법이다. SMOTE의 경우 모든 소수 데이터에 대해 동일한 개수의 합성 데이터를 생성하는 반면에

ADASYN 기법은 주변의 다수 데이터를 고려하여 해당 데이터에서 생성할 데이터의 개수를 계산하고 이를 이용하여 

합성 데이터를 생성하는 방법이다.

즉 주변 데이터의 밀도에 따라 데이터를 생성하는 방법이라고 할 수 있다.

![다운로드 (5)](/assets/images/oversampling/다운로드 (5).png)


```python
from imblearn.over_sampling import ADASYN

start = time.time()

adasyn = ADASYN(random_state = 101)

X_over_ada, y_over_ada = adasyn.fit_resample(X, y)


end = time.time()

np.unique(y_over_ada, return_counts = True) # 동일한 개수로 oversampling 되었음


print(f"{end - start:.5f} sec")
```

    9.90477 sec


# Distribution SMOTE

합성 데이터를 생성하는 방법은 다음과 같다.

1. 합성 데이터의 개수를 정한다. (보통 1:1의 비율을 맞춘다.) 예를 들어 소수 데이터가 10개, 다수의 데이터가 100개라면 합성할 데이터의 개수는 90개

2. $ k = int(S_{syn} / S_{min})$ 을 결정 이 때 $S_{min}$ 은 소수 클래스 데이터의 개수를 의미

3. 각 $x_i \in S_{min}$에 대해 같은 클래스의 평균 거리와 다른 클래스간의 평균거리를 계산한다.

4. $ \alpha = \overline{d_{intra}} / \overline{d_{extra}} $ 를 계산한다. 이 때 이 값이 0.5보다 작으면 smote와 같이 합성 데이터 생성 이 값이 0.5보다 크면 합성 데이터를 생성하지 않는다.

![제목 없음](/assets/images/oversampling/제목 없음.png)



```python
import smote_variants as sv


oversampler= sv.NDO_sampling(random_state = 101, n_jobs = -1)


start = time.time()

X_samp, y_samp = oversampler.sample(np.array(X), np.array(y))

end = time.time()

print(f"{end - start:.5f} sec")
```

    2021-12-04 18:06:33,085:INFO:NDO_sampling: Running sampling via ('NDO_sampling', "{'proportion': 1.0, 'n_neighbors': 5, 'T': 0.5, 'n_jobs': -1, 'random_state': 101}")
    I1204 18:06:33.085295 20816 _smote_variants.py:17104] NDO_sampling: Running sampling via ('NDO_sampling', "{'proportion': 1.0, 'n_neighbors': 5, 'T': 0.5, 'n_jobs': -1, 'random_state': 101}")


    18.65919 sec



```python
np.unique(y_samp, return_counts = True) # 동일한 개수로 oversampling 되었음
```




    (array([0, 1], dtype=int64), array([221004, 221004], dtype=int64))



위와 같이 oversampling된 데이터에 대해서 train을 실시하고 test-set으로 모델의 최종적인 성능을 평가하면 된다.

이 때 주의할 점은 test-set의 경우 oversampling이나 undersampling 기법을 적용하면 안된다는 점이다.

추가적으로 앞서 소개한 3가지 기법 이외에도 boderline smote, smote-bagging등과 같이 여러 smote의 응용 방법과

rusboost와 같이 undersampling을 적용한 기법도 존재하며 oversampling과 undersampling을 결합한 hybrid 방법이 존재

Loan Data를 이용해 train/test 7:3의 비율로 분할한 후 5-fold-cross validation을 적용하여 데이터 분석을 진행

각각 LDA/QDA/KNN/Random-forest/XGBoost/Light-GBM/Decision-tree 모델을 적용하였다.



## 평가 지표 선정


```python
def get_clf_eval(y_test, y_pred):
    confmat=pd.DataFrame(confusion_matrix(y_test, y_pred),
                    index=['True[0]', 'True[1]'],
                    columns=['Predict[0]', 'Predict[1]'])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    g_means = geometric_mean_score(y_test, y_pred)
    print(confmat)
    print("\n정확도 : {:.3f} \n정밀도 : {:.3f} \n재현율 : {:.3f} \nf1-score : {:.3f} \nAUC : {:.3f} \n기하평균 : {:.3f} \n".format(accuracy,
                                        precision, recall, f1, AUC, g_means))
    

```

## 모델 적합 (Non-oversampling)


```python
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
import time

models_X = []
models_X.append(('LDA', LinearDiscriminantAnalysis()))  # LDA 모델
models_X.append(('QDA', QuadraticDiscriminantAnalysis()))  # QDA 모델
models_X.append(('KNN', KNeighborsClassifier())) # KNN 모델
models_X.append(('DT', DecisionTreeClassifier()))  # 의사결정나무 모델
models_X.append(('RF', RandomForestClassifier()))  # 랜덤포레스트 모델
models_X.append(('XGB', XGBClassifier()))  # XGB 모델
models_X.append(('Light_GBM', LGBMClassifier())) # Light_GBM 모델

for name, model in models_X:
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time() - start
    msg = "%s - train_score : %.3f, test score : %.3f, time : %.5f 초" % (name, model.score(X_train, y_train), model.score(X_test, y_test), end)
    print(msg)
```

## SMOTE 적용


```python
from sklearn.model_selection import train_test_split

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size = 0.3, random_state = 101)

models_s = []
models_s.append(('LDA', LinearDiscriminantAnalysis()))  # LDA 모델
models_s.append(('QDA', QuadraticDiscriminantAnalysis()))  # QDA 모델
models_s.append(('KNN', KNeighborsClassifier())) # KNN 모델
models_s.append(('DT', DecisionTreeClassifier()))  # 의사결정나무 모델
models_s.append(('RF', RandomForestClassifier()))  # 랜덤포레스트 모델
models_s.append(('XGB', XGBClassifier()))  # XGB 모델
models_s.append(('Light_GBM', LGBMClassifier(boost_from_average=False))) # Light_GBM 모델

for name, model in models_s:
    start = time.time()
    model.fit(X_train_s, y_train_s)
    end = time.time() - start
    msg = "%s - train_score : %.3f, test score : %.3f, time : %.5f 초" % (name, model.score(X_train_s, y_train_s), model.score(X_test_s, y_test_s), end)
    print(msg)
    
# 모델 갯수
a = list(range(0,len(models_s)))

for i in a:
    print("----------SMOTE + %s 모델 적용----------" % (models_s[i][0]))
    get_clf_eval(y_test_s, models_s[i][1].predict(X_test_s))
```

![다운로드 (6)](/assets/images/oversampling/다운로드 (6).png)

이 외에도 ADASYN,Distribution-smote 방법도 동일하게 적용하면 된다.

앞서 설명한 바와 같이 accuracy의 경우 oversampling을 적용한 결과 정확도의 경우 기존에 비해 떨아지지만 

AUC의 경우 성능이 상승하였음 다음은 grid-search를 통한 간단한 튜닝 작업을 진행

## Tunning using Pipeline

다음은 ADASYN과 KNN을 결합하여 모델의 hyperparameter를 조정하기로 하였음


```python
start = time.time()


pipeline = Pipeline(steps= [("ADASYN", ADASYN(random_state = 101)),
                            ("KNN", KNeighborsClassifier())
                            ])

param_grid = {
    "ADASYN__sampling_strategy": [0.5,0.75,1],
    "KNN__n_neighbors": list(range(10,101,10))
              }



gs_pipeline_1 = GridSearchCV(pipeline, param_grid=param_grid, verbose=2, scoring=make_scorer(roc_auc_score), cv = 5)
gs_pipeline_1.fit(X_train, y_train)

end = time.time()

print(f"{end - start:.5f} sec")

# Store the best model
ada_best_model = gs_pipeline_1.best_estimator_

y_validation_preds = ada_best_model.predict(X_test)
roc_auc_score(y_test, y_validation_preds)

scores_df = pd.DataFrame(gs_pipeline_1.cv_results_)
```

추가적으로 다른 모델들도 튜닝 작업을 진행하였음  

결과는 다음과 같다.

![다운로드 (7)](/assets/images/oversampling/다운로드 (7).png)

AUC의 성능의 경우 Random-forest의 모델이 가장 좋았지만 computation-time이 굉장히 오래 걸렸음을 확인할 수 있음

그에 비해 Light-gbm의 경우 Random-forest에 비해서는 성능이 약간 떨어지지만, computation-time이 비약적으로 줄어들었음을 확인할 수 있음

## Grid-search Results of KNN

![다운로드 (8)](/assets/images/oversampling/다운로드 (8).png)
낮은 성능과 높은 성능을 보이는 일정한 영역이 존재함을 확인

=> 추가적으로 K의 개수를 10 이하로, oversampling-proportion을 1에 근접한 값으로 탐색한다면 안정적인 성능을 보일 것으로 기대

# Conclusion

1. 불균형 데이터의 경우 정확도보다 AUC/f1-socre 등의 지표를 선정하는 것이 바람직함

2. 불균형 데이터를 처리하기 위한 방법으로는 undersampling/oversampling/loss-function apporoach 등 3가지 존재

3. oversampling 기법을 적용한 결과 AUC 지표가 상승함을 확인할 수 있었음 즉 oversampling 기법이 불균형 데이터를 처리하는 타당함을 확인할 수 있었음
