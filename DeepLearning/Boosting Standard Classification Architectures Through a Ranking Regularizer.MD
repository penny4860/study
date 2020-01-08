# Boosting Standard Classification Architectures Through a Ranking Regularizer

## 정리

### 1. 요약

### 2. 질문

* center loss를 사용한 논문 (38) 에서는 softmax는 사용하지 않았나?
* triplet center loss를 사용한 논문 (8) 에서는 softmax는 사용하지 않았나?
* batch construction 방법이 무슨 말인지 모르겠음.
    * K=4? : ```batch size=32```중에서 4종류의 class만 sampling 한다는 건가?
    * 9번 논문에서의 구성방법 알아보자.

## 내용

### 1. Intro

* 일반적인 image classification task는 softmax loss 를 사용한다.
* sotfmax loss의 한계 : embedding 에 한계
    * embedding space상에서 샘플을 잘 분류하는 일만한다.
    * embedding space상에서의 margin을 고려하지 않는다.
* 좋은 embedding?
    * 같은 class 끼리는 거리가 가까워야 함.
    * 다른 class 끼리는 거리가 멀어야 함.
* 논문의 contribution
    1) 2-head 구조를 제안
        * softmax loss 기반, triplet loss를 regularization loss로 사용
        * 더 좋은 feature embedding을 얻음으로써, 분류기 성능향상
    2) triplet loss 에서의 Batch size 평가
        * 작은 batch size로도 좋은 성능을 만들어냄.
    3) standard classification 보다 더 좋은 retriaval 성능을 만들어냄.

### 2. Related Work

* softmax 기반의 classifier는 ```intra-class compatness, iner-class maximization```을 고려하지 않는다.
* Embedding regularization은 이러한 한계를 해결하기 위한 방법이다.

#### 2.1. Center loss

* intra class variation을 줄이는 regularization loss
* batch sample과 해당 sample의 class center의 l2-distance를 minimize

#### 2.2. Magnet loss

* center loss의 경우 같은 class는 uni-mode임을 가정
* intra class에서 multi-mode를 허용

#### 2.3. Triplet Center Loss

* class compactness + inter-class margin을 함께 고려
* Triplet loss와의 차이점은 class별 center를 구하느냐의 차이.

### 3. The Triplet Loss Regularizer

#### 3.1. Triplet Loss

* 기존 연구에서의 Triplet loss 사용사례
    * feature embedding Tool
    * object 끼리의 similarity 측정
    * clustering mertic
* 논문에서의 Triplet Loss
    * softmax와 함께 regularization loss로 사용
    * ```loss_triplet = D(a, p) - D(a, n) + margin```
        * a : anchor sample
        * p : a와 같은 class의 sample
        * n : a와 다른 class의 sample
        * margin = 2
    * ```L = L_soft + lambda*L_tri ```
        * lambda = 1

* batch sampling 방법
    * negative sample의 부류
        * ```anchor -- hard negative -- positive -- semi-hard negative -- margin -- easy negative```
        * anchor : 기준점
        * negative
            1) hard : positive보다 가까움.
            2) semi-hard : positive 보다 멀지만 margin 보다 가깝다.
            3) easy : margin보다 멀다. 분류하기 쉬움
    * 논문에서는 negative sample을 ```semi-hard -> easy -> hard```를 우선순위로 sampling
        * batch size가 작기때문(b=32)에 semi-hard 또는 hard만을 선택하지 않는다.

#### 3.2. Two-head architecture

* conv feature map ```(7,7,2048)```
    * 1st head
        * (flatten) vector ```(100352)```
        * (fc) 최종 embedding vector ```(256)```
    * 2nd head
        * (GAP + flatten) vector ```(2048)```
        * (fc) 최종 pred vector ```(n_class)```

### 4. Experiments

#### 4.1. Evaluation on FGVR

* Datasets
    * Aircraft
    * NABirds
    * Flower-102
    * Stanford Cars 
    * Stanford Dogs
* Baselines
    1) softmax
    2) 2-head leveraging center loss
        * 논문에서 제안한 2-head 구조
        * regularization loss 를 center loss로 사용
* Proposed Method : 2-head + Triplet loss
    * batch size : 32
    * embedding dimension : 256
    * batch sampling
        * hard sampling
        * semi-hard sampling
            * margin = 2
    * batch construction procedure
        * ?



