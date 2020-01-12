# In defense of the triplet loss for person re-identification

## 정리

### 1. 요약

### 2. 질문

## 내용

### 1. Introduction

* contribution
    1) Classical Triplet loss의 variation을 소개
    2) 기존 cnn 구조에 triplet loss를 사용한 end-to-end 모델을 제안

### 2. Learning Metric Embeddings, the Triplet Loss, and the Importance of Mining

* Learning Metric Embeddings
    * Semantically similar sample(이미지)이 embedding space에서 가까운 distance를 갖도록 학습한다.

* Triplet Loss
    * ```loss_triplet = D(a, p) - D(a, n) + margin```
    * 같은 class의 data sample이 임베딩 space에서 하나의 cluster를 형성

* the Importance of Mining
    * Dataset의 크기가 커질수록 possible triplet(anchor, positive, negative)의 숫자가 많아진다.
    * 따라서 batch sample의 구성을 잘 정의해야함.

* **논문에서 사용한 batch sampling 전략**
    1) P class를 random으로 선택
    2) 각 class 별로 K개의 image를 선택
    3) ```P * K``` anchor sample 에 대해서 hardest positive / hardest negative로 triplet을 구성
        * hardest positive : anchor와 같은 클래스중 가장 거리가 먼것.
        * hardest negative : anchor와 다른 클래스중 가장 거리가 가까운 것.

* Distance Measure
    * squared euclidean distance가 아닌 euclidean distance를 distance metric으로 사용
    * 더 안정적으로 학습되었다고 함.

* Soft margin
    * Hinge Function 
        * Triplet loss 연산시 주로 사용하는 함수
        * ```hinge = max(margin + ..., 0) ```
    * Softplus (soft margin)
        * hard cut off를 갖지않고 smooth한 decay curve를 보임.
    * 논문에서는 Hinge / Softplus 두가지 함수를 모두 사용해서 실험했다고 함.

### 3. Experiments

(리뷰중)
