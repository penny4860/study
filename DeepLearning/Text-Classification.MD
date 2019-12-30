# Text Classification

* 참고자료
    * 텐서플로와 머신러닝으로 시작하는 자연어 처리
    * [이미지와 Text정보들을 이용한 쇼핑 카테고리 분류 AI](https://deview.kr/2019/schedule/290)
    * [convolutional neural network for sentence classification](https://www.aclweb.org/anthology/D14-1181.pdf)

## 1. 데이터 분석

* Histogram
    * x축 : 레이블
    * y축 : 데이터 숫자 / 문장의 길이
* Box Plot
    * 레이블 별 데이터 숫자
* 워드 클라우드
    * 가장 많이 사용된 단어를 시각화

## 2. 데이터 전처리

* 형분기로 tokenizing
    * 어간 추출 옵션 : ```stem=True```
    * sentence -> word list
* 불용어 (stopword) 삭제
* vectorize
    * index 변환 : tf.Tokenizer 모듈
        * word list를 index list로 변환
    * 패딩 : tf.pad_sequence 모듈
        * 모든 index list를 fixed length로 맞춘다.

## 3. RNN text 분류기 

* index list
* Embedding Layer
* LSTM -> LSTM
    * 이전 step의 출력과 
    * 현재 step의 Embedding Layer 출력을 입력받는다.
* fc + softmax


## 4. CNN text 분류기

* index list : ```[sequence,]```
    * ```[30]```
* embed list : ```[sequence, embed-size, 1]```
    * ```[30, 64, 1]```
* cnn :
    * input tensor : ```[sequence, embed-size, 1]```
    * kernel : 
        * ```[kh-size, kw-size, input-depth, output-depth]```
        * ```[1/2/3/4, embed-size, 1, n-filters]```
            * kh-size를 여러개로 filtering 하고 concat
            * 네이버 카테고리 매칭의 경우 1/2/3/4 4개를 사용했다고 함.
    * output tensor : ```[sequence, 1, n-filters*4]```
* max pooling
    * ```[sequence, 1, n-filters*4]``` ==> ```[1, 1, n-filters*4]```
* flatten
    * ```[1, 1, n-filters*4]``` ==> ```[n-filters*4,]```
* fc + softmax
    * ```[n-filters*4,]``` ==> ```[n-categories,]```
