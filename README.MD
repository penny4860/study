# Study Note


## 질문

* EfficientNet
	* FLOPS 는 정확히 어떤 연산을 count하는 걸까?
	* MBConv?
	* Squeeze-and-Excitation?
	* stochastic depth?
	* batchnorm 에서 momentum의 의미?
	* 읽어볼 논문
		* mnasnet
		* swish activation : Searching for activation functions. 
			* relu에 비해 얼마나 느리고 효율적인지 확인
		* fixed autoaugment policy : Autoaugment: Learning augmentation policies from data.
		* Do better imagenet models transfer better?
			* Transfer Learning의 학습 setting 확인할 것.
		* Proxylessnet
* Boosting Standard Classification Architectures Through a Ranking Regularizer 
    * center loss를 사용한 논문에서는 softmax는 사용하지 않았나?
        * Y. Wen, K. Zhang, Z. Li, and Y. Qiao. A discriminative feature learning approach for deep face recognition. In ECCV. Springer, 2016.
    * triplet center loss를 사용한 논문에서는 softmax는 사용하지 않았나?
        * X. He, Y. Zhou, Z. Zhou, S. Bai, and X. Bai. Tripletcenter loss for multi-view 3d object retrieval. arXiv preprint arXiv:1803.06189, 2018

## Todo

* Tensorflow-2.0 코드 리뷰
    * dataset API
    * model 구현 방식
    * training loop 구현방식 : ```model.fit()```
        * callback
        * tf summary
        * custom loss 사용방법
    * mobilenet 으로 cifar-100 학습해보기

* CNN 모델 연구
    * efficientnet 논문리뷰
    * mobilenet
    * resnet

* Text 분류기 모델

* "Boosting Standard Classification Architectures Through a Ranking Regularizer" 에 나온 FGVR 데이터 셋으로 efficientnet 학습해보기.
    * efficientnet
        * gpu memory
        * inference time
        * training time

* 지속적인 학습 사례 리뷰
    * http://dmqm.korea.ac.kr/activity/seminar/266

* tfjs-efficientnet 구현
    * keras 모델을 tf-js로 변환 : https://www.tensorflow.org/js/tutorials/conversion/import_keras
    * node.js 에서 tfjs 모델을 테스트
        * local 모델을 로드하는 예제
            * https://github.com/tensorflow/tfjs-examples/blob/master/mnist-acgan/index.js#L168
    * html + js 로 UI 구현
    * deploy 실습
        * https://www.youtube.com/watch?v=LnGgndT308Q&t=182s
        * Netlify 사용

* style transfer
    * https://deview.kr/2019/schedule/309

## Image Embedding 관련 논문

* SoftTriple Loss: Deep Metric Learning Without Triplet Sampling
* A Theoretically Sound Upper Bound on the Triplet Loss for Improving the Efficiency of Deep Distance Metric Learning
* Triplet-Center Loss for Multi-View 3D Object Retrieval




