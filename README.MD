# Study Note


## 질문

* yolo3
	* box decoding
	    * box decoding 할때 width, height를 구하는 과정이 unstable하지 않을까?
	        * exp(tw) 가 들어가는 부분
	    * 다른 논문에서의 decoding 방식 비교
	    * anchor 사용방식 비교
	* Coco metric이 기존의 Voc metric과 어떻게 다르지?
	* retinanet에서 objectness threshold를 어떻게 쓰지?
	    * yolo3에서는 objectness score를 예측하고 이 score로 thresholding
	    * retinanet에서는 최대 score class에 thresholding 하는가?
* EfficientDet
    * cosine decay rule?
    * resnet50과 efficientnet-B3의 inference time
        * gpu inference time : resnet50이 더 빠름.
        * cpu inference time : B3가 더 빠름.
        * 왜 차이가 날까?
* EfficientNet
	* FLOPS 는 정확히 어떤 연산을 count하는 걸까?
	* MBConv?
	* Squeeze-and-Excitation?
	* stochastic depth?
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
* ```Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour``` 에서 warmup training으로 object detection / segmentaion task를 fine tuning 한 방법


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




