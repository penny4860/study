# Study Note

## Todo

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


## 질문

* Boosting Standard Classification Architectures Through a Ranking Regularizer 
    * batch construction 방법이 무슨 말인지 모르겠음.
        * K=4? : ```batch size=32```중에서 4종류의 class만 sampling 한다는 건가?
        * 아래 논문에서의 구성방법 알아보자.
            * A. Hermans, L. Beyer, and B. Leibe. In defense of the triplet loss for person re-identification. arXiv preprint arXiv:1703.07737, 2017.
    * center loss를 사용한 논문에서는 softmax는 사용하지 않았나?
        * Y. Wen, K. Zhang, Z. Li, and Y. Qiao. A discriminative feature learning approach for deep face recognition. In ECCV. Springer, 2016.
    * triplet center loss를 사용한 논문에서는 softmax는 사용하지 않았나?
        * X. He, Y. Zhou, Z. Zhou, S. Bai, and X. Bai. Tripletcenter loss for multi-view 3d object retrieval. arXiv preprint arXiv:1803.06189, 2018

