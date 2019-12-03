# Todo

## tfjs

* deploy 실습
    * https://www.youtube.com/watch?v=LnGgndT308Q&t=182s
    * Netlify 사용
* custom model 로 classifier를 구현
    * keras 모델을 tf-js로 변환 : https://www.tensorflow.org/js/tutorials/conversion/import_keras
    * local 모델을 로드하는 예제
        * https://github.com/tensorflow/tfjs-examples/blob/master/mnist-acgan/index.js#L168
    * 변환된 모델 코드를 google cloud storage에 저장
    * gcs에서 모델을 읽어서 html문서에 rendering
