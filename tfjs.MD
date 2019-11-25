# Tensorflow JS

## Tensorflow JS 개발환경 구축

1) 설정파일 복사
    * .babelrc
    * package.json
2) ```project-root> yarn```
    * node_modules 디렉토리 생성
    * ```yarn.lock``` 생성
3) 실행 : ```yarn watch```
    * dist 디렉토리 생성
    * 로컬 서버에서 웹앱이 실행

## Todo

* mobilenet js 코드 분석
    * .babelrc / package.json / npm 등으로 frontend 코드 개발환경 구성방식 리뷰
    * https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet
* deploy 실습
    * https://www.youtube.com/watch?v=LnGgndT308Q&t=182s
    * Netlify 사용
* custom model 로 classifier를 구현
    * keras 모델을 tf-js로 변환 : https://www.tensorflow.org/js/tutorials/conversion/import_keras
    * 변환된 모델 코드를 google cloud storage에 저장
    * gcs에서 모델을 읽어서 html문서에 rendering

