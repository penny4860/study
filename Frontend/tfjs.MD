# Frontend Note

## Note

* CSS 적용방법
    * head태그에서 적용 : ```<link rel="stylesheet" href="style.css">```

* 자바스크립트 적용방법
    * body태그에서 적용

* jQuery 적용방법
    * head태그에서 적용 : ```<link rel="stylesheet" href="style.css">```

* web API 호출방법
    * html문서 : 버튼클릭에 대한 이벤트 함수등록
    * 자바스크립트 : 이벤트함수에서 REST API 를 호출하는 코드 구현
        * ```$.getJSON(url, function(data){...});```
            * function(data) 에 url에서 async하게 가져온 데이터가 저장되어있다.
        * ```$.ajax()```

* 장고프로젝트에서 자바스크립트를 사용
    * html 문서 : ```{% load staticfiles %}``` 를 문서 상단에 추가
    * javascript 파일경로 : ```"/static/main.js```

### js 파일을 html과 브라우저 독립적으로 실행하는 방법

* 참고자료 : https://www.daleseo.com/js-babel-node/
* Method
    * babel-cli를 설치
        * ```yarn add -D babel-cli```
        * es6문법을 사용하기 위함.
    * node를 이용해서 자바스크립트를 실행
        * ```npx babel --presets env .\nodetest.js | node```

### 비동기처리, async awati, promise 코드 이해하기

* https://joshua1988.github.io/web-development/javascript/promise-for-beginners/

### package version을 지정하는 방법

```yarn add ws@3.3.1```

