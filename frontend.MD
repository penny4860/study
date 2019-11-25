# Frontend Study

* 개발환경 구축 : visual studio code
    * 확장도구설치
        * Git Graph
        * open-in-browser (coderfee)
        * code runner
            * ```ctrl + alt + n``` : 자바스크립트만 콘솔에서 실행
    * 단축키
        * [문서](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf)
        * ```ctrl + alt + o``` : 브라우져에서 실행

## Todo

* http://www.jquerysample.com/
* https://codepen.io/rajdgreat007/pen/ZpZWbw
* JavaScript Promises
* Front-End Frameworks


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

