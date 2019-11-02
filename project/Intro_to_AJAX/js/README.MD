# Intro to AJAX 

* https://classroom.udacity.com/courses/ud110

## 1. Google Map API

* html 문서
    * 입력 폼을 만들어 사용자의 텍스트입력을 받는다.
        * 입력 폼 : ```<form id="form-container" class="form-container">```
        * 입력 태그 : ```<input type="text" id="street" value="">```
* 자바스크립트 
    * 이벤트 처리함수를 구현
        * ```$('#form-container').submit(loadData); //입력 폼이 submit 되면 loadData()를 실행```
    * jQuery selector로 input tag에 입력한 주소을 가져온다.
        * ```var streetStr = $('#street').val();```
    * 구글 지도 API에 query url을 만들어서 이미지를 가져온다.
    * body tag 에 image를 추가
        * ```$body.append('<img class="bgimg" src="' + streetviewUrl + '">');```

## 2. getJSON()

* ```$.getJSON(url, function(data){...});```
    * function(data) 에 url에서 async하게 가져온 데이터가 저장되어있다.

## 3. ajax()

```javascript
$.ajax({
    url: wikiUrl,
    dataType: "jsonp",
    jsonp: "callback",
    success: function( response ) {
        var articleList = response[1];

        for (var i = 0; i < articleList.length; i++) {
            articleStr = articleList[i];
            var url = 'http://en.wikipedia.org/wiki/' + articleStr;
            $wikiElem.append('<li><a href="' + url + '">' + articleStr + '</a></li>');
        };

        clearTimeout(wikiRequestTimeout);
    }
});
```

