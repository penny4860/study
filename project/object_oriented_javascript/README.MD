
## 질문

* const / var 의 차이는?
    * https://gist.github.com/LeoHeo/7c2a2a6dbcf80becaaa1e61e90091e5d

## Lesson 1 : Objects in Depth

```javascript
const cat = {
  age: 2,
  name: 'Bailey',
  meow: function () {
    console.log('Meow!');
  },
  greet: function (name) {
    console.log(`Hello ${name}`);
  }
};
```

* window object
    * browser에서 제공하는 최상위 객체
    * global로 선언한 변수/함수는 window객체의 property가 된다.

* Extracting key/value
    * object란 key/value의 조합
    * ```Object.keys()``` 객체의 키를 array로 리턴
    * ```Object.values()``` 객체의 value를 array로 리턴

## Lesson 2 : Functions at Runtime

* First-clss Functions
    * JS에서는 function도 하나의 element
        1) 변수에 저장가능
        2) 함수에서 리턴 가능
        3) 함수의 argument로 사용가능 : callback 함수


```javascript
// forEach() 예제코드
[1, 5, 2, 4, 6, 3].forEach(function (n) {
  if (n % 2 !== 0) {
    console.log(n);
  }
});
```

* 자바스크립트 함수에서 access 가능한 객체
    * argument
    * local variable
    * parent function scope
    * global variable

* closure
    1) 내부함수가 외부함수의 지역변수에 접근할 수 있음.
    2) (외부함수의 지역변수를 사용하는 )내부함수가 살아있으면 외부함수도 살아있음.

```javascript
// closure를 사용한 클래스 구현
function factory_movie(title){
    return {
        get_title : function (){
            return title;
        },
        set_title : function(_title){
            title = _title
        }
    }
}
ghost = factory_movie('Ghost in the shell');
matrix = factory_movie('Matrix');
```


