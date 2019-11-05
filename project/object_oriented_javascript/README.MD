
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

* immediately-invoked function expressions : 함수선언과 동시에 함수를 호출

```html
<body>
    <button id='button'>Click me!</button>
    <script>
        const button = document.getElementById('button');
        button.addEventListener('click', (function() {
            // count를 private scope로 보호
            let count = 0;
            return function() {
                count += 1;
                if (count === 2) {
                    alert('This alert appears every other press!');
                    count = 0;
                }
            };
            })());
    </script>            
</body>
```

## Lecture 3 : Classes and Objects

### Constructor Function 

object를 만들어주는 function을 constructor function이라고 한다.

```javascript
function SoftwareDeveloper(name) {
    this.favoriteLanguage = 'JavaScript';
    this.name = name;
}
let instructor = new SoftwareDeveloper('Andrew');
```


