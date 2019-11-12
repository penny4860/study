
# ECMA6

* 변수선언
    * var
    * let / const
        * 블록 scope 적용 : 변수의 scope가 블록(```{}```)으로 한정된다.

* 디폴트 인자
    * ```function join(name="test"){...} ```

* arrow function
    * 익명함수를 arrow 기호로 표현하는 방식

* 클래스 정의 / 상속

```javascript
// class 정의 방법
class Person
{
    constructor (id, name){
        this.id = id;
        this.name = name
    }
    printName(){
        //...
    }
}

// 상속하는 방법
class Participant extends Person {
    constructor (id, name, channel) {
        super(id, name)
        this.channel = channel;
    }
}
```

* promise
    * 비동기처리를 쉽게해주는 객체

* 모듈
    * 모듈형태로 파일을 분리하는 것이 가능
    * export : 외부모듈에서 사용가능하도록
    * import : 외부모듈을 불러와서 사용


