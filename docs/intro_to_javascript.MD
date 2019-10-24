# Intro to Javascript

https://classroom.udacity.com/courses/ud803

## Lesson1 : what is javascript

* Javascript console
	* 우클릭 -> inspect -> console
	* ctrl + shift + j
	* 자바스크립트 코드를 브라우져에서 실행

* ```console.log()```
	* used to display content to the JavaScript console.


## Lesson2 : Data types and Varriables

* Types
	1) number
	2) string
	3) boolean
	4) null
		* nothing
	5) undefined
		* absence of value

## Lesson3 : 

### 3.1. if / else

```
var a = 1;
var b = 2;

if (a > b) {
  console.log("a is greater than b");
} else {
  console.log("a is less than or equal to b");
}
```

### 3.2. logical operators

* and : ```&&```
* or : ```||```
* not : ```!```

### 3.3. switch case

```
var option = 3;
switch (option) {
  case 3:
    console.log("You selected option 3.");
    break;
  case 6:
    console.log("You selected option 6.");
    break; // technically, not needed
}
```

## Lesson4 : Loop

* while loop

```
var start = 0; // when to start
while (start < 10) { // when to stop
  console.log(start);
  start = start + 2; // how to get to the next item
}
```

* for loop

```
for (var i = 0; i < 6; i = i + 1) {
  console.log("Printing out i = " + i);
}
```

## Lesson5 : Functions

### 5.1. Function Declaration

```
function catSays(max) {
	...
}
catSays();
```

### 5.2. Function Expression

```
// an anonymous function, a function with no name
var catSays = function(max) {
	...
}
catSays();
```

## Lesson6 : Arrays

```
// creates a `donuts` array with three strings
var donuts = ["glazed", "powdered", "jelly"];
console.log(donuts[0]); 
```

## Lesson7 : Objects

* dictionay 비슷한 자료구조

```
var umbrella = {
    color : "pink",
    isOpen : false
};
```

