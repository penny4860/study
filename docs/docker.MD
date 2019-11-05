# docker 란 무엇인가 : 기본개념

[초보를 위한 도커 안내서 - 도커란 무엇인가?](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html)

## 질문

* Docker를 쓰면 좋은 점
	* program(웹 서버등)의 실행환경을 독립적으로 관리할 수 있다.
	* anaconda env의 상위호환
* (centos) : yum remove / erase 의 차이
	* yum version이 업데이트되면서 패키지를 지우는 명령이 바뀐듯.
	* ```yum --help```로 확인해보자.
* (centos) : ```/etc/yum.repos.d/```가 하는 일
	* yum package를 설치하는 원격저장소 정보가 명시되어있음.
* DevOps?
* 쿠버네틱스?


## Note

* Docker
	* Linux container 기반의 가상화 도구

* Image
	* program 실행에 필요한 모든것을 묶어 파일로 만든 것
	* 라이브러리, 소스코드등이 포함되어있음.

* Container
	* Docker Image를 실행한 상태

* Docker File
	* Docker Image를 빌드하기 위한 build script

* Layer
	* Git의 commit


