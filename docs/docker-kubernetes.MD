# 도커/쿠버네티스를 활용한 컨테이너 개발 실전 입문

## 1장 : 도커의 기초

* Docker의 장점
    * container에 실행환경을 포함해서 배포가능
    * 소스코드 + 실행환경

* Docker 파일 ====> Docker 이미지 ====> Docker container
    * 빌드
        * Docker 파일 -> Docker 이미지
    * 실행
        * Docker 이미지 -> Docker container

* Docker의 확장
    * Docker : 1개 앱
    * Docker compose : 여러개의 application
    * Docker container orchestrator : 여러개의 서버

## 2장 : 도커 Container 배포

### 2.1. Container 실행방법

1) Dockerfile 작성
    * Dockerfile instruction으로 작성
    * FROM / RUN / COPY / CMD / ENTRYPOINT
2) Docker image 빌드
    * ```docker image build -t example/echo:latest .```
        * ```example/echo``` : 도커이미지 이름
            * example : namespace
        * ```:latest``` : 태그
        * ```.``` : 도커파일경로
3) Container 실행
    * ```docker container run -d -p 9000:8080 example/echo:latest```
        * -d : 백그라운드 실행옵션
        * -p : port 포워딩
    * 기타 옵션
        * -it : docker container를 ssh처럼사용
        * -rm : container 종료시 container를 파기
        * -v : volume 지정 (host:docker)

```project-root> sudo docker container run -d -p 50051:50051 -v /home/TMON/jeongjoonsup/index_samples:/app/index_samples faiss/visualsearch:latest```

### 2.2. docker-compose를 사용한 container 실행방법

```yml
version: "3"
services:
    echo:
        build: .
        ports:
            - 9000:8080
```

* dockerfile과 docker-compose.yml을 같은 디렉토리에 넣고 다음 command를 실행
* docker-compose up -d --build

### 2.3. docker container에 ssh 접속하기

```docker container exec -it $(container 이름) sh```
