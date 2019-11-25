
## 로직

1. model 로드
    * ```net = await mobilenet.load();```

2. prediction할 이미지를 가져온다.
    * ```const imgEl = document.getElementById('img');```

3. prediction 수행
    * ```const result = await net.classify(imgEl);```

## 질문

* 다른 입력이미지는 왜 제대로 안되지?


