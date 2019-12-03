
# tfjs-mobilenet

* 소스코드 : https://github.com/penny4860/tfjs-mobilenet

## Note

### html 문서에 텍스트를 변경

1. (html) js로 제어할 element를 정의
    * ```<div id="status"></div>```
2. (js) status 함수정의
    * ```const demoStatusElement = document.getElementById('status');```
    * ```const status = msg => demoStatusElement.innerText = msg;```
3. (js) 함수사용
    * ```status("Loading....")```

### 태그의 내용을 숨겼다가 특정 시점에 렌더링

1. (html) img 태그에서 style 속성으로 display off
    * ```<img style="display: none" id="cat" src="cat.jpg" width=224 height=224 />```
2. (js) image 렌더링을 원하는 시점에 display 속성 변경
    * ```const catElement = document.getElementById('cat');```
    * ```catElement.style.display = '';```

### tfjs 의 inference

1. load model
    * ```mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);```
2. image element 를 준비
    * ```const imgElement = document.getElementById('cat');```
3. prediction 과정을 익명함수로 정의
4. 익명함수를 실행
    * ```const values = await logits.data();```

### image file을 업로드

1. (html) input 태그를 준비
    * ```Upload an image: <input type="file" id="files" name="files[]" multiple />```
2. input element를 가져와서 리스너를 등록
    * ```const FILES_ELEMENT = document.getElementById('files');```
    * ```FILES_ELEMENT.addEventListener('change', e => fileUploadListener(e));```
3. 이벤트 리스너를 구현
    * ```let reader = new FileReader();```
    * ```reader.onload = e => imgLoader(e);```
4. FileReader 리스너를 구현

