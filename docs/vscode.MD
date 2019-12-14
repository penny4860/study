# visual studio code

## 1. 확장도구

* Git Graph


## 2. 단축키

[공식문서](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf)

* 검색
    * 전체폴더에서 검색
        * ctrl + shift + f
    * 파일이름을 검색
        * ctrl + shift + e
    * Replace
        * ctrl + h
* 라인단위 처리
    * 코드라인을 Move
        * Alt + up/down
    * 코드라인을 Copy
        * Alt + Shift + up/down
* Insert Cursor : 여러곳에 커서만들기
    * alt + click
* 해당단어를 파일에서 전체선택
    * 단어를 블록선택후 ctrl + D


## 3. python 개발환경

```project-root/.vscode/settings.json```, ```project-root/.env``` 두가지 파일을 아래와 같이 설정해준다.


### 3.1. project-root/.vscode/settings.json

```json
{
    "python.pythonPath": "C:\\Users\\jeongjoonsup\\AppData\\Local\\Continuum\\anaconda3\\envs\\mlapi\\python.exe",
    "python.linting.enabled": true,
    "python.formatting.provider": "autopep8",
    "python.linting.pycodestyleEnabled": true,
    "python.linting.pycodestyleArgs": [
        "--ignore=E501"
    ],
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--enable=F401",
        "--ignore=E303,E501"
    ],
    "editor.minimap.enabled": false,
    "editor.formatOnSave": true,
    "editor.formatOnPaste": false,
    "editor.formatOnType": false,
    "[html]": {
        "editor.formatOnSave": false
    },
    "python.autoComplete.showAdvancedMembers": true,
    "python.autoComplete.addBrackets": true,
    "editor.cursorBlinking": "expand",
    "editor.renderWhitespace": "boundary",
    "editor.formatOnSaveTimeout": 1500,
    "python.formatting.blackArgs": [
        "--skip-string-normalization",
        "--line-length",
        "99"
    ],
    "python.jediEnabled": false,
}
```

### 3.2. project-root/.env

```shell
PYTHONPATH = ./com/tmoncorp/data/ml_api;${PYTHONPATH}
```

* ```project-root/com/tmoncorp/data/ml_api```를 python-path로 지정하고 싶으면 위와같이 설정
* .env 파일을 project-root에 만든다.


