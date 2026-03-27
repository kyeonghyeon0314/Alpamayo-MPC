# 컨테이너 환경 셋업 노트

컨테이너 최초 실행 이후 Jupyter 개발 환경을 구성하는 과정에서 발생한 문제와 해결 방법을 정리한 문서.

## 문제 1: jupyter notebook 실행 실패 (distutils)

### 원인
Ubuntu apt로 설치된 `notebook==6.4.8`이 `/usr/bin/jupyter-notebook`으로 등록되어 있는데,
Python 3.12에서 `distutils` 모듈이 제거되어 호환되지 않음.

### 해결
`jupyter lab` 명령 사용 (apt 버전 대신 uv로 설치된 JupyterLab 사용):
```bash
uv pip install jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

---

## 문제 2: VS Code Jupyter 커널 연결 방법

컨테이너에서 Jupyter 서버를 실행하고 VS Code에서 원격 커널로 연결.

```bash
# 컨테이너에서 서버 시작
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

VS Code에서:
- 노트북 우측 상단 커널 선택 → **Select Another Kernel** → **Existing Jupyter Server**
- 터미널에 출력된 URL(`http://127.0.0.1:8888/lab?token=...`) 붙여넣기

---

## 정상 실행 순서 (컨테이너 재시작 후)

```bash
# 1. 컨테이너 접속
docker start alpamayo-r1 && docker attach alpamayo-r1

# 2. 패키지 동기화 (최초 1회 또는 의존성 변경 시)
cd /workspace/alpamayo-R1
uv sync --no-install-project

# 3. Jupyter 서버 시작
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```


