# 🗣️ 시선추적, 제스처 인식, STT 기반 AI 발표 트레이너 스피칭, SPitching!

## 👩🏻‍💻 SPitching! AI

> AI 발표 트레이너 AI – 시선 추적, 제스처 분석, 발표 유창성 피드백까지!
입력 영상을 3가지 AI 분석을 통해 처리 후, 결과를 구조화하여 S3 저장 및 백엔드 Webhook 전송
> 
> 
> `Python + FastAPI + Docker + SwaggerUI + AWS + Pycharm`
> 

## 🎯 **Project Goal**

발표에 대한 불안감을 극복하고 자신감을 높이기 위해

**시선, 제스처, 음성** 데이터를 분석한 종합 피드백을 제공하는 **AI 발표 연습 웹 애플리케이션**입니다.

- 시선추적, 제스처, 음성 기반 AI 분석 결과 시각화
- 대본과 실제 발표의 유사도 측정
- 반복 연습에 따른 발표력 향상률 제공
- AI 기반 Q&A 챗봇으로 질의응답 연습 가능

## 🧩 AI Server **Stack**

| 카테고리 | 기술 스택 |
| --- | --- |
| Framework | FastAPI (Uvicorn ASGI 서버) |
| Language | Python 3.10.12 |
| AI/ML | OpenCV, MediaPipe, TensorFlow, XGBoost, Google STT |
| Cloud Storage | AWS S3 |
| Deployment | Docker, AWS EC2 |
| CI/CD | GitHub Actions (구성만, 현재 미사용) |

## **📁** **Folder Structure**

```
AI_SERVER
├──📂.github
│   └── workflows
│     └── deploy.yml                      # GitHub Actions용 CI/CD 설정파일 (현재 미사용)
├──📂app
│   ├── __init__.py                       # 패키지 초기화 파일
│   └── main.py                           # FastAPI 서버 엔트리포인트 (라우팅 및 서비스 통합)
│   └── eyecontact.py                     # 1. 시선추적 기능
│   └── gesture.py                        # 2. 제스처 인식 기능
│   └── stt.py                            # 3. STT 기능
│   └── s3_upload.py                      # 분석 결과 영상 AWS S3 업로드 처리
├──📂label
│   └── gesture_keypoint_classifier_label.csv  # 제스처 분류 모델의 라벨 정의 파일
├──📂models
│   ├── filler_classifier_model.h5        # STT 어/음/그 다중분류 모델
│   ├── filter_determine_model.h5         # STT 추임새 판단용 이진분류 모델
│   └── gesture_XGB_model.pkl             # 제스처 분류용 XGBoost 모델
│
├── .dockerignore                         # Docker 이미지 빌드시 제외할 파일 설정
├── .gitignore                            # Git 추적 제외 파일 설정
├── Dockerfile                            # FastAPI 서버용 Docker 이미지 빌드 스크립트
├── README.md                             # 프로젝트 소개 및 실행 가이드 문서
└── requirements.txt                      # Python 의존성 패키지 목록
```

## **🔧 How to install**

```bash
git clone https://github.com/2024-capstone-project/SPitching-AI_SERVER.git
cd SPitching-AI_SERVER
python -m venv venv
source venv/bin/activate  # 윈도우면 venv\Scripts\activate
pip install -r requirements.txt
```

### **1. 실행 환경**

- Python 3.10.12
- OS : Ubuntu 22.04
- Docker 컨테이너 기반 실행
- AWS EC2 t3.large 인스턴스(vCPU 2, RAM 8GB)
- AWS S3 접근권한 필요

### **2. 환경 변수 설정**

.env 파일을 루트 디렉토리에 생성하고 다음 항목을 채워주세요 : 

```jsx
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=...
S3_BUCKET_NAME=...
WEBHOOK_URL=...
```

💡 **주의사항** : .env파일에는 민감한 정보가 포함되어 있으므로 Git에 올리지 말고, EC2 인스턴스에 수동으로 업로드하세요. (.env 파일이 필요하시다면 공식 이메일로 문의 바랍니다.)

## **🐳 How to build**

로컬에서 Docker 이미지 빌드

```bash
docker build -t spitching-ai-server .               # 도커 이미지 빌드
docker login                                         # 도커 허브 로그인
docker tag spitching-ai-server dayejang/spitching-ai-server:latest # 태그 지정
docker push dayejang/spitching-ai-server:latest      # 도커 허브로 푸시
```

## **☁️ How to deploy**

EC2 서버에 Docker 컨테이너 배포

```bash
docker rm -f spitching-ai-server                    # 기존 컨테이너 제거 - 최초 배포 시 생략
docker pull dayejang/spitching-ai-server:latest     # 최신 이미지 pull

docker run -d \
  --name spitching-ai-server \
  --env-file /home/ubuntu/spitching-ai-server/.env \
  -p 8000:8000 \
  dayejang/spitching-ai-server:latest
```

💡 **참고사항 :** 현재는 CI/CD 파이프라인을 사용하지 않으며, 로컬에서 build 및 push 후, EC2에서 pull 및 run하는 방식으로 수동 배포합니다.

## 🧪 How to test

- **개발 중**에는 Swagger UI를 통해 app/main.py에 정의된 엔드포인트를 수동 테스트할 수 있습니다. develop 브랜치의 코드를 실행시키고  http://localhost:8000/docs 에 접속하여, 약 1분 가량의 발표연습 영상을 삽입하면 3가지 AI 분석 결과를 직접 테스트 해볼 수 있습니다.
- **배포 후** 테스트는 다음과 같은 방식으로 진행합니다 :
    1. 프론트엔드를 통해 영상 업로드 → AI 서버로 분석 요청
    2. AI 서버는 하나의 영상에 대한 3가지 기능을 병렬적으로 분석한 뒤, 그 결과를 Webhook 방식을 통해 백엔드 API로 전송
    3. 백엔드 DB에 결과가 정상적으로 저장되었는지 확인 (MySQL 수동 조회) & AWS S3에 분석 처리된 영상이 정상적으로 저장되었는지 확인

💡 **유의사항 :** AI 서버 단독 테스트는 불가능하고, 반드시 Webhook 대상 Spring 백엔드 서버가 함께 실행되어야 합니다. 

## 🔗 **Related Links**

- ⛳ [프론트엔드 리드미](https://github.com/2024-capstone-project/SPitching-FE.git)
- 🔐 [백엔드 리드미](https://github.com/2024-capstone-project/SPitching-BE.git)
- 📋 시제품 사용설명서 *(구현 후 링크 연결 예정)*

## 🔍 Reference

- https://github.com/AtulkrishnanMU/The-Interview-Buster-Job-Interview-Coach.git
- https://github.com/EwhaSpeakUP/SpeakUP_ML.git
- https://github.com/TEAM-ITERVIEW/ML_SERVER.git
