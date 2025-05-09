import httpx
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from datetime import datetime

from app.stt import get_prediction
from app.gesture import body
from app.eyecontact import eyecontact
from io import BytesIO
import os
import av
import shutil

# httpx의 로깅을 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("httpx")
logger.setLevel(logging.DEBUG)

# 프로젝트 root 디렉토리 설정
app_dir = os.path.dirname(__file__)
project_root = os.path.dirname(app_dir)

# static 디렉토리 내의 uploads와 outpouts 폴더 경로 설정
UPLOAD_FOLDER = os.path.join(project_root, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(project_root, 'static', 'outputs')

# 폴더가 존재하지 않으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 웹훅 URL 기본 설정 (백엔드 서버의 URL)
WEBHOOK_URL = "https://api.spitching.store/api/v1/feedback"

app = FastAPI()
@app.post("/api/v1/feedback/eyecontact")
async def analyze_eyecontact(
        file: UploadFile = File(...),
        userId: int = Form(...),
        presentationId: int = Form(...),
        practiceId: int = Form(...)):

    original_name, _ = os.path.splitext(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 업로드된 원본 파일 저장
    video_filename = f"{original_name}_{timestamp}.mp4"
    video_filepath = os.path.join(UPLOAD_FOLDER, video_filename)

    with open(video_filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # eyecontact 분석 실행
    try:
        output_frames, message, eyecontact_score = eyecontact(video_filepath)

        # 비디오 파일을 분석 결과와 함께 저장
        original_name, _ = os.path.splitext(file.filename)  # 확장자 제거한 원본 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 날짜 및 시간
        output_video_filename = f"{original_name}_시선추적_{timestamp}.mp4"
        output_video_path = os.path.join(OUTPUT_FOLDER, output_video_filename)

        # PyAV를 사용해 비디오 처리 후 저장
        output_video_bytes = BytesIO()
        output_frames = np.array(output_frames)

        # 비디오 파일에서 프레임 레이트 가져오기
        input_container = av.open(video_filepath)
        input_stream = input_container.streams.video[0]
        fps = input_stream.average_rate

        with av.open(output_video_bytes, 'w', format='mp4') as container:
            stream = container.add_stream('h264', rate=fps)
            for frame in output_frames:
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)

            # 모든 프레임을 처리한 후, 남은 패킷을 마무리
            packet = stream.encode(None) # None을 인코딩하여 남은 프레임을 모두 처리
            if packet:
                container.mux(packet)

        # 비디오 처리 결과를 저장
        with open(output_video_path, "wb") as out_file:
            out_file.write(output_video_bytes.getvalue())

        # 처리된 비디오 URL 생성 (로컬에서는 로컬 URL 사용)
        video_url = f"/static/outputs/{output_video_filename}"
        # 배포 환경에서는 서버의 URL을 기준으로 비디오 URL을 반환해야 하므로, videoUrl을 서버의 실제 URL 경로로 설정
        # app.mount("/static", StaticFiles(directory="static"), name="static")

        # 분석 결과와 비디오 URL 반환
        eyecontact_feedback = {
            "userId": userId,
            "presentationId": presentationId,
            "practiceId": practiceId,
            "eyecontactScore": eyecontact_score,
            "videoUrl": video_url
        }

        # 웹훅 URL에 시선추적 경로 추가
        webhook_url_eyecontact = f"{WEBHOOK_URL}/eyecontact"

        # 웹훅 호출
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(webhook_url_eyecontact, json=eyecontact_feedback)

            # 응답 상태코드와 본문로그 출력
            logger.debug(f"Webhook response status code : {response.status_code}")
            logger.debug(f"Webhook response body : {response.text}")

            if response.status_code != 200:
                return JSONResponse(status_code=500, content={
                    "message": f"Error sending webhook: {response.status_code} : {response.text}"
                })

        # 웹훅 호출 성공
        return JSONResponse(content=eyecontact_feedback)

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing video: {str(e)}"})


@app.post("/api/v1/feedback/gesture")
async def analyze_gesture(
        file: UploadFile = File(...),
        userId: int = Form(...),
        presentationId: int = Form(...),
        practiceId: int = Form(...)
        ):

    try:
        # 파일 저장
        original_name, _ = os.path.splitext(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        video_filename = f"{original_name}_{timestamp}.mp4"
        video_filepath = os.path.join(UPLOAD_FOLDER, video_filename)

        with open(video_filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 제스처 분석 실행
        output_frames, message, gesture_score, straight_score, explain_score, crossed_score, raised_score, face_score = body(video_filepath)

        # 입력 비디오에서 FPS 추출
        input_container = av.open(video_filepath)
        input_stream = input_container.streams.video[0]
        fps = input_stream.average_rate # 입력 비디오의 fps 값 가져오기

        # PyAV를 사용해 비디오 처리 후 저장
        output_video_bytes = BytesIO()
        with av.open(output_video_bytes, 'w', format='mp4') as container:
            stream = container.add_stream('h264', rate=fps)
            for frame in output_frames:
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)

            # 모든 프레임을 처리한 후, 남은 패킷을 마무리
            packet = stream.encode(None) # None을 인코딩 하여 남은 프레임을 모두 처리
            if packet:
                container.mux(packet)

        # 비디오 처리 결과를 저장
        output_video_filename = f"{original_name}_제스처_{timestamp}.mp4"
        output_video_path = os.path.join(OUTPUT_FOLDER, output_video_filename)
        with open(output_video_path, "wb") as out_file:
            out_file.write(output_video_bytes.getvalue())

        # 처리된 비디오 URL 생성
        video_url = f"/static/outputs/{output_video_filename}"

        gesture_feedback = {
            "userId": userId,
            "presentationId": presentationId,
            "practiceId": practiceId,
            "gestureScore": int(gesture_score),
            "straightScore": int(straight_score),
            "explainScore": int(explain_score),
            "crossedScore": int(crossed_score),
            "raisedScore": int(raised_score),
            "faceScore": int(face_score),
            "videoUrl": video_url
            # 제스처 피드백 메세지는 제외한 상태
        }

        # 웹훅 URL에 제스처 경로 추가
        webhook_url_gesture = f"{WEBHOOK_URL}/gesture"

        # 웹훅 호출
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(webhook_url_gesture, json=gesture_feedback)

            # 응답 상태코드와 본문로그 출력
            logger.debug(f"Webhook response status code : {response.status_code}")
            logger.debug(f"Webhook response body : {response.text}")

            if response.status_code != 200:
                return JSONResponse(status_code=500, content={
                    "message" : f"Error sending webhook: {response.status_code} : {response.text}"
                })

        # 웹훅 호출 성공
        return JSONResponse(content=gesture_feedback)

        # # 분석 결과와 비디오 URL 반환
        # return JSONResponse(content={
        #     "gestureScore": gesture_score,
        #     "straightScore": straight_score,
        #     "explainScore": explain_score,
        #     "crossedScore": crossed_score,
        #     "raisedScore": raised_score,
        #     "faceScore": face_score,
        #     "message": message, # 제스처 피드백 메세지 포함된 기존 반환값
        #     "videoUrl": video_url
        # })

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing video: {str(e)}"})

@app.post("/api/v1/feedback/stt")
async def analyze_stt(
        file: UploadFile = File(...),
        userId: int = Form(...),
        presentationId: int = Form(...),
        practiceId: int = Form(...)):

    try:
        # 파일 읽기
        video_data = await file.read()

        # STT 분석 실행
        statistics_filler, statistics_silence, fluency_score, transcript = await get_prediction(video_data)

        statistics_filler = statistics_filler[0]
        statistics_silence = statistics_silence[0]

        # 피드백 데이터 구성
        stt_feedback = {
            "userId": userId,
            "presentationId": presentationId,
            "practiceId": practiceId,
            "fluencyScore": fluency_score,
            "statisticsFiller": [
                {
                    "eo": statistics_filler['어'],
                    "eum": statistics_filler['음'],
                    "geu": statistics_filler['그'],
                    "totalFillerCount": statistics_filler['불필요한 추임새 총 개수'],
                    "fillerRatio": statistics_filler['발화시간 대비 추임새 비율(%)']
                }
            ],
            "statisticsSilence": [
                {
                    "silenceRatio": statistics_silence['침묵비율(%)'],
                    "speakingRatio": statistics_silence['발화비율(%)'],
                    "totalPresentationTime": statistics_silence['전체발표시간(초)']
                }
            ],
            "transcript": transcript
        }

        # 웹훅 URL에 STT 경로 추가
        webhook_url_stt = f"{WEBHOOK_URL}/stt"

        # 웹훅 호출
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(webhook_url_stt, json=stt_feedback)

            # 응답 상태코드와 본문 로그 출력
            logger.debug(f"Webhook response status code : {response.status_code}")
            logger.debug(f"Webhook response body : {response.text}")

            if response.status_code != 200:
                return JSONResponse(status_code=500, content={
                    "message": f"Error sending webhook: {response.status_code} : {response.text}"
                })

        # 웹훅 호출 성공
        return JSONResponse(content=stt_feedback)

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing STT: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)