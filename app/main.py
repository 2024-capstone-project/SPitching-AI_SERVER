import httpx
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from io import BytesIO
from dotenv import load_dotenv
import os
import av
import asyncio
from fastapi.middleware.cors import CORSMiddleware

from app.s3_upload import upload_file_to_s3
from app.stt import get_prediction
from app.gesture import body
from app.eyecontact import eyecontact

load_dotenv()
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

app = FastAPI()

# CORS 설정
origins = [
    "https://www.spitching.store",
    "https://spitching.store",
    "https://spitching.vercel.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # 위 origin(프론트 주소들)만 허용
    allow_credentials=True,    # 쿠키,세션 등 자격증명 포함 허용
    allow_methods=["*"],       # 모든 메서드 허용
    allow_headers=["*"],       # 프론트 요청 시 사용하는 모든 헤더 허용
)

# httpx의 로깅을 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("httpx")
logger.setLevel(logging.DEBUG)

@app.post("/api/v1/feedback")
async def analyze_all_feedback(
    file: UploadFile = File(...),
    userId: int = Form(...),
    presentationId: int = Form(...),
    practiceId: int = Form(...)
):
    try:
        # 영상 1회 읽기
        video_data = await file.read()
        tmp_input_path = f"/tmp/{file.filename}"
        with open(tmp_input_path, "wb") as f:
            f.write(video_data)
        base_filename = os.path.splitext(file.filename)[0]

        # FPS 추출 1회만
        input_container = av.open(tmp_input_path)
        fps = input_container.streams.video[0].average_rate

        # 백그라운드에서 병력적으로 분석 작업 시작 (각 분석이 끝나는 대로 웹훅 호출)
        asyncio.create_task(analyze_eyecontact(tmp_input_path, base_filename, fps, userId, presentationId, practiceId))
        asyncio.create_task(analyze_gesture(tmp_input_path, base_filename, fps, userId, presentationId, practiceId))
        asyncio.create_task(analyze_stt(video_data, userId, presentationId, practiceId))

        # 프론트에 바로 응답
        return {"message": "All feedback analysis started"}

    except Exception as e:
        logger.error(f"Failed to start analysis: {str(e)}", exc_info=True)
        return {"message": f"Internal server error: {str(e)}"}

async def analyze_eyecontact(
    path: str, base_filename: str, fps, userId: int, presentationId: int, practiceId: int
) -> None:

    try:
        # eyecontact 분석 실행
        output_frames, message, eyecontact_score = eyecontact(path)

        # 분석 결과 비디오 바이트로 변환
        output_video_bytes = BytesIO()

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

        output_video_bytes.seek(0)

        # S3에 비디오 처리결과 업로드
        s3_key = f"outputs/{base_filename}_eyecontact.mp4"
        video_url = upload_file_to_s3(output_video_bytes.getvalue(), s3_key)

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
            logger.info(f"[Eyecontact] Webhook response: {response.status_code} {response.text}")

    except Exception as e:
        logger.error(f"[Eyecontact] Error: {str(e)}", exc_info=True)

async def analyze_gesture(
    path: str, base_filename: str, fps, userId: int, presentationId: int, practiceId: int
) -> None:
    try:
        # 제스처 분석 실행
        output_frames, message, gesture_score, straight_score, explain_score, crossed_score, raised_score, face_score = body(path)

        # PyAV를 사용해 mp4 바이트 변환 후 저장
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

        output_video_bytes.seek(0)

        # S3에 비디오 처리결과 업로드
        s3_key = f"outputs/{base_filename}_gesture.mp4"
        video_url = upload_file_to_s3(output_video_bytes.getvalue(), s3_key)

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
            logger.info(f"[Gesture] Webhook response: {response.status_code} {response.text}")

    except Exception as e:
        logger.error(f"[Gesture] Error: {str(e)}")

async def analyze_stt(
    video_data: bytes, userId: int, presentationId: int, practiceId: int
) -> None:

    try:
        # STT 분석 실행
        statistics_filler, statistics_silence, fluency_score, transcript = await get_prediction(video_data)
        statistics_filler = statistics_filler[0] if statistics_filler else {}
        statistics_silence = statistics_silence[0] if statistics_silence else {}

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
            logger.info(f"[STT] Webhook response: {response.status_code} {response.text}")

    except Exception as e:
        logger.error(f"[STT] Error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)