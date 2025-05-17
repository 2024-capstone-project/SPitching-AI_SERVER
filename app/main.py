import httpx
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.s3_upload import upload_file_to_s3
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

# 웹훅 URL 기본 설정 (백엔드 서버의 URL)
WEBHOOK_URL = "https://api.spitching.store/api/v1/feedback"

app = FastAPI()
@app.post("/api/v1/feedback/eyecontact")
async def analyze_eyecontact(
        file: UploadFile = File(...),
        userId: int = Form(...),
        presentationId: int = Form(...),
        practiceId: int = Form(...)
        ):

    try:
        # 업로드 파일 읽고 tmp 디렉토리에 저장
        video_data = await file.read()
        tmp_input_path = f"/tmp/{file.filename}"
        with open(tmp_input_path, "wb") as f:
            f.write(video_data)

        # eyecontact 분석 실행
        output_frames, message, eyecontact_score = eyecontact(tmp_input_path)

        # 분석 결과 비디오 바이트로 변환
        output_video_bytes = BytesIO()
        output_frames = np.array(output_frames)

        # 비디오 파일에서 프레임 레이트 가져오기
        input_container = av.open(tmp_input_path)
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

        # S3에 비디오 처리결과 업로드
        base_filename = os.path.splitext(file.filename)[0]
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
        # 업로드 파일 읽고 tmp 디렉토리에 저장
        video_data = await file.read()
        tmp_input_path = f"/tmp/{file.filename}"
        with open(tmp_input_path, "wb") as f:
            f.write(video_data)

        # 제스처 분석 실행
        output_frames, message, gesture_score, straight_score, explain_score, crossed_score, raised_score, face_score = body(tmp_input_path)

        # 입력 비디오에서 FPS 추출
        input_container = av.open(tmp_input_path)
        input_stream = input_container.streams.video[0]
        fps = input_stream.average_rate # 입력 비디오의 fps 값 가져오기

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

        # S3에 비디오 처리결과 업로드
        base_filename = os.path.splitext(file.filename)[0]
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

            # 응답 상태코드와 본문로그 출력
            logger.debug(f"Webhook response status code : {response.status_code}")
            logger.debug(f"Webhook response body : {response.text}")

            if response.status_code != 200:
                return JSONResponse(status_code=500, content={
                    "message" : f"Error sending webhook: {response.status_code} : {response.text}"
                })

        # 웹훅 호출 성공
        return JSONResponse(content=gesture_feedback)

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