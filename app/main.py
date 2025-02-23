import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.stt import get_prediction, ResponseModel
from gesture import body
from head_eye import head_eye
from io import BytesIO
import os
import av
import shutil
import uuid

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'outputs')

app = FastAPI()
@app.post("/api/v1/model/head_eye")
async def analyze_head_eye(
        file: UploadFile = File(...),
        pk: str = Form(default=str(uuid.uuid4()))):

    # 1. 파일 저장
    video_filename = f"{pk}_{file.filename}"
    video_filepath = os.path.join(UPLOAD_FOLDER, video_filename)

    with open(video_filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. head_eye 분석 실행
    try:
        # head_eye 함수에서 필요한 데이터들 받기
        output_frames, message, head_score, eye_score = head_eye(video_filepath)

        # 3. 비디오 파일을 분석 결과와 함께 저장
        output_video_filename = f"processed_{uuid.uuid4().hex}.mp4"
        output_video_path = os.path.join(OUTPUT_FOLDER, output_video_filename)

        # PyAV를 사용해 비디오 처리 후 저장
        output_video_bytes = BytesIO()
        output_frames = np.array(output_frames)

        with av.open(output_video_bytes, 'w', format='mp4') as container:
            stream = container.add_stream('h264', rate=20)
            for frame in output_frames:
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)

        # 비디오 처리 결과를 저장
        with open(output_video_path, "wb") as out_file:
            out_file.write(output_video_bytes.getvalue())

        # 4. 처리된 비디오 URL 생성 (로컬에서는 로컬 URL을 사용)
        video_url = f"/static/outputs/{output_video_filename}"
        # 배포 환경에서는 서버의 URL을 기준으로 비디오 URL을 반환해야 하므로, videoUrl을 서버의 실제 URL 경로로 설정
        # app.mount("/static", StaticFiles(directory="static"), name="static")

        # 5. 분석 결과와 비디오 URL 반환
        return JSONResponse(content={
            "headScore": head_score,
            "eyeScore": eye_score,
            "message": message,
            "videoUrl": video_url
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing video: {str(e)}"})


@app.post("/api/v1/model/gesture")
async def analyze_gesture(
        file: UploadFile = File(...),
        pk: str = Form(default=str(uuid.uuid4()))):
    try:
        # 1. 파일 저장
        file_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, file_filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. 제스처 분석 실행
        # 'body' 함수에서 필요한 데이터들 받기
        output_frames, message, pose_score = body(file_path)

        # 3. PyAV를 사용해 비디오 처리 후 저장
        output_video_bytes = BytesIO()
        with av.open(output_video_bytes, 'w', format='mp4') as container:
            stream = container.add_stream('h264', rate=20)  # H264 codec, 20 fps
            for frame in output_frames:
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)

        # 4. 비디오 처리 결과를 저장
        output_video_filename = f"processed_{uuid.uuid4().hex}.mp4"
        output_video_path = os.path.join(OUTPUT_FOLDER, output_video_filename)
        with open(output_video_path, "wb") as out_file:
            out_file.write(output_video_bytes.getvalue())

        # 5. 처리된 비디오 URL 생성
        video_url = f"/static/outputs/{output_video_filename}"

        # 6. 분석 결과와 비디오 URL 반환
        return JSONResponse(content={
            "poseScore": pose_score,
            "message": message,
            "videoUrl": video_url
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing video: {str(e)}"})

@app.post("/api/v1/model/stt", response_model=ResponseModel)
async def stt(pk: str = Form(...), file: UploadFile = File(...)):
    webm_file = await file.read()
    transcript, statistics = await get_prediction(webm_file)

    response_data = {
        "interviewQuestionId": pk,
        "mumble": statistics[0]['mumble'],
        "silent": statistics[0]['silent'],
        "talk": statistics[0]['talk'],
        "time": statistics[0]['time'],
        "text": transcript
    }
    return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
