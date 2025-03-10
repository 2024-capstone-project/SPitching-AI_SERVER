import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from datetime import datetime

from app.stt import get_prediction
from gesture import body
from eyecontact import eyecontact
from io import BytesIO
import os
import av
import shutil
import uuid

# 프로젝트 root 디렉토리 설정
app_dir = os.path.dirname(__file__)
project_root = os.path.dirname(app_dir)

# static 디렉토리 내의 uploads와 outpouts 폴더 경로 설정
UPLOAD_FOLDER = os.path.join(project_root, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(project_root, 'static', 'outputs')

app = FastAPI()
@app.post("/api/v1/model/eyecontact")
async def analyze_eyecontact( file: UploadFile = File(...)):
    original_name, _ = os.path.splitext(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 업로드된 원본 파일 저장
    video_filename = f"{original_name}_{timestamp}.mp4"
    video_filepath = os.path.join(UPLOAD_FOLDER, video_filename)

    with open(video_filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. eyecontact 분석 실행
    try:
        # eyecontact 함수에서 필요한 데이터들 받기
        output_frames, message, eyecontact_score = eyecontact(video_filepath)

        # 3. 비디오 파일을 분석 결과와 함께 저장
        original_name, _ = os.path.splitext(file.filename)  # 확장자 제거한 원본 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 날짜 및 시간
        output_video_filename = f"{original_name}_eyecontact_{timestamp}.mp4"
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

        # 4. 처리된 비디오 URL 생성 (로컬에서는 로컬 URL 사용)
        video_url = f"/static/outputs/{output_video_filename}"
        # 배포 환경에서는 서버의 URL을 기준으로 비디오 URL을 반환해야 하므로, videoUrl을 서버의 실제 URL 경로로 설정
        # app.mount("/static", StaticFiles(directory="static"), name="static")

        # 5. 분석 결과와 비디오 URL 반환
        return JSONResponse(content={
            "eyecontactScore": eyecontact_score,
            "message": message,
            "videoUrl": video_url
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing video: {str(e)}"})


@app.post("/api/v1/model/gesture")
async def analyze_gesture(file: UploadFile = File(...)):

    try:
        # 1. 파일 저장
        original_name, _ = os.path.splitext(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        video_filename = f"{original_name}_{timestamp}.mp4"
        video_filepath = os.path.join(UPLOAD_FOLDER, video_filename)

        with open(video_filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. 제스처 분석 실행
        output_frames, message, pose_score = body(video_filepath)

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
        output_video_filename = f"{original_name}_gesture_{timestamp}.mp4"
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

@app.post("/api/v1/model/stt")
async def analyze_stt(file: UploadFile = File(...)):

    video_data = await file.read()
    statistics_filler, statistics_silence, stt_score_feedback, transcript = await get_prediction(video_data)

    response_data = {
        "statistics_filler": statistics_filler,
        "statistics_silence": statistics_silence,
        "stt_score_feedback": stt_score_feedback,
        "transcript": transcript
    }
    return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
