import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from gesture import body
from head_eye import head_eye

app = FastAPI()

@app.post("/api/v1/model/gesture")
async def analyze_gesture(file: UploadFile = File(...), pk: str = Form(default=str(uuid.uuid4()))) -> JSONResponse:
    file_content = await file.read()
    nparr = np.frombuffer(file_content, np.uint8)
    vid = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Progress bar placeholder (substitute)
    class LoadingBar:
        def progress(self, value):
            pass

    loading_bar_pose = LoadingBar()

    # gesture.py의 body 함수를 호출
    output_frames, message, pos_score = body(vid, loading_bar_pose)

    # processed_output_video_url 임시로 대체
    video_url = f"https://gesture_feedback_video/{pk}.mp4"

    response = {
        "gestureScore": pos_score,
        "message": message,
        "videoUrl": video_url
    }

    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
