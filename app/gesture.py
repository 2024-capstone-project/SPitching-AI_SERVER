import os
import copy
import itertools
from functools import lru_cache

import cv2
import numpy as np
import mediapipe as mp
import joblib
import xgboost as xgb

# 기존 .pkl 모델 로드
model = joblib.load('gesture_XGB_model.pkl')

# GPU에서 사용할 수 있도록 json으로 저장
model.save_model('gesture_XGB_model.json')

app_dir = os.path.dirname(__file__)
project_root = os.path.dirname(app_dir)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark[0:25]:  # Only take the first 23 landmarks
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark[11:25]):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_bounding_rect(use_brect, image, brect, rect_color):
    if use_brect:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), rect_color, 2)

    return image


def draw_info_text(image, brect, facial_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                  (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Gesture :' + facial_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


@lru_cache()
def load_model():
    # 모델 경로 설정
    model_path = os.path.join(project_root, 'models','gesture_XGB_model.pkl')
    booster = xgb.Booster({'predictor': 'gpu_predictor'})
    booster.load_model(model_path)
    return booster

def body(vid):
    # Rest of your code remains mostly unchanged, with adjustments for the Pose models

    # 발표력을 향상시키는 제스처 (+)
    explain = 0  # 1) 설명하는 손동작
    straight = 0  # 2) 바른 자세
    pos = 0 # explain + straight

    # 발표력을 저해하는 제스처 (-)
    crossed = 0 # 3) 팔짱 끼는 동작
    raised = 0 # 4) 팔을 들어올림
    face = 0 # 5) 손을 얼굴로 가져다댐

    count = 0
    cap_device = 0
    cap_width = 1920
    cap_height = 1080
    output_frames = []

    mp_draw = mp.solutions.drawing_utils
    use_brect = True

    # Camera preparation
    cap = cv2.VideoCapture(vid)  # You may need to adjust the camera source

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Load the XGBoost models
    xg_boost_model = load_model()

    # Read labels
    label_file_path = os.path.join(project_root, 'label', 'gesture_keypoint_classifier_label.csv')
    with open(label_file_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = f.read().splitlines()

    mode = 0

    progress = 20

    while True:
        # Process Key (ESC: end)
        # key = cv2.waitKey(10)
        # if key == 27:  # ESC
        # break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        # image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)

        # Find the dimensions of the frame
        height, width, _ = debug_image.shape

        # Determine the scaling factor to make the longest edge 600 pixels
        scaling_factor = 800 / max(height, width)

        # Calculate the new dimensions
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)

        # Resize the frame
        debug_image = cv2.resize(debug_image, (new_width, new_height))

        # Detection implementation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks is not None:
            count = count + 1

            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, results.pose_landmarks)

            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, results.pose_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Emotion classification using SVM models
            facial_emotion_id = xg_boost_model.predict([pre_processed_landmark_list])[0]

            # Determine the color of the bounding rectangle
            if facial_emotion_id == 0:
                crossed = crossed + 1
            elif facial_emotion_id == 1:
                raised = raised + 1
            elif facial_emotion_id == 2:
                explain = explain + 1
            elif facial_emotion_id == 3:
                straight = straight + 1
            elif facial_emotion_id == 4:
                face = face + 1

            if facial_emotion_id in [2, 3]:
                rect_color = (0, 255, 0)  # Green
                pos = pos + 1
            else:
                rect_color = (0, 0, 255)  # Red

            # Drawing part
            debug_image = draw_bounding_rect(use_brect, debug_image, brect, rect_color)
            debug_image = draw_info_text(
                debug_image,
                brect,
                keypoint_classifier_labels[facial_emotion_id])

        # Screen reflection
        # mp_draw.draw_landmarks(debug_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # cv2.imshow('Pose Recognition', debug_image)
        output_frames.append(debug_image)

    cap.release()
    # cv2.destroyAllWindows()

    try:
        # 긍정 제스처 2개 점수
        straight_score = int(round((straight / count) * 100, 1))
        explain_score = int(round((explain / count) * 100, 1))
        positive_score = straight_score + explain_score

        # 부정 제스처 3개 점수
        crossed_score = int(round((crossed / count) * 100, 1))
        raised_score = int(round((raised / count) * 100, 1))
        face_score = int(round((face / count) * 100, 1))
        negative_score = crossed_score + raised_score + face_score

        # 기본 점수
        base_score = 50

        # 제스처 점수는 최소 0점, 최대 100점, 소수점 없이 정수로 반환
        gesture_score = base_score + positive_score - negative_score
        gesture_score = max(0, min(100, gesture_score))

        message = ''

        messagep = '긍정적인 부분: '
        messagen = '개선이 필요한 부분: '

        if gesture_score >= 70:
            messagep += (
                " 발표 자세가 매우 안정적이고 손동작을 효과적으로 활용하여 청중의 이해를 도왔습니다. "
                "자연스럽고 자신감 있는 제스처 사용이 발표 전달력을 극대화하는 데 큰 도움이 되었습니다."
            )
        else:
            messagen += (
                " 발표 중 몸의 균형을 유지하고 적극적인 손동작을 활용하면 더욱 설득력 있는 발표가 가능합니다. "
                "적절한 손동작은 발표 내용의 핵심을 강조하는 데 효과적이며, 청중의 집중도를 높이는 데 기여할 수 있습니다."
            )

        if crossed_score >= 10:
            messagen += (
                " 발표 중 팔짱을 끼는 습관은 청중에게 방어적인 인상을 줄 수 있습니다. "
                "팔짱을 푸는 것만으로도 더욱 개방적이고 친근한 태도를 연출할 수 있으니 신경 써보세요."
            )

        if raised_score >= 10:
            messagen += (
                " 손을 과도하게 올리는 제스처는 자칫 청중에게 강압적이거나 불필요한 긴장감을 줄 수 있습니다. "
                "자연스럽고 절제된 손동작을 사용하면 보다 전문적인 발표 태도를 유지할 수 있습니다."
            )

        if face_score >= 10:
            messagen += (
                " 발표 중 얼굴을 자주 만지는 행동은 청중에게 긴장하거나 불안한 인상을 줄 수 있습니다. "
                "손의 움직임을 자연스럽게 조절하고, 시선과 제스처를 활용하여 자신감을 표현하는 것이 더욱 효과적입니다."
            )
        if messagep == '긍정적인 부분: ':
            messagep = ''
        if messagen == '개선이 필요한 부분: ':
            messagen = ''

        message = messagep + messagen

    except Exception as e:
        print(e)
        gesture_score = 0
        message = '사용자가 감지되지 않았습니다.'

    return output_frames, message, gesture_score, straight_score, explain_score, crossed_score, raised_score, face_score
