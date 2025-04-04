import cv2
import mediapipe as mp
import numpy as np
import math

# landmarks

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# iris
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_iris_center = [468]
R_iris_center = [473]

CHIN = [167, 393]

THAADI = [200]

NOSE = [4]

LH_LEFT = [33]
LH_RIGHT = [133]
RH_LEFT = [362]
RH_RIGHT = [263]

# colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)


# Eyecontact FUNCTIONS
def find_leftmost_rightmost(coordinates):
    leftmost = (float('inf'), float('inf'))
    rightmost = (-float('inf'), -float('inf'))

    for x, y in coordinates:
        leftmost = (min(leftmost[0], x), min(leftmost[1], y))
        rightmost = (max(rightmost[0], x), max(rightmost[1], y))

    return leftmost, rightmost


def transform_coordinates(coordinates):
    leftmost, rightmost = find_leftmost_rightmost(coordinates)

    # Calculate the scaling factor
    scaling_factor = 100 / (rightmost[0] - leftmost[0])

    transformed_coordinates = []
    for x, y in coordinates:
        # Translate to (50, 50)
        translated_x = x - leftmost[0] + 50
        translated_y = y - leftmost[1] + 50

        # Scale the coordinates
        scaled_x = translated_x * scaling_factor
        scaled_y = translated_y * scaling_factor

        transformed_coordinates.append([int(scaled_x), int(scaled_y)])

    return [np.array(transformed_coordinates)]


def landmarkdet(img, results):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]

    return mesh_coord


# euclidean dist
def eucli(p1, p2):
    x, y = p1
    x1, y1 = p2
    dist = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return dist


def head_pose_estimate(model_points, landmarks, K):
    dist_coef = np.zeros((4, 1))
    ret, rvec, tvec = cv2.solvePnP(model_points, landmarks, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)

    rot_mat = cv2.Rodrigues(rvec)[0]
    P = np.hstack((rot_mat, np.zeros((3, 1), dtype=np.float64)))
    eulerAngles = cv2.decomposeProjectionMatrix(P)[6]
    yaw = int(eulerAngles[1, 0] * 360)
    pitch = int(eulerAngles[0, 0] * 360)
    roll = eulerAngles[2, 0] * 360
    return roll, yaw, pitch


def newirispos2(transformed_eye_coordinates, image):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p1 = flat_cords[0]
    p4 = flat_cords[8]
    iris = flat_cords[17]

    p = (p1 + p4) / 2

    con = p - iris
    con = (abs(con[0]), abs(con[1]))

    point1_int = (int(p[0]), int(p[1]))
    point2_int = (int(iris[0]), int(iris[1]))

    #cv2.circle(image, point1_int, 5, (0, 0, 255), -1)  # Red color for point1
    #cv2.circle(image, point2_int, 5, (0, 255, 0), -1)  # Green color for point2

    return con


# blink ratio
def newbratio(transformed_eye_coordinates):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p2 = flat_cords[13]
    p6 = flat_cords[3]
    p3 = flat_cords[11]
    p5 = flat_cords[5]
    right = flat_cords[0]
    left = flat_cords[8]

    earclosed = ((5.385164807134504 + 4.47213595499958) / 2 * (eucli(right, left)))
    earopen = ((35.12833614050059 + 31.400636936215164) / 2 * (eucli(right, left)))

    ear = (eucli(p2, p6) + eucli(p3, p5)) / 2 * (eucli(right, left))

    thresh = (earopen + earclosed) / 2

    if ear <= thresh:
        return True
    else:
        return False

def draw_bounding_rect(use_brect, image, brect, rect_color):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), rect_color, 2)

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_info_text(image, brect, facial_text):
    info_text = ''
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                  (0, 0, 0), -1)

    if facial_text != "":
        info_text = facial_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def eyecontact(vid):
    count = 0
    text = ''

    eyecount = 0
    headcount = 0

    straight = 0

    blinkcount = 0
    blinklist = []

    fps = 0

    prev = 0
    consecutive_blink = 0
    blink_too_long = 0

    output_frames = []

    map_face_mesh = mp.solutions.face_mesh

    rect_color = (0, 255, 0)  # Green

    cap = cv2.VideoCapture(vid)

    output_file = r'E:\project demo\media\eye-contact.mp4'

    # print(output_file)

    # Create a VideoCapture object
    # cap = cv2.VideoCapture(r'E:\website files\bodylang\myapp\pgms\WhatsApp Video 2023-08-18 at 20.00.43.mp4')

    # Check if the camera or video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'avc1' for H.264 codec
    fps = 30.0  # Frames per second (you can adjust this)

    # Define the output video dimensions (use the same as the input if not resizing)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    progress = 30

    with map_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as face_mesh:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            # Find the dimensions of the frame
            height, width, _ = frame.shape

            # Determine the scaling factor to make the longest edge 600 pixels
            scaling_factor = 800 / max(height, width)

            # Calculate the new dimensions
            new_height = int(height * scaling_factor)
            new_width = int(width * scaling_factor)

            # Resize the frame
            frame = cv2.resize(frame, (new_width, new_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fps = fps + 1

            face_3d = []
            face_2d = []

            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    count = count + 1

                    if fps % 1441 == 0:
                        blinklist.append(blinkcount)
                        blinkcount = 0
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * width, lm.y * height)
                                nose_3d = (lm.x * width, lm.y * height, lm.z * 3000)

                            x, y = int(lm.x * width), int(lm.y * height)

                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * width

                    cam_matrix = np.array([
                        [focal_length, 0, height / 2],
                        [0, focal_length, width / 2],
                        [0, 0, 1]
                    ])

                    # distortion parameters
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    # X,Y,Z = head_pose_estimate(face_3d, face_2d, cam_matrix)
                    # print(X,Y,Z)

                    rot_matrix, jac = cv2.Rodrigues(rot_vec)

                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_matrix)

                    x = angles[0] * 360  # pitch
                    y = angles[1] * 360  # yaw
                    # z = angles[2] * 360

                    # print(x,y,z)

                    # x_deg = angles[0]
                    # y_deg = angles[1]
                    # z_deg = angles[2]

                    # cv2.putText(frame, str(z), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                    # print(x,' ', y)

                    # cv2.putText(frame, str(int(y)) + ", " + str(int(x)), (100,200), cv2.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)

                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                     dist_matrix)

                    p1 = (int(nose_2d[0]) - 100, int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10) - 100, int(nose_2d[1] - x * 10))

                    # if not (-5 < int(y) < 5 and -5 < int(x) < 5):
                    #     straight = 0
                    #     cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                    #     cv2.putText(frame, 'Look straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                    #     # print(count, ": Head not straight")
                    # else:
                    #     # cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                    #     # cv2.putText(frame, 'Head straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                    #     straight = 1
                    #     # print(count, ": Head straight")

                mesh_coords = landmarkdet(frame, results)

                mesh_points = np.array(mesh_coords)

                # cv2.line(frame, p1, p2, (255, 0, 0), 3)
                fhead = tuple(mesh_points[151])
                chin = tuple(mesh_points[175])

                # cv2.line(frame, fhead, chin, (0, 255, 0), 2)

                threshold = 10

                # print(abs(fhead[0] - chin[0]))

                # Check if the slope is almost straight
                # if straight == 1:
                #     if abs(fhead[0] - chin[0]) < threshold:
                #         straight = 1
                #         cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                #         cv2.putText(frame, 'Head straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                #         headcount = headcount + 1
                #     else:
                #         straight = 0
                #         cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                #         cv2.putText(frame, 'Look straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                # detecting head_eye

                forehead = tuple(mesh_points[FACE_OVAL][0])
                # print(forehead)

                lip_point1 = tuple(mesh_points[LOWER_LIPS][0])
                lip_point2 = tuple(mesh_points[LOWER_LIPS][16])
                lip_point3 = tuple(mesh_points[UPPER_LIPS][13])
                lip_point4 = tuple(mesh_points[LOWER_LIPS][10])

                # cv2.circle(frame, mesh_points[CHIN][0], 2, (255,0,255), 1, cv2.LINE_AA)
                # cv2.circle(frame, mesh_points[CHIN][1], 2, (255,0,255), 1, cv2.LINE_AA)

                # cv2.circle(frame, mesh_points[L_iris_center][0], 2, (255,0,255), 1, cv2.LINE_AA)
                # cv2.circle(frame, mesh_points[R_iris_center][0], 2, (255,0,255), 1, cv2.LINE_AA)

                # cv2.circle(frame, mesh_points[LOWER_LIPS][0], 2, (255,0,255), 1, cv2.LINE_AA)
                # cv2.circle(frame, mesh_points[LOWER_LIPS][10], 2, (255,0,255), 1, cv2.LINE_AA)
                # cv2.circle(frame, mesh_points[LOWER_LIPS][16], 2, (255,0,255), 1, cv2.LINE_AA)

                x1, y1 = mesh_points[LOWER_LIPS][0]
                x2, y2 = mesh_points[LOWER_LIPS][10]
                x3, y3 = mesh_points[LIPS][25]
                x4, y4 = mesh_points[THAADI][0]

                x5, y5 = mesh_points[CHIN][0]
                x6, y6 = mesh_points[CHIN][1]

                eye_coordinates = []
                eye_cont_coordinates = []
                r_eye_cont_coordinates = []

                for i in LEFT_EYE:
                    eye_coordinates.append(tuple(mesh_points[i]))

                LEFT_EYE_and_IRIS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
                                     468, 473]
                RIGHT_EYE_and_IRIS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 473,
                                      468]

                for i in RIGHT_EYE_and_IRIS:
                    r_eye_cont_coordinates.append(tuple(mesh_points[i]))

                for i in LEFT_EYE_and_IRIS:
                    eye_cont_coordinates.append(tuple(mesh_points[i]))

                # Transform the coordinates
                transformed_eye_coordinates = transform_coordinates(eye_coordinates)
                transformed_eyecont_coordinates = transform_coordinates(eye_cont_coordinates)
                rtransformed_eyecont_coordinates = transform_coordinates(r_eye_cont_coordinates)

                # cv2.polylines(frame, transformed_eye_coordinates, True, GREEN)

                blink = newbratio(transformed_eye_coordinates)
                if blink:
                    if prev == 1:
                        if consecutive_blink <= 72:
                            consecutive_blink = consecutive_blink + 1
                        else:
                            blink_too_long = 1
                    prev = 1
                else:
                    if prev == 1:
                        blinkcount = blinkcount + 1
                    prev = 0
                    consecutive_blink = 0

                # cont =  newirispos2(transformed_eyecont_coordinates)
                cont = newirispos2(transformed_eyecont_coordinates, frame)
                rcont = newirispos2(rtransformed_eyecont_coordinates, frame)

                # print((cont[0]+rcont[0])/2, (cont[1]+rcont[1])/2)

                if 0 <= ((cont[0] + rcont[0]) / 2) <= 2.5 and 0 <= ((cont[1] + rcont[1]) / 2) <= 3.5:
                    contact = True
                else:
                    contact = False

                # if straight == 1:
                if not blink:
                    if contact:
                        text = 'Eye Contact'
                        rect_color = (0, 255, 0)  # Green
                        # print(count, ": Eye contact")
                        eyecount = eyecount + 1
                        '''
                    if 45<(int(ratio*100))<55 and 11<=int((1/topratio)*100)<=17:
                            #cv2.rectangle(frame, (25, 40), (200, 66), BLACK, -1)
                            text = 'Eye Contact'
                            rect_color = (0, 255, 0)  # Green
                            #print(count, ": Eye contact")
                            #cv2.putText(frame, str(eucli(tuple(center_right),rv_bottom)), (200,100), cv2.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)
                    '''
                    else:
                        text = 'Not Eye contact'
                        rect_color = (0, 0, 255)  # Red
                        # print(count, ": Not Eye contact")
                else:
                    text = 'Blink'
                    rect_color = (0, 0, 255)  # Red
                    # print(count, ": Not Eye contact")
                # else:
                # text = "Look straight"
                # rect_color = (0, 0, 255)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate bounding rectangle
                    brect = calc_bounding_rect(frame, face_landmarks)

                    # Draw bounding rectangle around the head
                    frame = draw_bounding_rect(True, frame, brect, rect_color)

                    frame = draw_info_text(
                        frame,
                        brect,
                        text)

            # cv2.imshow('Frame', frame)
            output_frames.append(frame)
            # st.image(frame, channels="BGR", caption="Processed Frame")

        # Release the VideoWriter object.

        # cv2.imshow('Image', frame)
        # cv2.destroyAllWindows()

    try:
        eyecontact_score = int(round(((eyecount / count) * 100), 1))

        messagep = '긍정적인 부분: '
        messagen = '개선이 필요한 부분: '

        if eyecontact_score <= 25:
            messagen += (
                "발표 중 시선을 자주 피하는 경향이 있습니다. "
                "이는 청중에게 자신감이 부족하다는 인상을 줄 수 있습니다. "
                "청중과의 신뢰를 형성하고 메시지 전달력을 높이기 위해 의식적으로 아이 컨택을 늘리는 연습이 필요합니다. "
                "거울 앞에서 연습하거나, 발표 영상을 촬영하여 본인의 시선 처리를 점검해보는 것도 좋은 방법입니다."
            )

        elif 25 < eyecontact_score <= 50:
            messagen += (
                "시선 처리가 제한적이며, 발표 도중 눈이 자주 흔들리는 경향이 있습니다. "
                "이는 청중과의 소통을 단절시키고, 발표자가 긴장하고 있다는 인상을 줄 수 있습니다. "
                "발표할 때 주요 청중을 여러 번 바라보며 고르게 시선을 배분하는 연습을 해보세요. "
                "이를 통해 보다 자연스럽고 자신감 있는 발표를 할 수 있습니다."
            )

        elif 50 < eyecontact_score <= 75:
            messagen += (
                "시선 처리가 비교적 안정적이지만, 청중과의 아이 컨택을 더욱 지속적으로 유지하면 발표의 신뢰도를 한층 높일 수 있습니다. "
                "중요한 메시지를 전달할 때는 특정 청중을 바라보며 강조하는 방식으로 말하면 더욱 효과적입니다. "
                "또한, 발표 공간 전체를 두루 살펴보면서 다양한 청중과 시선을 맞추는 습관을 들이면 더욱 자연스럽고 설득력 있는 발표가 될 것입니다."
            )

        elif 75 < eyecontact_score <= 90:
            messagep += (
                "전반적으로 안정적이고 자연스러운 시선 처리를 유지했습니다. "
                "청중과의 연결이 원활하게 이루어졌으며, 발표 내용의 신뢰도를 높이는 데 긍정적인 영향을 주었습니다. "
                "앞으로도 중요한 핵심 메시지를 전달할 때 적절한 시선 처리를 활용하면 더욱 효과적인 발표가 될 것입니다."
            )

        elif 90 < eyecontact_score:
            messagep += (
                "훌륭합니다! 청중과의 적극적인 아이 컨택을 통해 강한 신뢰감과 자신감을 전달했습니다. "
                "눈을 피하지 않고 청중을 주의 깊게 바라보며 말하는 모습은 매우 인상적이었으며, 발표의 설득력을 높이는 데 큰 역할을 했습니다. "
                "이러한 자연스럽고 안정적인 시선 처리는 청중과의 소통을 원활하게 만들며,발표자로서의 전문성을 더욱 돋보이게 합니다."
            )

        if messagep == '긍정적인 부분: ':
            messagep = ''
        if messagen == '개선이 필요한 부분: ':
            messagen = ''

        message = messagep + messagen

    except:
        eyecontact_score = 0
        message = '얼굴이 감지되지 않았습니다.'

    return output_frames, message, eyecontact_score
