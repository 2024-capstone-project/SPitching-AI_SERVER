import tensorflow as tf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa
import numpy as np
import io
import os
import speech_recognition as sr
import subprocess

app_dir = os.path.dirname(__file__)
project_root = os.path.dirname(app_dir)

OUTPUT_FOLDER = os.path.join(project_root, 'static', 'outputs')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # 폴더 없으면 생성
temp_wav_path = os.path.join(OUTPUT_FOLDER, "temp.wav")

# 모델 경로 설정
classifier_model_path = os.path.join(project_root, 'models', 'filler_classifier_model.h5')
determine_model_path = os.path.join(project_root, 'models', 'filter_determine_model.h5')

# 모델 불러오기
filler_classifier_model = tf.keras.models.load_model(classifier_model_path)
filter_determine_model = tf.keras.models.load_model(determine_model_path)

# 전역 변수 선언
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

frame_length = 0.025
frame_stride = 0.0010

# 어떤 비디오 파일이든 WebM으로 변환
def convert_video_to_webm(video_content: bytes) -> bytes:
    ffmpeg_process = subprocess.Popen(
        [
            "ffmpeg", "-i", "pipe:",  # 입력을 파이프에서 받음
            "-c:v", "libvpx", "-b:v", "1M",  # 비디오 코덱 설정 (VP8)
            "-c:a", "libvorbis",  # 오디오 코덱 설정 (Vorbis)
            "-f", "webm", "pipe:"  # WebM 포맷으로 출력
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    # FFmpeg에 비디오 데이터 전달하여 변환 실행
    webm_content, _ = ffmpeg_process.communicate(input=video_content)

    return webm_content

# Helper functions for processing audio
def convert_webm_to_wav(webm_content):
    input_data = webm_content
    ffmpeg_process = subprocess.Popen(
        ["ffmpeg", "-i", "pipe:", "-vn", "-acodec", "pcm_s24le", "-ar", "48000", "-ac", "2", "-f", "wav", "pipe:"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    wav_content, _ = ffmpeg_process.communicate(input=input_data)
    return wav_content

def match_target_amplitude(sound, target_dBFS):
    normalized_sound = sound.apply_gain(target_dBFS - sound.dBFS)
    return normalized_sound

def predict_filler(audio_file):
    audio_file.export(temp_wav_path, format="wav")

    wav, sr = librosa.load(temp_wav_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=wav)
    padded_mfcc = pad2d(mfcc, 40)
    padded_mfcc = np.expand_dims(padded_mfcc, 0)

    result = filler_classifier_model.predict(padded_mfcc)

    # 판별 완료된 음성 파일 삭제
    os.remove(temp_wav_path)

    if result[0][0] >= result[0][1]:
        return 0
    else:
        return 1

def predict_filler_type(audio_file):
    wav, sr = librosa.load(temp_wav_path, sr=16000)
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))

    mfcc = librosa.feature.mfcc(y=wav)
    padded_mfcc = pad2d(mfcc, 40)
    padded_mfcc = np.expand_dims(padded_mfcc, 0)
    result = filler_classifier_model.predict(padded_mfcc)

    # 판별 완료된 음성 파일 삭제
    os.remove(temp_wav_path)
    return np.argmax(result)

def shorter_filler(json_result, audio_file, min_silence_len, start_time, non_silence_start):

    # 침묵 길이를 더 짧게
    min_silence_length = (int)(min_silence_len / 1.2)

    intervals = detect_nonsilent(audio_file,
                                 min_silence_len=min_silence_length,
                                 silence_thresh=-32.64
                                 )
    for interval in intervals:
        interval_audio = audio_file[interval[0]:interval[1]]

        # padding 40 길이 이상인 경우 더 짧게
        if (interval[1] - interval[0] >= 460):
            non_silence_start = shorter_filler(json_result, interval_audio, min_silence_length,
                                               interval[0] + start_time, non_silence_start)

        else: # padding 40 길이보다 짧은 경우 predict
            if predict_filler(interval_audio) == 0: # 추임새인 경우
                json_result.append({'start': non_silence_start, 'end': start_time + interval[0], 'tag': '1000'}) # tag: 1000 means non-slience
                non_silence_start = start_time + interval[0]

                # 추임새 tagging
                json_result.append({'start': start_time + interval[0], 'end': start_time + interval[1], 'tag': '1111'}) # tag: 1111 means filler word

    return non_silence_start

def create_json(audio_file):
    intervals_jsons = []
    min_silence_length = 70
    intervals = detect_nonsilent(audio_file,
                                 min_silence_len=min_silence_length,
                                 silence_thresh=-32.64
                                 )
    if not intervals:
        return intervals_jsons

    if intervals[0][0] != 0:
        intervals_jsons.append({'start': 0, 'end': intervals[0][0], 'tag': '0000'}) # tag: 0000 means silence
    non_silence_start = intervals[0][0]
    before_silence_start = intervals[0][1]

    for interval in intervals:
        interval_audio = audio_file[interval[0]:interval[1]]

        # 800ms초 이상의 공백 부분 처리
        if (interval[0] - before_silence_start) >= 800:
            intervals_jsons.append({'start': non_silence_start, 'end': before_silence_start + 200, 'tag': '1000'}) # tag: 1000 means non-slience
            non_silence_start = interval[0] - 200
            intervals_jsons.append({'start': before_silence_start, 'end': interval[0], 'tag': '0000'}) # tag: 0000 means slience

        # 추임새인 경우
        if predict_filler(interval_audio) == 0:
            if len(interval_audio) <= 460:
                intervals_jsons.append({'start': non_silence_start, 'end': interval[0], 'tag': '1000'}) # tag: 1000 means non-slience
                non_silence_start = interval[0]
                intervals_jsons.append({'start': interval[0], 'end': interval[1], 'tag': '1111'})
            else:
                non_silence_start = shorter_filler(intervals_jsons, interval_audio, min_silence_length, interval[0],
                                                   non_silence_start)
        before_silence_start = interval[1]

    if non_silence_start != len(audio_file):
        intervals_jsons.append({'start': non_silence_start, 'end': len(audio_file), 'tag': '1000'})

    return intervals_jsons

def STT_with_json(audio_file, jsons):
    global first_silence_interval
    first_silence = 0
    num = 0
    unrecognizable_start = 0
    r = sr.Recognizer()
    transcript_json = []
    statistics_filler_json = []
    statistics_silence_json = []
    stt_score_feedback_json = []

    filler_1 = 0 # 어
    filler_2 = 0 # 음
    filler_3 = 0 # 그
    filler_total_time = 0 # 어음그 총 사용시간

    audio_total_length = audio_file.duration_seconds
    silence_interval = 0

    for json in jsons:
        if json['tag'] == '0000': # 침묵
            # 통역 개시 지연시간
            if num == 0:
                first_silence = first_silence + (json['end'] - json['start']) / 1000
            else:
                silence_interval = silence_interval + (json['end'] - json['start']) / 1000
                silence = "(" + str(round((json['end'] - json['start']) / 1000)) + "초).."
                transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '0000', 'result': silence})

        elif json['tag'] == '1111':
            # 전사 개시 지연시간
            if num == 0:
                silence = "(" + str(round(first_silence)) + "초).."
                transcript_json.append({'start': 0, 'end': json['start'], 'tag': '0000', 'result': silence})
                first_silence_interval = first_silence

            # 추임새 어/음/그 분류
            filler_type = predict_filler_type(audio_file[json['start']:json['end']])
            duration = (json['end'] - json['start']) / 1000 # 초 단위 변환

            if filler_type == 0:
                transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '1001', 'result': '어'})
                filler_1 = filler_1 + 1
            elif filler_type == 1:
                transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '1010', 'result': '음'})
                filler_2 = filler_2 + 1
            else:
                transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '1100', 'result': '그'})
                filler_3 = filler_3 + 1

            filler_total_time += duration # 총 filler word 사용시간 증가
            num = num + 1

        elif json['tag'] == '1000': # 인식 불가 처리
            if unrecognizable_start != 0:
                audio_file[unrecognizable_start:json['end']].export(temp_wav_path, format="wav")
            else:
                audio_file[json['start']:json['end']].export(temp_wav_path, format="wav")
            temp_audio_file = sr.AudioFile(temp_wav_path)
            with temp_audio_file as source:
                audio = r.record(source)
            try:
                stt = r.recognize_google(audio_data=audio, language="ko-KR")
                first_silence_interval = 0

                # 전사 개시 지연시간
                if num == 0:
                    silence = "(" + str(round(first_silence)) + "초).."
                    transcript_json.append({'start': 0, 'end': json['start'], 'tag': '0000', 'result': silence})
                    first_silence_interval = first_silence
                if unrecognizable_start != 0:
                    transcript_json.append(
                        {'start': unrecognizable_start, 'end': json['end'], 'tag': '1000', 'result': stt})
                else:
                    transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '1000', 'result': stt})
                unrecognizable_start = 0
                num = num + 1
            except:
                if unrecognizable_start == 0:
                    unrecognizable_start = json['start']

    filler_ratio = round(100*filler_total_time/(audio_total_length - first_silence_interval - silence_interval), 2)
    stt_filler_score, stt_feedback = calculate_filler_score(filler_ratio)
    fluency_score = stt_filler_score
    # stt_score_feedback_json.append({'발표 유창성 점수' : stt_filler_score,
    #                                 '발표 유창성 피드백' : stt_feedback})

    statistics_filler_json.append({'어': filler_1,
                                   '음': filler_2,
                                    '그': filler_3,
                                   '불필요한 추임새 총 개수': filler_1+filler_2+filler_3,
                                   '발화시간 대비 추임새 비율(%)': filler_ratio})

    statistics_silence_json.append({'침묵비율(%)': round(100 * silence_interval / audio_total_length),
                                    '발화비율(%)': round(100 * (
                                                audio_total_length - first_silence - silence_interval) / audio_total_length),
                                    '전체발표시간(초)': round(audio_total_length)})

    return statistics_filler_json, statistics_silence_json, fluency_score, transcript_json

def calculate_filler_score(filler_ratio):

    if filler_ratio <= 1: # 95점 이상
        score = 100 - filler_ratio*5
        stt_feedback = "아주 훌륭합니다! 발표에서 불필요한 추임새가 거의 사용되지 않았습니다. 자연스럽고 유창한 발표였습니다."
    elif filler_ratio <= 3: # 82점 이상
        score = 100 - filler_ratio*6
        stt_feedback = "발표 중 ‘어, 음, 그’와 같은 불필요한 추임새가 다소 반복적으로 사용되었습니다. " \
                   "발표의 흐름이 살짝 끊기는 느낌이 있었지만, 내용 전달에는 큰 영향을 주지 않았습니다." \
                   "다음 발표에서는 문장을 더 명확하게 구성하는 연습을 해보는 것이 좋겠습니다."
    elif filler_ratio <= 5: # 65점 이상
        score = 100 - filler_ratio*7
        stt_feedback = "발표에서 불필요한 추임새가 자주 등장하여 발표의 명확성이 다소 저하되었습니다. " \
                   "발표를 듣는 동안 청중이 핵심 내용을 파악하는데 방해가 될 가능성이 있습니다." \
                   "발표 전 충분한 연습을 통해 자연스럽게 발표 내용을 연결하는 연습을 해보세요."
    else:
        score = 100 - filler_ratio*8 # 60점 미만
        stt_feedback = "발표에서 불필요한 추임새가 과도하게 사용되었습니다. " \
                   "발표 흐름이 끊기고, 청중의 집중도가 낮아질 가능성이 높습니다." \
                   "발표 내용이 전달되기 어려울 수 있으므로, 명확하고 간결한 문장을 구성하는 연습이 필요합니다." \
                   "짧은 문장으로 말하는 연습을 하거나, 발표 전에 키워드를 미리 정리하여 불필요한 추임새를 줄여보세요."

    stt_filler_score = max(0, score)
    return stt_filler_score, stt_feedback

async def get_prediction(video_data):
    webm_content = convert_video_to_webm(video_data)
    wav_file = convert_webm_to_wav(webm_content)
    audio = AudioSegment.from_wav(io.BytesIO(wav_file))
    intervals_jsons = create_json(audio)
    statistics_filler, statistics_silence, fluency_score, transcript = STT_with_json(audio, intervals_jsons)
    return statistics_filler, statistics_silence, fluency_score, transcript