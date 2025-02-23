import tensorflow as tf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa
import numpy as np
import io
import subprocess
import os
from pydantic import BaseModel
import speech_recognition as sr

# 모델 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
classifier_model_path = os.path.join(current_dir, '..', 'models', 'filter_classifier_model.h5')
determine_model_path = os.path.join(current_dir, '..', 'models', 'filter_determine_model.h5')

# 모델 불러오기
filler_classifier_model = tf.keras.models.load_model(classifier_model_path)
filter_determine_model = tf.keras.models.load_model(determine_model_path)

# 전역 변수 선언
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

frame_length = 0.025
frame_stride = 0.0010


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
    audio_file.export("temp.wav", format="wav")
    wav, sr = librosa.load("temp.wav", sr=16000)
    mfcc = librosa.feature.mfcc(y=wav)
    padded_mfcc = pad2d(mfcc, 40)
    padded_mfcc = np.expand_dims(padded_mfcc, 0)

    result = filler_classifier_model.predict(padded_mfcc)
    if result[0][0] >= result[0][1]:
        return 0
    else:
        return 1


def predict_filler_type(audio_file):
    audio_file.export("temp.wav", format="wav")

    wav, sr = librosa.load("temp.wav", sr=16000)
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))

    mfcc = librosa.feature.mfcc(y=wav)
    padded_mfcc = pad2d(mfcc, 40)
    padded_mfcc = np.expand_dims(padded_mfcc, 0)
    result = filler_classifier_model.predict(padded_mfcc)

    os.remove("temp.wav")
    return np.argmax(result)


def shorter_filler(json_result, audio_file, min_silence_len, start_time, non_silence_start):
    min_silence_length = (int)(min_silence_len / 1.2)

    intervals = detect_nonsilent(audio_file,
                                 min_silence_len=min_silence_length,
                                 silence_thresh=-32.64
                                 )
    for interval in intervals:
        interval_audio = audio_file[interval[0]:interval[1]]
        if (interval[1] - interval[0] >= 460):
            non_silence_start = shorter_filler(json_result, interval_audio, min_silence_length,
                                               interval[0] + start_time, non_silence_start)
        else:
            if predict_filler(interval_audio) == 0:
                json_result.append({'start': non_silence_start, 'end': start_time + interval[0], 'tag': '1000'})
                non_silence_start = start_time + interval[0]
                json_result.append({'start': start_time + interval[0], 'end': start_time + interval[1], 'tag': '1111'})
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
        intervals_jsons.append({'start': 0, 'end': intervals[0][0], 'tag': '0000'})
    non_silence_start = intervals[0][0]
    before_silence_start = intervals[0][1]

    for interval in intervals:
        interval_audio = audio_file[interval[0]:interval[1]]

        if (interval[0] - before_silence_start) >= 800:
            intervals_jsons.append({'start': non_silence_start, 'end': before_silence_start + 200, 'tag': '1000'})
            non_silence_start = interval[0] - 200
            intervals_jsons.append({'start': before_silence_start, 'end': interval[0], 'tag': '0000'})

        if predict_filler(interval_audio) == 0:
            if len(interval_audio) <= 460:
                intervals_jsons.append({'start': non_silence_start, 'end': interval[0], 'tag': '1000'})
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
    first_silence = 0
    num = 0
    unrecognizable_start = 0
    r = sr.Recognizer()
    transcript_json = []
    statistics_filler_json = []
    statistics_silence_json = []
    filler_1 = 0
    filler_2 = 0
    filler_3 = 0
    audio_total_length = audio_file.duration_seconds
    silence_interval = 0
    for json in jsons:
        if json['tag'] == '0000':
            if num == 0:
                first_silence = first_silence + (json['end'] - json['start']) / 1000
            else:
                silence_interval = silence_interval + (json['end'] - json['start']) / 1000
                silence = "(" + str(round((json['end'] - json['start']) / 1000)) + "초).."
                transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '0000', 'result': silence})
        elif json['tag'] == '1111':
            if num == 0:
                silence = "(" + str(round(first_silence)) + "초).."
                transcript_json.append({'start': 0, 'end': json['start'], 'tag': '0000', 'result': silence})
                first_silence_interval = first_silence
            filler_type = predict_filler_type(audio_file[json['start']:json['end']])
            if filler_type == 0:
                transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '1001', 'result': '어(추임새)'})
                filler_1 = filler_1 + 1
            elif filler_type == 1:
                transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '1010', 'result': '음(추임새)'})
                filler_2 = filler_2 + 1
            else:
                transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '1100', 'result': '그(추임새)'})
                filler_3 = filler_3 + 1
            num = num + 1
        elif json['tag'] == '1000':
            if unrecognizable_start != 0:
                audio_file[unrecognizable_start:json['end']].export("temp.wav", format="wav")
            else:
                audio_file[json['start']:json['end']].export("temp.wav", format="wav")
            temp_audio_file = sr.AudioFile('temp.wav')
            with temp_audio_file as source:
                audio = r.record(source)
            try:
                stt = r.recognize_google(audio_data=audio, language="ko-KR")
                first_silence_interval = 0
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

    statistics_filler_json.append({'어': filler_1, '음': filler_2, '그': filler_3})
    statistics_silence_json.append({'mumble': round(100 * first_silence_interval / audio_total_length),
                                    'silent': round(100 * silence_interval / audio_total_length),
                                    'talk': round(100 * (
                                                audio_total_length - first_silence - silence_interval) / audio_total_length),
                                    'time': round(audio_total_length)})

    transcript = [item['result'] for item in transcript_json]
    filtered_transcript = [value for value in transcript if ('(추임새)' not in value) and ('..' not in value)]

    filtered_transcript = ' '.join(filtered_transcript)
    return filtered_transcript, statistics_silence_json


async def get_prediction(audio_content):
    wav_file = convert_webm_to_wav(audio_content)
    audio = AudioSegment.from_wav(io.BytesIO(wav_file))
    intervals_jsons = create_json(audio)
    transcript, statistics = STT_with_json(audio, intervals_jsons)
    return transcript, statistics

# FastAPI endpoint
class ResponseModel(BaseModel):
    interviewQuestionId: str
    mumble: int
    silent: int
    talk: int
    time: int
    text: str