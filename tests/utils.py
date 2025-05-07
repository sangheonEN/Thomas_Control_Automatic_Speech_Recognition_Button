import pyaudio
import sys


import params


def event_matching(inf_text, similarity_cal, similarity_config):
    
    similarity_function = similarity_config["function"]
    threshold = similarity_config["threshold"]

    if similarity_function == "gestalt_pattern_matching":
        event_flag, max_similarity, _ = similarity_cal.gestalt_pattern_matching(inf_text, threshold)
    else:
        event_flag, max_similarity, _ = similarity_cal.sentence_transformers(inf_text, threshold)
        
    return event_flag, max_similarity


def list_input_devices():
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    input_devices = []

    for i in range(num_devices):
        device_info = audio.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            input_devices.append((i, device_info.get('name')))
    
    audio.terminate()
    return input_devices  # List of (index, name)


def check_mic_connection():
    """
    사용자 설정에 저장된 input_device_index가 유효한 마이크 장치인지 확인하는 함수.
    UI에서 선택된 마이크 index만 확인하면 되므로 device_name 조건은 제거됨.
    """
    result = None
    try:
        audio = pyaudio.PyAudio()
        index = params.recorder_config.get("input_device_index", None)

        if index is None:
            result = "입력 장치 index가 설정되지 않았습니다."
            return False, result

        device_info = audio.get_device_info_by_index(index)

        if device_info.get('maxInputChannels', 0) > 0:
            result = f"선택된 마이크: [{index}] {device_info.get('name')}"
            return True, result
        else:
            result = f"선택된 장치는 입력 장치가 아닙니다: [{index}] {device_info.get('name')}"
            return False, result

    except Exception as e:
        print(f"❌ 마이크 확인 중 오류 발생: {e}")
        result = f"마이크 확인 중 오류 발생: {e}"
        return False, result

# def check_mic_connection():
#     """
#     mic check 입력 파라미터 loading code 참고: https://github.com/WindyYam/gemini_voice_companion/blob/main/scripts/voice_recognition.py#L5
    
#     이 코드는 사용자가 지정한 오디오 장치 이름(device_name)이 있을 경우, 시스템에 연결된 모든 오디오 장치를 순회하면서 입력 기능(마이크 등)을 제공하는 장치들 중에서 이름에 device_name이 포함된 장치를 찾는 역할을 합니다. 

#         구체적으로:

#         1. 장치 유효성 확인: 먼저 device_name이 존재하는지 확인합니다.
#         2. 장치 반복: 0부터 numdevices까지 모든 오디오 장치를 반복문으로 확인합니다.
#         3. 입력 장치 필터링: 각 장치의 maxInputChannels 값이 0보다 큰지 확인하여 입력이 가능한 장치인지 판단합니다.
#         4. 이름 매칭: 해당 장치의 이름에 device_name 문자열이 포함되어 있는지 검사합니다.
#         5. 인덱스 할당: 조건에 맞는 장치를 찾으면, 그 장치의 인덱스를 device_index에 저장합니다.

#         즉, 이 코드는 원하는 이름을 가진 입력 장치를 자동으로 선택하기 위해 사용됩니다.

#     return: True if a microphone is found, False otherwise.
#     """

#     try:

#         audio = pyaudio.PyAudio()
        
#         info = audio.get_default_host_api_info()
#         numdevices = info.get('deviceCount')
#         device_index = None
#         device_name = params.recorder_config["device_name"]

#         if device_name:
#             for i in range(0, numdevices):
#                 if audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
#                     if device_name in audio.get_device_info_by_host_api_device_index(0, i).get('name'):
#                         device_index = i
#         if device_index:
#             print('Setting Recorder: ', audio.get_device_info_by_host_api_device_index(
#                 0, device_index).get('name'))
#         else:
#             print('Setting Recorder: ',
#                 audio.get_default_input_device_info().get('name'))

#         params.recorder_config['input_device_index'] = device_index
#         print(f"params.recorder_config['input_device_index']: {params.recorder_config['input_device_index']}")

#         # Check if a valid input device index is found
#         if device_index is not None:
#             print(f"Microphone '{device_name}' is connected.")

#     except OSError:
#         # Handle the error if no default input device is available
#         print("No default input mic device available.")
#         sys.exit(1)