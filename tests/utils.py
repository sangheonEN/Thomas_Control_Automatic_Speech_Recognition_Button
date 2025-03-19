import os
import editdistance
import pyaudio
import sys
from pydub import AudioSegment
from pydub.playback import play
import time
import serial

import params
from serial_protocol import Serial_protocol


def main_process(inf_text, start_time, communicator, similarity_cal, similarity_config, thomas_event_state):
    """
        inf_text를 calculate_event_flag에 전달하여 최종 event_flag를 전달받고 communicator를 활용해 토마스에 event_flag를 sending!
        
        Args:
            inf_text: STT 추론 TEXT
            start_time: main process 실행 시작 시간
            communicator: utils.Serial_protocol(**communicator_config)

        Return: N/A

    """

    similarity_function = similarity_config["function"]
    threshold = similarity_config["threshold"]

    if similarity_function == "gestalt_pattern_matching":
        event_flag, max_similarity, _ = similarity_cal.gestalt_pattern_matching(inf_text, threshold)
    else:
        event_flag, max_similarity, _ = similarity_cal.sentence_transformers(inf_text, threshold)

    print(f"similarity : {max_similarity}\n")
    print(f"event_flag : {event_flag}\n")

    if event_flag == None:
        return thomas_event_state
    
    else:
        # Serial Protocol
        thomas_event_state = communicator.sending_param(event_flag, thomas_event_state, inf_text)
        # print(f"received data : {communicator.received_param()}\n")
        end_time = time.time()
        process_time = end_time - start_time

        print(f"Processing Time : {process_time}\n")
        
        return thomas_event_state


def calculate_cer(reference, hypothesis):
    """
        cer 계산 함수
        Args:
            reference: reference text
            hypothesis: prediction text
        Return: cer

    """
    distance = editdistance.eval(reference, hypothesis)
    cer = distance / len(reference) if len(reference) > 0 else 0
    return cer


def check_mic_connection():
    """
    mic check 입력 파라미터 loading code 참고: https://github.com/WindyYam/gemini_voice_companion/blob/main/scripts/voice_recognition.py#L5
    
    이 코드는 사용자가 지정한 오디오 장치 이름(device_name)이 있을 경우, 시스템에 연결된 모든 오디오 장치를 순회하면서 입력 기능(마이크 등)을 제공하는 장치들 중에서 이름에 device_name이 포함된 장치를 찾는 역할을 합니다. 

        구체적으로:

        1. 장치 유효성 확인: 먼저 device_name이 존재하는지 확인합니다.
        2. 장치 반복: 0부터 numdevices까지 모든 오디오 장치를 반복문으로 확인합니다.
        3. 입력 장치 필터링: 각 장치의 maxInputChannels 값이 0보다 큰지 확인하여 입력이 가능한 장치인지 판단합니다.
        4. 이름 매칭: 해당 장치의 이름에 device_name 문자열이 포함되어 있는지 검사합니다.
        5. 인덱스 할당: 조건에 맞는 장치를 찾으면, 그 장치의 인덱스를 device_index에 저장합니다.

        즉, 이 코드는 원하는 이름을 가진 입력 장치를 자동으로 선택하기 위해 사용됩니다.

    return: True if a microphone is found, False otherwise.
    """

    try:

        audio = pyaudio.PyAudio()
        
        info = audio.get_default_host_api_info()
        numdevices = info.get('deviceCount')
        device_index = None
        device_name = params.recorder_config["device_name"]

        if device_name:
            for i in range(0, numdevices):
                if audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                    if device_name in audio.get_device_info_by_host_api_device_index(0, i).get('name'):
                        device_index = i
        if device_index:
            print('Setting Recorder: ', audio.get_device_info_by_host_api_device_index(
                0, device_index).get('name'))
        else:
            print('Setting Recorder: ',
                audio.get_default_input_device_info().get('name'))

        params.recorder_config['input_device_index'] = device_index
        print(f"params.recorder_config['input_device_index']: {params.recorder_config['input_device_index']}")

        # Check if a valid input device index is found
        if device_index is not None:
            print(f"Microphone '{device_name}' is connected.")
            return True

    except OSError:
        # Handle the error if no default input device is available
        print("No default input mic device available.")
        sys.exit(1)


def check_communicator(communicator_config):
    try:
        communicator = Serial_protocol(**communicator_config)
        if communicator.ser.is_open:
            print("Serial connection established.")
            return communicator
    except serial.SerialException as e:
        # Handle the error if no default input device is available
        print(f"No connected thomas device available. Error : {e}")
        raise RuntimeError(f"No connected thomas device available. Error: {e}") from e
        
    """ 사용 가능한 포트 확인 debug
        # import serial.tools.list_ports
        # ports = serial.tools.list_ports.comports()
        # available = [port.device for port in ports]
        # print("현재 사용 가능한 포트:", available)
    """