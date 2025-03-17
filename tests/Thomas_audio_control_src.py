"""
1. inf text similarity calculation using levenshtein_similarity
2. event flag calculation using max similarity score
3. Developing functions tailored to event flags (run voice file)
- Future work of robot control communication technology is necessary

window 기준
_transcription_worker -> torch.multiprocess
_audio_data_worker -> torch.multiprocess
_recording_worker -> threading.Thread

1. AudioToTextRecorder 객체 생성되자마자 아래 두 audio data worker는 계속 실행

_audio_data_worker data=stream.read(buffer_size)

_audio_data_worker audio_queue.put(processed_data)

2. 녹화 시작 flag가 True 되었을때,

_recording_worker self.frames.append(data)

3. 음성 감지 멈췄을때,

wait_audio self.audio=audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

transcribe self.parent_transcription_pipe.send((self.audio, self.language))
_audio_data_worker data=stream.read(buffer_size)

-----
2024/10/08 추가해야하는 기능
1) 마이크 음성 크기 줄이기. -> WINDOW 마이크 볼륨 기능
2) 정확도 평가를 위한 코드 적용 -> 말한 뒤에 제대로 EVENT FLAG를 출력하는지 COUNT
3) text() 호출시 콜백함수로 제어신호 event_function 전달 코드작성.
2024/10/08 3)까지 완료

2024/10/28 추가한 기능
1) Serial 통신을 활용한 int 형식 데이터 송수신 기능 추가
2024/10/28 완료

2024/11/06 추가한 기능
1) Serial 연결 끊김 시 재 연결 코드 추가
2024/11/06 완료

2024/11/07 추가한 기능
1) Faster-whisper, silero model local pc path로 설정 
2024/11/07 완료

개발해야하는 기능
4) 이벤트 FLAG로 사용하는 정답 TEXT가 추가 될 수 있으니, event_flag 번호를 얻는 방법을 index로 말고 다른 걸로 하는 방법 찾기.
5) wake word 모델이 활성화된 후 토마스까지 text 추론 되는 문제 해결.
6) wake word 를 가지고 지금 처럼 wake word 말하고 한 문장 말하고 하는게 아니고 그냥 음성인식을 모듈을 활성화할지 안할지를 정하는 트리거로 사용할지 판단.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch.multiprocessing as mp

import params
from RealTimeSTT_LEE.audio_recorder import AudioToTextRecorder
import utils
from text_similarity import Similarity_cal

if __name__ == '__main__':
    
    mp.freeze_support()
    
    check_mic_connection = utils.check_mic_connection()
    communicator = utils.check_communicator(params.communicator_config)
    recorder = AudioToTextRecorder(**params.recorder_config)
    similarity_cal = Similarity_cal(params.similarity_config['model_path'])
    
    print("Say something...")

    try:
            
        print("잘 실행됨?\n")

        while (True):
            
            # 버튼식 Wake Up Trigger!
            while (True):
                communicator.check_push_button()
                time.sleep(0.08) # 이거 안하면, check push button을 위한 received data가 순서대로 안들어옴. 레이턴시 맞추기 debug -> print(f"data1: {data1}\n"), print(f"data2: {data2}\n")
                
                if communicator.push_button_trigger:
                    # 다시 push_button_trigger False로 초기화
                    communicator.push_button_trigger = False
                    break
                else:
                    continue
            # print("push button on!")
            start_time = time.time()
            recorder.text(utils.main_process, start_time, communicator, similarity_cal, params.similarity_config, recorder)
            time.sleep(0.08) # 이거 안하면 sending data cross check를 위한 received data가 순서대로 안들어옴. 레이턴시 맞추기 debug -> print(f"Received data1:{data}")

            
    except KeyboardInterrupt:
        communicator.close()
        print("Serial protocol terminated due to KeyboardInterrupt\n")