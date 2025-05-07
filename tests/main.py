import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import time
import torch.multiprocessing as mp
import serial

import utils
import params
from RealTimeSTT_LEE.audio_recorder import AudioToTextRecorder
from text_similarity import Similarity_cal
from async_serial_protocol import AsyncSerialCommunicator


async def main():
        
    utils.check_mic_connection()
    communicator = AsyncSerialCommunicator(params.communicator_config["port"], params.communicator_config["baudrate"])
    await communicator.connect()
    recorder = AudioToTextRecorder(**params.recorder_config)
    similarity_cal = Similarity_cal(params.similarity_config['model_path'])
    
    print("Say something...")
    
    # 메인 루프: 반복적으로 push button("START") 감지 후 전송/응답 체크
    try:
        
        while True:
            # 1. "START" 메시지를 비동기적으로 모니터링
            await communicator.monitor_push_button()
            if communicator.push_button_trigger:
                # push_button_trigger를 감지하면 이를 초기화하고 다음 단계 진행
                communicator.push_button_trigger = False
                # 2. 녹음 또는 텍스트 처리 작업 전에 시간 기록
                start_time = time.time()  # 실제 recorder.text() 호출 시 전달되는 start_time 예시

                # inf_text = recorder.text()
                inf_text = await asyncio.to_thread(recorder.text)
                event_flag, max_similarity = utils.event_matching(inf_text, similarity_cal, params.similarity_config)
                print(f"event_flag:{event_flag}\n")
                print(f"inf_text: {inf_text}\n")
                print(f"max_similarity: {max_similarity}\n")
                
                if event_flag == None:
                    # 이벤트 루프 끝나기 전, 남은 메시지가 있다면 비우기
                    # await asyncio.sleep(0.05)
                    await communicator.clear_queue()
                    continue
                
                units = event_flag % 10  # 나머지
                tens = event_flag // 10  # 몫
                print(f"tens: {tens}, units: {units}\n")

                # 3. event_flag를 예로 01 (tens=0, units=1)이라고 가정하고, 전송 및 echo 응답 체크
                thomas_event_state = await communicator.async_sending_param(tens, units, thomas_event_state="ok")
                print("Final thomas_event_state:", thomas_event_state)
                # 4. 에러 발생 시 종료하거나, 아니면 다음 사이클 반복
                if thomas_event_state == "serial_error":
                    print("Serial communication error detected. Exiting loop...")
                    communicator.close()
                    recorder.shutdown()
                    break
                end_time = time.time()
                print(f"Processing Time : {end_time - start_time}\n")
                # 사이클 간 잠시 대기
                await asyncio.sleep(0.01)
                
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        communicator.close()
        recorder.shutdown()
    except KeyboardInterrupt:
        print("Program terminated due to KeyboardInterrupt\n")
        communicator.close()
        recorder.shutdown()
    except Exception as e:
        print(f"An error occurred: {e}")
        communicator.close()
        recorder.shutdown()

if __name__ == "__main__":
    mp.freeze_support()
    asyncio.run(main())
