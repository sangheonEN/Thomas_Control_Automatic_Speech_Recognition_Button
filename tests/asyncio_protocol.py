import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import serial_asyncio
import time
import torch.multiprocessing as mp
import serial

import utils
import params
from RealTimeSTT_LEE.audio_recorder import AudioToTextRecorder
from text_similarity import Similarity_cal


class SerialProtocolHandler(asyncio.Protocol):
    def __init__(self, queue):
        self.queue = queue
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        print("Serial connection established:", transport)

    def data_received(self, data):
        try:
            message = data.decode("utf-8")
        except Exception as e:
            message = repr(data)
        print("Data received:", message)
        # 데이터는 큐에 저장하여 소비자가 처리할 수 있도록 함
        self.queue.put_nowait(message)

    def connection_lost(self, exc):
        print("Serial connection lost:", exc)

class AsyncSerialCommunicator:
    """
    비동기 serial communicator  
    1. monitor_push_button(): "START" 메시지를 감지하면 push_button_trigger를 True로 설정  
    2. async_sending_param(): 데이터를 전송한 후, echo 응답(OK_target)을 기다려서 확인  
    """
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.push_button_trigger = False
        self.queue = asyncio.Queue()  # 들어오는 모든 메시지를 저장
        self.transport = None
        self.protocol = None

    async def connect(self):
        loop = asyncio.get_running_loop()
        self.transport, self.protocol = await serial_asyncio.create_serial_connection(
            loop, lambda: SerialProtocolHandler(self.queue), self.port, baudrate=self.baudrate
        )

    async def monitor_push_button(self):
        """
        "START" 메시지가 들어올 때까지 큐에서 메시지를 읽어오고,
        감지되면 push_button_trigger를 True로 설정하고 종료.
        """
        print("Waiting for 'START' signal...")
            
        while True:
            message = await self.queue.get()
            print("[Monitor] Message from queue:", message)
            if "START" in message:
                self.push_button_trigger = True
                print("[Monitor] 'START' detected!")
                break

    async def async_sending_param(self, tens, units, thomas_event_state, timeout=4.0, retries=3):
        """
        전송할 데이터를 보낸 후, echo 응답(OK_target)이 올 때까지 큐에서 메시지를 기다립니다.
        OK_target이 확인되면 정상 상태(thomas_event_state)를 반환하고, 그렇지 않으면 "serial_error"를 반환합니다.
        """
        OK_target = f'\x02{tens}{units}OK\x03'
        # 전송할 데이터 생성 (예: [STX, tens, units, ETX])
        val = [2, ord(str(tens)), ord(str(units)), 3]
        byte_array = bytearray(val)

        # 전송 전, 남은 메시지가 있다면 비우기
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # 명령 전송
        self.transport.write(byte_array)
        print("[Preprocess] Sent command:", byte_array)

        start_time = time.time()
        attempt = 0
        while time.time() - start_time < timeout:
            try:
                # echo 응답을 기다림 (timeout 내에 메시지가 없으면 TimeoutError 발생)
                message = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                print("[Preprocess] Received echo:", message)
                if OK_target in message:
                    print("[Preprocess] OK_target matched!")
                    return thomas_event_state
                elif "START" in message:
                    print("[Preprocess] 'START' message detected. Ignoring...")
                    continue
                else:
                    print("[Preprocess] Echo mismatch. Retrying command...")
                    attempt += 1
                    if attempt >= retries:
                        return "serial_error"
                    self.transport.write(byte_array)
            except asyncio.TimeoutError:
                print("[Preprocess] Timeout waiting for echo response.")
                return "serial_error"
        return "serial_error"

    def close(self):
        if self.transport:
            self.transport.close()

async def main():
    
    mp.freeze_support()
    
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
                """
                2025/03/20 여기서 recorder.text() 처리하고 이벤트 번호 산정해서 아래에 async_sending_param로 넘겨주자
                1. recorder STT 전사 처리 inf_text return
                2. event_matching 함수로 event_flag 산정 
                """
                inf_text = recorder.text()
                event_flag, max_similarity = utils.event_matching(inf_text, similarity_cal, params.similarity_config)
                print(f"event_flag:{event_flag}\n")
                print(f"inf_text: {inf_text}\n")
                print(f"max_similarity: {max_similarity}\n")
                
                if event_flag == None:
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
    asyncio.run(main())
