# github에서는 오직 source code의 history를 저장하기 위함. 그래서 수정된 코드만 업로드하여 이력을 관리할것임. D:\STT_V1\STT\RealtimeSTT_Button\tests\ 경로에 faster_whisper_model, silero_model 폴더 및 내부 파일을 넣어야 정상적으로 코드가 작동됩니다. 필요하시면 jteks6@gmail.com으로 연락주세요.

# VERSION 설명
V1.0 : 통신 기능 동기식 코드

V1.1 : 통신 기능의 출력 버퍼 초기화 적용, utils.main_process 함수 thread화 코드 적용 (20250312_BUTTON식음성인식모듈코드변경 사항.txt 참고)

V1.2 : 토마스 이벤트 송수신 상태 체크 변수 적용 및 Main Process에서 shutdown 함수 적용 전환으로 올바른 종료 코드 적용 (20250313_BUTTON식음성인식모듈코드변경사항.txt 참고)
   -> utils.main_process 함수내에서 본인의 함수를 처리하는 thread를 join하면 계속 기다리니, 이걸 main process내에서 종료하도록 변경함. 
   
V1.3 : 통신 기능 비동기식 적용

- 비동기적 시리얼 통신 reading 기능을 적용하여 I/O 작업을 개선했다.

- 기존에 문제점은 동기적으로 데이터 통신하여 지속적으로 들어오는 동안 읽기 처리가 늦어져서 입력 버퍼가 넘치고 데이터가 버려지는 현상("버퍼 오버플로우" 또는 "데이터 드랍”)으로 데이터가 유실되어 정상적인 데이터를 송신 받지 못하여 처리가 안됨.

- 구현 내용
    - https://github.com/pyserial/pyserial-asyncio를 활용해서 pyserial-asyncio ver(0.6) 코드 구현
- 시행 착오 내용
    1. https://github.com/Lei-k/async-pyserial를 활용해서 Reading하는 기능을 개발했지만, “START” 대신 “b'STP*\xf5’”가 읽어지는 문제가 발생하여 해당 깃은 제외
    2. 버튼을 꾹 눌러서 “START”가 계속 들어오면 “OK 시그널” 체크에서 “START”가 reading되어 serial error가 트리거 되는 버그 발생. → 예외 처리 하여 개선

V1.4 : 전체 코드 개선 및 버그 수정

0. 변경 사항 : 기존의 asyncio_protocol.py 코드를 main.py(메인 코드), async_serial_protocol.py(시리얼 클래스 정의)로 분리

1. 변경 사항 : 여러번 버튼 눌렀을 때 "START"가 쌓이는 문제 해결

V1.5 : text_similarity.py 코드 개선
개선 내용 : dict에 순서 상관 없이 1, 2, 3, 4가 아닌 3, 1, 2, 4로 시나리오 데이터를 저장했을때, 처리하기 위해서는 index를 가지고 best_match text를 특정하여 event_flag = params.event_flag[best_match] 이렇게 value 값을 key값을 활용해 추출함.

V1.6 : 시리얼 통신 수신 데이터 끊겼을때, STX, ETX 활용해서 정확한 수신 데이터를 얻는 코드로 개선. 기존에는 STA, RT 이런식으로 끊겨서 들어오면 동작 안됨. STX인식해서 들어온 값을 _byte_buffer 버퍼에 저장해두고 ETX인식해서 STX, ETX 사이에 들어온 값을 합치고 최종 큐에 put -> self.queue.put_nowait(complete_message)

# py source description.

1. Thomas_audio_control_src.py : Main code. Thomas Connection + Mic Connection + RealTimeSTT + Thomas Sending the Event parameters
   - Button을 활용해서 push 하면 realtimestt 모듈이 wake up 되도록 구현함.

2. serial_protocol.py : Serial 통신을 위한 파라미터를 저장하고, 이벤트 기능을 통해 전달 받은 event_flag 변수를 sending하는 클래스를 포함하는 src

3. text_similarity.py : Senario reference와 Prediction text 간의 유사도를 계산하는 변수와 함수가 구현된 클래스를 포함하는 src.

4. utils.py : 기타 처리 기능들이 포함되는 src.

5. params.py : 시나리오에 reference text 및 전역 parameters가 포함되는 src.
