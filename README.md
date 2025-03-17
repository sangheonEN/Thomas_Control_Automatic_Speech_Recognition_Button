# github에서는 오직 source code의 history를 저장하기 위함. 그래서 수정된 코드만 업로드하여 이력을 관리할것임. D:\STT_V1\STT\RealtimeSTT_Button\tests\ 경로에 faster_whisper_model, silero_model 폴더 및 내부 파일을 넣어야 정상적으로 코드가 작동됩니다. 필요하시면 jteks6@gmail.com으로 연락주세요.

# VERSION 설명
V1.0 : 통신 기능 동기식 코드

V1.1 : 통신 기능의 출력 버퍼 초기화 적용, utils.main_process 함수 thread화 코드 적용 (20250312_BUTTON식음성인식모듈코드변경 사항.txt 참고)

V1.2 : 토마스 이벤트 송수신 상태 체크 변수 적용 및 Main Process에서 shutdown 함수 적용 전환으로 올바른 종료 코드 적용 (20250313_BUTTON식음성인식모듈코드변경사항.txt 참고)
   -> utils.main_process 함수내에서 본인의 함수를 처리하는 thread를 join하면 계속 기다리니, 이걸 main process내에서 종료하도록 변경함. 
   
V1.3 : 통신 기능 비동기식 적용 예정

# py source description.

1. Thomas_audio_control_src.py : Main code. Thomas Connection + Mic Connection + RealTimeSTT + Thomas Sending the Event parameters
   - Button을 활용해서 push 하면 realtimestt 모듈이 wake up 되도록 구현함.

2. serial_protocol.py : Serial 통신을 위한 파라미터를 저장하고, 이벤트 기능을 통해 전달 받은 event_flag 변수를 sending하는 클래스를 포함하는 src

3. text_similarity.py : Senario reference와 Prediction text 간의 유사도를 계산하는 변수와 함수가 구현된 클래스를 포함하는 src.

4. utils.py : 기타 처리 기능들이 포함되는 src.

5. params.py : 시나리오에 reference text 및 전역 parameters가 포함되는 src.