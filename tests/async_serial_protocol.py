import serial_asyncio
import asyncio

class SerialProtocolHandler(asyncio.Protocol):
    def __init__(self, queue):
        """
        Args:
            queue: asyncio.Queue (메시지를 저장하는 큐, FIFO 방식으로 처리)
            transport: asyncio.Transport (serial 통신을 위한 transport 객체)
        Functions:
            connection_made: serial connection이 문제 없이 처리되면 호출
            data_received: 데이터를 수신받으면 그 즉시 호출
            connection_lost: serial connection이 끊기면 호출
            
        Description:
        SerialProtocolHandler는 asyncio.Protocol을 상속받아,
        connection_made, data_received, connection_lost 등 asyncio.Protocol에 정의된 콜백 함수를 오버라이드하여
        시리얼 연결이 성립되거나, 데이터를 수신하거나, 연결이 끊길 때 자동으로 호출되도록 하는 역할을 수행합니다.
        즉, 이 클래스는 시리얼 포트와의 통신 이벤트에 대해 비동기적으로 반응하는 콜백 로직을 정의하는 역할입니다.
        """
        self.queue = queue
        self.transport = None
        self._byte_buffer = bytearray()

    def connection_made(self, transport):
        self.transport = transport
        print("Serial connection established:", transport)

    def data_received(self, data):

        self._byte_buffer.extend(data)
        print(f"Raw data chunk received: {data}")

        while True:
            # 버퍼에서 STX와 ETX 위치 찾기
            stx_index = self._byte_buffer.find(b'\x02')
            if stx_index == -1:
                # STX가 없으면 완전한 메시지가 없음.
                break
            etx_index = self._byte_buffer.find(b'\x03', stx_index + 1)
            if etx_index == -1:
                # ETX가 아직 도착하지 않음 => 더 대기
                break

            # STX부터 ETX까지 (ETX 포함) 를 완전한 메시지로 간주
            complete_bytes = self._byte_buffer[stx_index:etx_index + 1]
            try:
                # 완전한 메시지를 디코딩 (원하는 경우, control 문자를 남길 수 있음)
                complete_message = complete_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                complete_message = repr(complete_bytes)
            print("Complete message extracted:", complete_message)
            self.queue.put_nowait(complete_message)
            # 처리한 메시지 부분은 버퍼에서 제거
            del self._byte_buffer[:etx_index + 1]
            

    def connection_lost(self, exc):
        print("Serial connection lost:", exc)

class AsyncSerialCommunicator:

    def __init__(self, port, baudrate):
        """
        Args:
            port: serial port 이름
            baudrate: 통신 속도 
            push_button_trigger: "START" 메시지를 감지하면 True로 설정
            queue: asyncio.Queue (메시지를 저장하는 큐, FIFO 방식으로 처리)
            transport: asyncio.Transport (serial 통신을 위한 transport 객체)
            protocol: SerialProtocolHandler (시리얼 통신 이벤트를 처리하는 프로토콜 핸들러)
            
        Functions:
        - monitor_push_button(): "START" 메시지를 감지하면 push_button_trigger를 True로 설정  
        - async_sending_param(): 데이터를 전송한 후, echo 응답(OK_target)을 기다려서 확인
        - connect(): 시리얼 연결을 수립
        - clear_queue(): 큐에 남아있는 모든 메시지를 비움
           
        Description:
            
        AsyncSerialCommunicator 내에 선언된 비동기 함수들(예: connect, clear_queue, monitor_push_button, async_sending_param)은 이벤트 루프에 의해 관리됩니다.
        이벤트 루프는 이 함수들을 작업(task)으로 스케줄링하여, 각각이 실행되어야 할 때 실행되도록 하고, 한 함수가 await로 대기하는 동안 다른 함수들을 실행할 수 있도록 합니다.
        즉, 이벤트 루프는 단일 스레드 내에서 여러 비동기 함수들이 협력적으로 실행되도록 스케줄링하는 역할을 하며, 이를 통해 각 비동기 함수가 필요한 시점에 실행되도록 관리합니다.
          
        스케줄링 방법: asyncio.Queue를 사용하여, 데이터를 저장하고 소비하는 방식으로 비동기 함수 간의 데이터 교환을 수행
            
        """
        self.port = port
        self.baudrate = baudrate
        self.push_button_trigger = False
        self.queue = asyncio.Queue()  # 들어오는 모든 메시지를 저장
        self.transport = None
        self.protocol = None

    async def connect(self):
        loop = asyncio.get_running_loop()
        # transport는 시리얼 연결 자체를 나타내며, 데이터를 송수신하는 역할을 함
        # protocol은 SerialProtocolHandler 인스턴스로, 이벤트 기반 콜백 함수들을 통해 데이터 수신 등 시리얼 통신 이벤트를 처리함
        self.transport, self.protocol = await serial_asyncio.create_serial_connection(
            loop, lambda: SerialProtocolHandler(self.queue), self.port, baudrate=self.baudrate
        )
        
    async def clear_queue(self):
        await asyncio.sleep(0.05)
        while not self.queue.empty():
            await self.queue.get()
            # await self.queue.get_nowait() 이거 적용하면 에러 발생

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

        # 데이터 전송 전, 남은 메시지가 있다면 비우기
        # await asyncio.sleep(0.05)
        await self.clear_queue()

        # 명령 전송
        self.transport.write(byte_array)
        print("[Preprocess] Sent command:", byte_array)

        attempt = 0
        try:
            async with asyncio.timeout(timeout):
                while True:
                    # echo 응답을 기다림 (timeout 내에 메시지 없으면, TimeoutError 발생)
                    message = await self.queue.get()
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
                            thomas_event_state = "serial_error"
                            return thomas_event_state
                        self.transport.write(byte_array)
                        
        except asyncio.TimeoutError:
            print("[Preprocess] Timeout waiting for echo response.")
            thomas_event_state = "serial_error"
            return thomas_event_state
        
        thomas_event_state = "serial_error"
        return thomas_event_state
                    

    def close(self):
        if self.transport:
            self.transport.close()