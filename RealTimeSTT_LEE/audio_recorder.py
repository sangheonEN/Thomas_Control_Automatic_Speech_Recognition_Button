"""

The AudioToTextRecorder class in the provided code facilitates
fast speech-to-text transcription.

The class employs the faster_whisper library to transcribe the recorded audio
into text using machine learning models, which can be run either on a GPU or
CPU. Voice activity detection (VAD) is built in, meaning the software can
automatically start or stop recording based on the presence or absence of
speech. It integrates wake word detection through the pvporcupine library,
allowing the software to initiate recording when a specific word or phrase
is spoken. The system provides real-time feedback and can be further
customized.

Features:
- Voice Activity Detection: Automatically starts/stops recording when speech
  is detected or when speech ends.
- Wake Word Detection: Starts recording when a specified wake word (or words)
  is detected.
- Event Callbacks: Customizable callbacks for when recording starts
  or finishes.
- Fast Transcription: Returns the transcribed text from the audio as fast
  as possible.

Author: Kolja Beigel

"""

from typing import Iterable, List, Optional, Union
import torch.multiprocessing as mp
import torch
from typing import List, Union
from ctypes import c_bool
from openwakeword.model import Model
from scipy.signal import resample
from scipy import signal
import signal as system_signal
import faster_whisper
import openwakeword
import collections
import numpy as np
import pvporcupine
import traceback
import threading
import webrtcvad
import itertools
import platform
import pyaudio
import logging
import struct
import halo
import time
import copy
import os
import re
import gc
import noisereduce as nr
import queue
# from pyannote.audio import Pipeline
# from pyannote.core import Segment


# demucs는 실패 입력 데이터 shape이 안맞아서 딥러닝 모델이 안돌아감.
# from demucs.pretrained import get_model
# from demucs.apply import apply_model

# demucs_model = get_model(name="htdemucs")
# device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# demucs_model.to(device_test)

# Set OpenMP runtime duplicate library handling to OK (Use only for development!)
# 이 환경 변수는 OpenMP 런타임 라이브러리의 여러 충돌 버전이 로드될 때 발생하는 오류를 방지하기 위해 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_MODEL_TRANSCRIPTION_REALTIME = "tiny"
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.2 # 값이 높을수록 침묵을 감지하고 더 빨리 녹음을 멈춤 / 0.0 (less sensitive) to 1.0 (more sensitive)
INIT_WEBRTC_SENSITIVITY = 3 # 값이 낮을수록 음성인식을 하는데 소리가 커야함 / 0 (very low sensitivity) to 3 (high sensitivity)
# 음성 후 무음 기간을 줄여서 짧은 시간 동안 무음을 인식하고 종료하도록!
# 주위의 사람들의 웅성대는 소리 때문에 음성 녹화가 만연하게 이루어짐.
# INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_POST_SPEECH_SILENCE_DURATION = 0.2 # 음성 처리 후 무음 기간
INIT_MIN_LENGTH_OF_RECORDING = 0.5 # recording이 self.min_length_of_recording초 동안 지속되었는지 확인, self.min_length_of_recording 보다 작으면 멈추지 않음. 
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0 # 녹음 사이에 충분한 시간이 지났는지 확인하는 역할
INIT_WAKE_WORDS_SENSITIVITY = 0.6 # wake words 민감도
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0 # 녹음이 공식적으로 시작되기 전에 최대 self.pre_recording_buffer_duration초 동안 오디오 데이터를 저장할 수 있는 버퍼(audio_buffer)가 생성
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0 
INIT_WAKE_WORD_TIMEOUT = 5.0
# INIT_WAKE_WORD_BUFFER_DURATION = 0.1
INIT_WAKE_WORD_BUFFER_DURATION = 2.0
ALLOWED_LATENCY_LIMIT = 10 # self.audio.queue prevent buffer overflow : ALLOWED_LATENCY_LIMIT은 시스템이 가장 오래된 오디오 청크를 삭제하기 시작하기 전에 대기열에 남아 있을 수 있는 최대 오디오 청크 수를 설정. 

TIME_SLEEP = 0.02
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0 # 오디오 형식이 paInt16이므로 이러한 16비트 정수는 -32768에서 32767까지 범위

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin': # macOS 아닌 경우 True
    INIT_HANDLE_BUFFER_OVERFLOW = True


class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting
    voice activity, and then transcribing the captured audio using the
    `faster_whisper` model.
    """

    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 device_name: str = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cuda",
                 on_recording_start=None, # 초기화 중에 on_recording_start 콜백이 제공된 경우 이 콜백 함수에 대입하면 사용할 수 있습니다. 이를 통해 녹음이 시작될 때 UI 업데이트나 로깅과 같은 외부 작업을 트리거할 수 있습니다.
                 on_recording_stop=None, # 초기화 중에 on_recording_stop 콜백이 제공된 경우 이 콜백 함수에 대입하면 사용할 수 있습니다.
                 on_transcription_start=None, # 초기화 중에 on_transcription_start 콜백이 제공된 경우 이 콜백 함수에 대입하면 사용할 수 있습니다.
                 ensure_sentence_starting_uppercase=True, # 앞 글자 대문자 적용
                 ensure_sentence_ends_with_period=False, # 마침표 적용
                 use_microphone=True, 
                 spinner=True,
                 level=logging.WARNING,

                 # _pyannote function enable flag
                #  pyannote_flag = False,
                 # reduce db enable flag
                 reduce_db_flag = False,
                 # reduce_noise enable flag
                 reduce_noise_flag = True,

                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 realtime_model_type=INIT_MODEL_TRANSCRIPTION_REALTIME,
                 realtime_processing_pause=INIT_REALTIME_PROCESSING_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,

                 # Voice activation parameters
                 silero_model_path: str = "",
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 silero_use_onnx: bool = False,
                 silero_deactivity_detection: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = (
                     INIT_POST_SPEECH_SILENCE_DURATION
                 ),
                 min_length_of_recording: float = (
                     INIT_MIN_LENGTH_OF_RECORDING
                 ),
                 min_gap_between_recordings: float = (
                     INIT_MIN_GAP_BETWEEN_RECORDINGS
                 ),
                 pre_recording_buffer_duration: float = (
                     INIT_PRE_RECORDING_BUFFER_DURATION
                 ),
                 on_vad_detect_start=None, # for call back function
                 on_vad_detect_stop=None, # for call back function

                 # Wake word parameters
                 wakeword_backend: str = "pvporcupine",
                 openwakeword_model_paths: str = None,
                 openwakeword_inference_framework: str = "onnx",
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = (
                    INIT_WAKE_WORD_ACTIVATION_DELAY
                 ),
                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
                 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
                 on_wakeword_detected=None,
                 on_wakeword_timeout=None,
                 on_wakeword_detection_start=None,
                 on_wakeword_detection_end=None,
                 on_recorded_chunk=None,
                 debug_mode=False,
                 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
                 beam_size: int = 5, # stt beam_size
                 beam_size_realtime: int = 3,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 initial_prompt: Optional[Union[str, Iterable[int]]] = None,
                 suppress_tokens: Optional[List[int]] = [-1],
                 ):
        """
        Initializes an audio recorder and  transcription
        and wake word detection.

        Args:
        - model (str, default="tiny"): Specifies the size of the transcription
          model to use or the path to a converted model directory.
                Valid options are 'tiny', 'tiny.en', 'base', 'base.en',
                'small', 'small.en', 'medium', 'medium.en', 'large-v1',
                'large-v2'.
                If a specific size is provided, the model is downloaded
                from the Hugging Face Hub.
        - language (str, default=""): Language code for speech-to-text engine.
            If not specified, the model will attempt to detect the language
            automatically.
        - compute_type (str, default="default"): Specifies the type of
            computation to be used for transcription.
            See https://opennmt.net/CTranslate2/quantization.html.
        - input_device_index (int, default=0): The index of the audio input
            device to use.
        - gpu_device_index (int, default=0): Device ID to use.
            The model can also be loaded on multiple GPUs by passing a list of
            IDs (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can
            run in parallel when transcribe() is called from multiple Python
            threads
        - device (str, default="cuda"): Device for model to use. Can either be 
            "cuda" or "cpu".
        - on_recording_start (callable, default=None): Callback function to be
            called when recording of audio to be transcripted starts.
        - on_recording_stop (callable, default=None): Callback function to be
            called when recording of audio to be transcripted stops.
        - on_transcription_start (callable, default=None): Callback function
            to be called when transcription of audio to text starts.
        - ensure_sentence_starting_uppercase (bool, default=True): Ensures
            that every sentence detected by the algorithm starts with an
            uppercase letter.
        - ensure_sentence_ends_with_period (bool, default=True): Ensures that
            every sentence that doesn't end with punctuation such as "?", "!"
            ends with a period
        - use_microphone (bool, default=True): Specifies whether to use the
            microphone as the audio input source. If set to False, the
            audio input source will be the audio data sent through the
            feed_audio() method.
        - spinner (bool, default=True): Show spinner animation with current
            state.
        - level (int, default=logging.WARNING): Logging level.
        - enable_realtime_transcription (bool, default=False): Enables or
            disables real-time transcription of audio. When set to True, the
            audio will be transcribed continuously as it is being recorded.
        - realtime_model_type (str, default="tiny"): Specifies the machine
            learning model to be used for real-time transcription. Valid
            options include 'tiny', 'tiny.en', 'base', 'base.en', 'small',
            'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
        - realtime_processing_pause (float, default=0.1): Specifies the time
            interval in seconds after a chunk of audio gets transcribed. Lower
            values will result in more "real-time" (frequent) transcription
            updates but may increase computational load.
        - on_realtime_transcription_update = A callback function that is
            triggered whenever there's an update in the real-time
            transcription. The function is called with the newly transcribed
            text as its argument.
        - on_realtime_transcription_stabilized = A callback function that is
            triggered when the transcribed text stabilizes in quality. The
            stabilized text is generally more accurate but may arrive with a
            slight delay compared to the regular real-time updates.
        - silero_sensitivity (float, default=SILERO_SENSITIVITY): Sensitivity
            for the Silero Voice Activity Detection model ranging from 0
            (least sensitive) to 1 (most sensitive). Default is 0.5.
        - silero_use_onnx (bool, default=False): Enables usage of the
            pre-trained model from Silero in the ONNX (Open Neural Network
            Exchange) format instead of the PyTorch format. This is
            recommended for faster performance.
		- silero_deactivity_detection (bool, default=False): Enables the Silero
            model for end-of-speech detection. More robust against background
            noise. Utilizes additional GPU resources but improves accuracy in
            noisy environments. When False, uses the default WebRTC VAD,
            which is more sensitive but may continue recording longer due
            to background sounds.
        - webrtc_sensitivity (int, default=WEBRTC_SENSITIVITY): Sensitivity
            for the WebRTC Voice Activity Detection engine ranging from 0
            (least aggressive / most sensitive) to 3 (most aggressive,
            least sensitive). Default is 3.
        - post_speech_silence_duration (float, default=0.2): Duration in
            seconds of silence that must follow speech before the recording
            is considered to be completed. This ensures that any brief
            pauses during speech don't prematurely end the recording.
        - min_gap_between_recordings (float, default=1.0): Specifies the
            minimum time interval in seconds that should exist between the
            end of one recording session and the beginning of another to
            prevent rapid consecutive recordings.
        - min_length_of_recording (float, default=1.0): Specifies the minimum
            duration in seconds that a recording session should last to ensure
            meaningful audio capture, preventing excessively short or
            fragmented recordings.
        - pre_recording_buffer_duration (float, default=0.2): Duration in
            seconds for the audio buffer to maintain pre-roll audio
            (compensates speech activity detection latency)
        - on_vad_detect_start (callable, default=None): Callback function to
            be called when the system listens for voice activity.
        - on_vad_detect_stop (callable, default=None): Callback function to be
            called when the system stops listening for voice activity.
        - wakeword_backend (str, default="pvporcupine"): Specifies the backend
            library to use for wake word detection. Supported options include
            'pvporcupine' for using the Porcupine wake word engine or 'oww' for
            using the OpenWakeWord engine.
        - openwakeword_model_paths (str, default=None): Comma-separated paths
            to model files for the openwakeword library. These paths point to
            custom models that can be used for wake word detection when the
            openwakeword library is selected as the wakeword_backend.
        - openwakeword_inference_framework (str, default="onnx"): Specifies
            the inference framework to use with the openwakeword library.
            Can be either 'onnx' for Open Neural Network Exchange format 
            or 'tflite' for TensorFlow Lite.
        - wake_words (str, default=""): Comma-separated string of wake words to
            initiate recording when using the 'pvporcupine' wakeword backend.
            Supported wake words include: 'alexa', 'americano', 'blueberry',
            'bumblebee', 'computer', 'grapefruits', 'grasshopper', 'hey google',
            'hey siri', 'jarvis', 'ok google', 'picovoice', 'porcupine',
            'terminator'. For the 'openwakeword' backend, wake words are
            automatically extracted from the provided model files, so specifying
            them here is not necessary.
        - wake_words_sensitivity (float, default=0.5): Sensitivity for wake
            word detection, ranging from 0 (least sensitive) to 1 (most
            sensitive). Default is 0.5.
        - wake_word_activation_delay (float, default=0): Duration in seconds
            after the start of monitoring before the system switches to wake
            word activation if no voice is initially detected. If set to
            zero, the system uses wake word activation immediately.
        - wake_word_timeout (float, default=5): Duration in seconds after a
            wake word is recognized. If no subsequent voice activity is
            detected within this window, the system transitions back to an
            inactive state, awaiting the next wake word or voice activation.
        - wake_word_buffer_duration (float, default=0.1): Duration in seconds
            to buffer audio data during wake word detection. This helps in
            cutting out the wake word from the recording buffer so it does not
            falsely get detected along with the following spoken text, ensuring
            cleaner and more accurate transcription start triggers.
            Increase this if parts of the wake word get detected as text.
        - on_wakeword_detected (callable, default=None): Callback function to
            be called when a wake word is detected.
        - on_wakeword_timeout (callable, default=None): Callback function to
            be called when the system goes back to an inactive state after when
            no speech was detected after wake word activation
        - on_wakeword_detection_start (callable, default=None): Callback
             function to be called when the system starts to listen for wake
             words
        - on_wakeword_detection_end (callable, default=None): Callback
            function to be called when the system stops to listen for
            wake words (e.g. because of timeout or wake word detected)
        - on_recorded_chunk (callable, default=None): Callback function to be
            called when a chunk of audio is recorded. The function is called
            with the recorded audio chunk as its argument.
        - debug_mode (bool, default=False): If set to True, the system will
            print additional debug information to the console.
        - handle_buffer_overflow (bool, default=True): If set to True, the system
            will log a warning when an input overflow occurs during recording and
            remove the data from the buffer.
        - beam_size (int, default=5): The beam size to use for beam search
            decoding.
        - beam_size_realtime (int, default=3): The beam size to use for beam
            search decoding in the real-time transcription model.
        - buffer_size (int, default=512): The buffer size to use for audio
            recording. Changing this may break functionality.
        - sample_rate (int, default=16000): The sample rate to use for audio
            recording. Changing this will very probably functionality (as the
            WebRTC VAD model is very sensitive towards the sample rate).
        - initial_prompt (str or iterable of int, default=None): Initial
            prompt to be fed to the transcription models.
        - suppress_tokens (list of int, default=[-1]): Tokens to be suppressed
            from the transcription output.

        Raises:
            Exception: Errors related to initializing transcription
            model, wake word detection, or audio recording.
        """
        self.shutdown_lock = threading.Lock()
        self.thomas_event_state = "normal" # 토마스 이벤트 송수신 상태 체크 변수
        # self.pyannote_flag = pyannote_flag
        self.reduce_db_flag = reduce_db_flag
        self.reduce_noise_flag = reduce_noise_flag
        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.device_name = device_name
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.wake_words = wake_words
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.ensure_sentence_starting_uppercase = (
            ensure_sentence_starting_uppercase
        )
        self.ensure_sentence_ends_with_period = (
            ensure_sentence_ends_with_period
        )
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.on_transcription_start = on_transcription_start
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.on_realtime_transcription_update = (
            on_realtime_transcription_update
        )
        self.on_realtime_transcription_stabilized = (
            on_realtime_transcription_stabilized
        )
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.allowed_latency_limit = ALLOWED_LATENCY_LIMIT

        self.level = level
        self.audio_queue = mp.Queue()
        """
        multiprocessing.Queue(또는 torch.multiprocessing.Queue)는 여러 프로세스 또는 스레드가 동시에 액세스해도 안전하도록 설계되었습니다.
내부적으로 Queue는 잠금 및 세마포어를 사용하여 주어진 시간에 하나의 프로세스 또는 스레드만 put() 또는 get() 작업을 수행할 수 있도록 합니다.
        """
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.speech_end_silence_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.silero_deactivity_detection = silero_deactivity_detection
        self.silero_model_path = silero_model_path
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.stream = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.last_transcription_bytes = None
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.use_wake_words = wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}

        # Initialize the logging configuration with the specified level
        log_format = 'RealTimeSTT: %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # log_format = 'RealTimeSTT: %(name)s - %(levelname)s - %(message)s'

        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(level)  # Set the root logger's level

        # Create a file handler and set its level
        file_handler = logging.FileHandler('realtimesst.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Create a console handler and set its level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(log_format))

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.is_shut_down = False
        # self.shutdown_event is terminate all ongoing processes -> _transcription_worker, _audio_data_worker
        # _transcription_worker 또는 _audio_data_worker 프로세스를 안전하게 종료해야 할 때 신호를 보내는 데 사용
        self.shutdown_event = mp.Event()

        try:
            logging.debug("Explicitly setting the multiprocessing start method to 'spawn'")
            mp.set_start_method('spawn') # cuda 쓰려면, spawn
        except RuntimeError as e:
            logging.debug(f"Start method has already been set. Details: {e}")

        logging.info("Starting RealTimeSTT")

        # mp.Event() 메서드 clear() -> enable (False), set() -> enable (True), is_set() -> enable check! (return True or False)
        # 전사가 중단되었을 때 이를 알리는 데 사용 -> _transcription_worker, _audio_data_worker 키보드로 종료 시 
        self.interrupt_stop_event = mp.Event()
        # 
        self.was_interrupted = mp.Event()
        # 전사 모델이 로드되어 데이터를 처리할 준비가 되면 설정됩니다.
        # _transcription_worker -> main_transcription_ready_event enable after faster-whisper model load
        self.main_transcription_ready_event = mp.Event()
        """
        mp.Pipe()를 호출하면 파이프의 두 끝을 나타내는 두 개의 연결 객체인 parent_transcription_pipe와 child_transcription_pipe가 반환됩니다.
        이 두 끝은 연결되어 있으므로 한 쪽에서 보낸 데이터를 다른 쪽에서 수신할 수 있습니다.

        - child_transcription_pipe
        역할: 전사 프로세스에서 child_transcription_pipe는 오디오 데이터를 수신하고 STT 모델을 사용하여 처리한 다음 파이프를 통해 결과를 다시 보냅니다.
        사용법: child_transcription_pipe는 parent_transcription_pipe에서 데이터를 수신받고 전사 결과를 다시 부모 파이프로 보내는 데 사용됩니다.
        
        - parent_transcription_pipe
        역할: parent_transcription_pipe는 오디오 데이터 및 언어 정보와 같은 데이터를 필사 프로세스로 보내고 필사 결과를 다시 받는 데 사용됩니다.
        사용법: 메인 프로세스에서 데이터를 보낸 후 프로세스는 parent_transcription_pipe에서 recv()를 호출하여 응답을 기다립니다.
        """
        self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()

        # Set device for model
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        
        # _start_thread에서 window os는 torch.multiprocessing을 사용.
        # 결국 _transcription_worker는 torch.multiprocessing이고 recording_thread는 Thread를 사용함.
        self.transcript_process = self._start_thread(
            target=AudioToTextRecorder._transcription_worker,
            args=(
                child_transcription_pipe,
                model,
                self.compute_type,
                self.gpu_device_index,
                self.device,
                self.main_transcription_ready_event,
                self.shutdown_event,
                self.interrupt_stop_event,
                self.beam_size,
                self.initial_prompt,
                self.suppress_tokens
            )
        )

        # Start audio data reading process
        if self.use_microphone.value:
            logging.info("Initializing audio recording"
                         " (creating pyAudio input stream,"
                         f" sample rate: {self.sample_rate}"
                         f" buffer size: {self.buffer_size}"
                         )
            self.reader_process = self._start_thread(
                target=AudioToTextRecorder._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.use_microphone
                )
            )

        # Initialize the realtime transcription model
        if self.enable_realtime_transcription:
            try:
                logging.info("Initializing faster_whisper realtime "
                             f"transcription model {self.realtime_model_type}"
                             )
                self.realtime_model_type = faster_whisper.WhisperModel(
                    model_size_or_path=self.realtime_model_type,
                    device=self.device,
                    compute_type=self.compute_type,
                    device_index=self.gpu_device_index
                )

            except Exception as e:
                logging.exception("Error initializing faster_whisper "
                                  f"realtime transcription model: {e}"
                                  )
                raise

            logging.debug("Faster_whisper realtime speech to text "
                          "transcription model initialized successfully")

        # Setup wake word detection
        if wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            self.wakeword_backend = wakeword_backend
            print(f"self.wakeword_backend: {self.wakeword_backend}\n")

            self.wake_words_list = [
                word.strip() for word in wake_words.lower().split(',')
            ]
            self.wake_words_sensitivity = wake_words_sensitivity
            self.wake_words_sensitivities = [
                float(wake_words_sensitivity)
                for _ in range(len(self.wake_words_list))
            ]

            if self.wakeword_backend in {'pvp', 'pvporcupine'}:
                print("pvporcupine enter !! \n")
                print(f"wake_words_list: {self.wake_words_list}\n")
                print(f"wake_words_sensitivities: {self.wake_words_sensitivities}\n")

                try:
                    self.porcupine = pvporcupine.create(
                        keywords=self.wake_words_list,
                        sensitivities=self.wake_words_sensitivities
                    )
                    self.buffer_size = self.porcupine.frame_length
                    self.sample_rate = self.porcupine.sample_rate

                except Exception as e:
                    logging.exception(
                        "Error initializing porcupine "
                        f"wake word detection engine: {e}"
                    )
                    raise

                logging.debug(
                    "Porcupine wake word detection engine initialized successfully"
                )

            elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
                    
                openwakeword.utils.download_models()

                try:
                    if openwakeword_model_paths:
                        model_paths = openwakeword_model_paths.split(',')
                        self.owwModel = Model(
                            wakeword_models=model_paths,
                            inference_framework=openwakeword_inference_framework
                        )
                        logging.info(
                            "Successfully loaded wakeword model(s): "
                            f"{openwakeword_model_paths}"
                        )
                    else:
                        self.owwModel = Model(
                            inference_framework=openwakeword_inference_framework)
                    
                    self.oww_n_models = len(self.owwModel.models.keys())
                    if not self.oww_n_models:
                        logging.error(
                            "No wake word models loaded."
                        )

                    for model_key in self.owwModel.models.keys():
                        logging.info(
                            "Successfully loaded openwakeword model: "
                            f"{model_key}"
                        )

                except Exception as e:
                    logging.exception(
                        "Error initializing openwakeword "
                        f"wake word detection engine: {e}"
                    )
                    raise

                logging.debug(
                    "Open wake word detection engine initialized successfully"
                )
            
            else:
                logging.exception(f"Wakeword engine {self.wakeword_backend} unknown/unsupported. Please specify one of: pvporcupine, openwakeword.")


        # Setup voice activity detection model WebRTC
        try:
            logging.info("Initializing WebRTC voice with "
                         f"Sensitivity {webrtc_sensitivity}"
                         )
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)

        except Exception as e:
            logging.exception("Error initializing WebRTC voice "
                              f"activity detection engine: {e}"
                              )
            raise

        logging.debug("WebRTC VAD voice activity detection "
                      "engine initialized successfully"
                      )

        # Setup voice activity detection model Silero VAD
        try:
            self.silero_vad_model, _ = torch.hub.load(
                repo_or_dir=self.silero_model_path,
                model="silero_vad",
                verbose=False,
                onnx=silero_use_onnx,
                source="local"
            )

        except Exception as e:
            logging.exception(f"Error initializing Silero VAD "
                              f"voice activity detection engine: {e}"
                              )
            raise

        logging.debug("Silero VAD voice activity detection "
                      "engine initialized successfully"
                      )

        self.audio_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       self.pre_recording_buffer_duration)
        )
        
        # self.frames는 녹음 상태 동안 self.audio_queue에서 검색된 모든 오디오 청크를 저장
        self.frames = []

        # Recording control flags
        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        # Start the recording worker thread
        """
        목적: 이 스레드는 녹음 프로세스가 주 프로그램 흐름과 독립적으로 실행되도록 하여 다른 작업을 차단하지 않고도 실시간 모니터링과 음성 활동에 대한 응답을 허용
        self.realtime_thread.daemon = True
        데몬 스레드가 메인 스레드와 동일한 프로세스에서 실행 중이더라도 독립적으로 동시에 작동합니다.
        데몬 스레드는 작업을 완료하거나 메인 스레드(및 프로세스)가 종료될 때까지 대상 함수(이 경우 _recording_worker)를 계속 실행합니다.
        """

        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True # 데몬 스레드는 백그라운드에서 실행되며 프로그램이 종료되는 것을 막지 않습니다. 
                                            # 메인 프로그램이 완료되면 데몬 스레드가 자동으로 종료됩니다.
        self.recording_thread.start()

        # Start the realtime transcription worker thread
        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
                   
        # Wait for transcription models to start
        logging.debug('Waiting for main transcription model to start')
        self.main_transcription_ready_event.wait()
        logging.debug('Main transcription model ready')

        logging.debug('RealtimeSTT initialization completed successfully')
                   
    def _start_thread(self, target=None, args=()):
        """
        Implement a consistent threading model across the library.

        This method is used to start any thread in this library. It uses the
        standard threading. Thread for Linux and for all others uses the pytorch
        MultiProcessing library 'Process'.
        Args:
            target (callable object): is the callable object to be invoked by
              the run() method. Defaults to None, meaning nothing is called.
            args (tuple): is a list or tuple of arguments for the target
              invocation. Defaults to ().
        """
        if (platform.system() == 'Linux'):
            thread = threading.Thread(target=target, args=args)
            thread.deamon = True
            thread.start()
            return thread
        else:
            thread = mp.Process(target=target, args=args)
            thread.start()
            return thread

    @staticmethod
    def _transcription_worker(conn,
                              model_path,
                              compute_type,
                              gpu_device_index,
                              device,
                              ready_event,
                              shutdown_event,
                              interrupt_stop_event,
                              beam_size,
                              initial_prompt,
                              suppress_tokens
                              ):
        """
        Worker method that handles the continuous
        process of transcribing audio data.

        This method runs in a separate process and is responsible for:
        - Initializing the `faster_whisper` model used for transcription.
        - Receiving audio data sent through a pipe and using the model
          to transcribe it.
        - Sending transcription results back through the pipe.
        - Continuously checking for a shutdown event to gracefully
          terminate the transcription process.

        Args:
            conn (multiprocessing.Connection): The connection endpoint used
              for receiving audio data and sending transcription results.
            model_path (str): The path to the pre-trained faster_whisper model
              for transcription.
            compute_type (str): Specifies the type of computation to be used
                for transcription.
            gpu_device_index (int): Device ID to use.
            device (str): Device for model to use.
            ready_event (threading.Event): An event that is set when the
              transcription model is successfully initialized and ready.
            shutdown_event (threading.Event): An event that, when set,
              signals this worker method to terminate.
            interrupt_stop_event (threading.Event): An event that, when set,
                signals this worker method to stop processing audio data.
            beam_size (int): The beam size to use for beam search decoding.
            initial_prompt (str or iterable of int): Initial prompt to be fed
                to the transcription model.
            suppress_tokens (list of int): Tokens to be suppressed from the
                transcription output.
        Raises:
            Exception: If there is an error while initializing the
            transcription model.
        """

        logging.info("Initializing faster_whisper "
                     f"main transcription model {model_path}"
                     )

        try:
            model = faster_whisper.WhisperModel(
                model_size_or_path=model_path,
                device=device,
                compute_type=compute_type,
                device_index=gpu_device_index,
            )

        except Exception as e:
            logging.exception("Error initializing main "
                              f"faster_whisper transcription model: {e}"
                              )
            raise
        
        # self.main_transcription_ready_event enable
        ready_event.set()

        logging.debug("Faster_whisper main speech to text "
                      "transcription model initialized successfully"
                      )

        while not shutdown_event.is_set():
            try:
                if conn.poll(0.5): # poll(0.5)을 사용하면 작업자가 최대 0.5초 동안 데이터를 기다리고, 데이터를 사용할 수 없으면 다른 코드를 계속 실행하거나 데이터가 없는 것을 적절히 처리
                    
                    audio, language = conn.recv() # self.parent_transcription_pipe.send((self.audio, self.language)) 이 호출되면, 파라미터 값을 반환 받음.
                    try:
                        segments = model.transcribe(
                            audio,
                            language=language if language else None,
                            beam_size=beam_size,
                            initial_prompt=initial_prompt,
                            suppress_tokens=suppress_tokens
                        )
                        segments = segments[0]
                        # for seg in segments:
                        #     print(f"seg: {seg}\n") 
                        # seg: Segment(id=1, seek=208, start=0.0, end=2.0, text=' 반갑습니다.', tokens=[50364, 16396, 27358, 3115, 13, 50464], temperature=0.0, avg_logprob=-0.4255022406578064, compression_ratio=0.64, no_speech_prob=0.210693359375, words=None)
                        
                        transcription = " ".join(seg.text for seg in segments)
                        transcription = transcription.strip()
                        # 여기서 conn.send(('success', transcription))를 활용해서 status, result = self.parent_transcription_pipe.recv()로 반환하는 값을 전달.
                        conn.send(('success', transcription))
                    except Exception as e:
                        logging.error(f"General transcription error: {e}")
                        conn.send(('error', str(e)))
                else:
                    # If there's no data, sleep / prevent busy waiting
                    time.sleep(0.02)
            except KeyboardInterrupt: # KeyboardInterrupt: 이 예외는 사용자가 일반적으로 Ctrl+C를 사용하여 프로그램 실행을 수동으로 중단할 때 발생
                interrupt_stop_event.set() # self.interrupt_stop_event -> True
                logging.debug("Transcription worker process "
                              "finished due to KeyboardInterrupt"
                              )
                break # 이것은 while 루프를 종료하여 전사 작업자 함수를 효과적으로 종료합니다.
    
    @staticmethod
    def _audio_data_worker(audio_queue,
                        target_sample_rate,
                        buffer_size,
                        input_device_index,
                        shutdown_event,
                        interrupt_stop_event,
                        use_microphone):
        """
        Worker method that handles the audio recording process.

        This method runs in a separate process and is responsible for:
        - Setting up the audio input stream for recording at the highest possible sample rate.
        - Continuously reading audio data from the input stream, resampling if necessary,
        preprocessing the data, and placing complete chunks in a queue.
        - Handling errors during the recording process.
        - Gracefully terminating the recording process when a shutdown event is set.

        Args:
            audio_queue (queue.Queue): A queue where recorded audio data is placed.
            target_sample_rate (int): The desired sample rate for the output audio (for Silero VAD).
            buffer_size (int): The number of samples expected by the Silero VAD model.
            input_device_index (int): The index of the audio input device.
            shutdown_event (threading.Event): An event that, when set, signals this worker method to terminate.
            interrupt_stop_event (threading.Event): An event to signal keyboard interrupt.
            use_microphone (multiprocessing.Value): A shared value indicating whether to use the microphone.

        Raises:
            Exception: If there is an error while initializing the audio recording.
        """
        import pyaudio
        import numpy as np
        from scipy import signal
        
        if __name__ == '__main__':
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

        def get_highest_sample_rate(audio_interface, device_index):
            """Get the highest supported sample rate for the specified device."""
            try:
                device_info = audio_interface.get_device_info_by_index(device_index)
                max_rate = int(device_info['defaultSampleRate'])
                
                if 'supportedSampleRates' in device_info:
                    supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
                    if supported_rates:
                        max_rate = max(supported_rates)
                
                return max_rate
            except Exception as e:
                logging.warning(f"Failed to get highest sample rate: {e}")
                return 48000  # Fallback to a common high sample rate

        def initialize_audio_stream(audio_interface, sample_rate, chunk_size):
            nonlocal input_device_index
            """Initialize the audio stream with error handling."""
            while not shutdown_event.is_set():
                try:
                    # Check and assign the input device index if it is not set
                    if input_device_index is None:
                        default_device = audio_interface.get_default_input_device_info()
                        input_device_index = default_device['index']

                    stream = audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=input_device_index,
                    )
                    logging.info("Microphone connected successfully.")
                    return stream

                except Exception as e:
                    logging.error(f"Microphone connection failed: {e}. Retrying...")
                    input_device_index = None
                    time.sleep(3)  # Wait for 3 seconds before retrying

        def preprocess_audio(chunk, original_sample_rate, target_sample_rate):
            """Preprocess audio chunk similar to feed_audio method."""
            if isinstance(chunk, np.ndarray):
                # Handle stereo to mono conversion if necessary
                if chunk.ndim == 2:
                    chunk = np.mean(chunk, axis=1)

                # Resample to target_sample_rate if necessary
                if original_sample_rate != target_sample_rate:
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)

                # Ensure data type is int16
                chunk = chunk.astype(np.int16)
            else:
                # If chunk is bytes, convert to numpy array
                chunk = np.frombuffer(chunk, dtype=np.int16)

                # Resample if necessary
                if original_sample_rate != target_sample_rate:
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)
                    chunk = chunk.astype(np.int16)

            return chunk.tobytes()

        audio_interface = None
        stream = None
        device_sample_rate = None
        chunk_size = 1024  # Increased chunk size for better performance

        def setup_audio():  
            nonlocal audio_interface, stream, device_sample_rate, input_device_index
            try:
                audio_interface = pyaudio.PyAudio()
                if input_device_index is None:
                    try:
                        default_device = audio_interface.get_default_input_device_info()
                        input_device_index = default_device['index']
                    except OSError as e:
                        input_device_index = None

                sample_rates_to_try = [16000]  # Try 16000 Hz first
                if input_device_index is not None:
                    highest_rate = get_highest_sample_rate(audio_interface, input_device_index)
                    if highest_rate != 16000:
                        sample_rates_to_try.append(highest_rate)
                else:
                    sample_rates_to_try.append(48000)  # Fallback sample rate

                for rate in sample_rates_to_try:
                    try:
                        device_sample_rate = rate
                        stream = initialize_audio_stream(audio_interface, device_sample_rate, chunk_size)
                        if stream is not None:
                            logging.debug(f"Audio recording initialized successfully at {device_sample_rate} Hz, reading {chunk_size} frames at a time")
                            # logging.error(f"Audio recording initialized successfully at {device_sample_rate} Hz, reading {chunk_size} frames at a time")
                            return True
                    except Exception as e:
                        logging.warning(f"Failed to initialize audio23 stream at {device_sample_rate} Hz: {e}")
                        continue

                # If we reach here, none of the sample rates worked
                raise Exception("Failed to initialize audio stream12 with all sample rates.")

            except Exception as e:
                logging.exception(f"Error initializing pyaudio audio recording: {e}")
                if audio_interface:
                    audio_interface.terminate()
                return False

        if not setup_audio():
            raise Exception("Failed to set up audio recording.")

        buffer = bytearray()
        silero_buffer_size = 2 * buffer_size  # silero complains if too short

        time_since_last_buffer_message = 0
        try:
            while not shutdown_event.is_set():
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)

                    # print(f"data len {len(data)}\n")
                    
                    if use_microphone.value:
                        processed_data = preprocess_audio(data, device_sample_rate, target_sample_rate)
                        buffer += processed_data

                        # Check if the buffer has reached or exceeded the silero_buffer_size
                        while len(buffer) >= silero_buffer_size:
                            # Extract silero_buffer_size amount of data from the buffer
                            to_process = buffer[:silero_buffer_size]
                            buffer = buffer[silero_buffer_size:]

                            # Feed the extracted data to the audio_queue
                            if time_since_last_buffer_message:
                                time_passed = time.time() - time_since_last_buffer_message
                                if time_passed > 1:
                                    logging.debug("_audio_data_worker writing audio data into queue.")
                                    time_since_last_buffer_message = time.time()
                            else:
                                time_since_last_buffer_message = time.time()

                            audio_queue.put(to_process)
                            

                except OSError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        logging.warning("Input overflowed. Frame dropped.")
                    else:
                        logging.error(f"OSError during recording: {e}")
                        # Attempt to reinitialize the stream
                        logging.info("Attempting to reinitialize the audio stream...")

                        try:
                            if stream:
                                stream.stop_stream()
                                stream.close()
                        except Exception as e:
                            pass

                        # if audio_interface:
                        #     audio_interface.terminate()
                        
                        # Wait a bit before trying to reinitialize
                        time.sleep(1)
                        
                        if not setup_audio():
                            logging.error("Failed to reinitialize audio stream. Exiting.")
                            break
                        else:
                            logging.info("Audio stream reinitialized successfully.")
                    continue

                except Exception as e:
                    logging.error(f"Unknown error during recording: {e}")
                    tb_str = traceback.format_exc()
                    logging.error(f"Traceback: {tb_str}")
                    logging.error(f"Error: {e}")
                    # Attempt to reinitialize the stream
                    logging.info("Attempting to reinitialize the audio stream...")
                    if stream:
                        stream.stop_stream()
                        stream.close()
                    if audio_interface:
                        audio_interface.terminate()
                    
                    # Wait a bit before trying to reinitialize
                    time.sleep(0.5)
                    
                    if not setup_audio():
                        logging.error("Failed to reinitialize audio stream. Exiting.")
                        break
                    else:
                        logging.info("Audio stream reinitialized successfully.")
                    continue

        except KeyboardInterrupt:
            interrupt_stop_event.set()
            logging.debug("Audio data worker process finished due to KeyboardInterrupt")
        finally:
            # After recording stops, feed any remaining audio data
            if buffer:
                audio_queue.put(bytes(buffer))
            
            if stream:
                stream.stop_stream()
                stream.close()
            if audio_interface:
                audio_interface.terminate()


    def wakeup(self):
        """
        If in wake work modus, wake up as if a wake word was spoken.
        """
        self.listen_start = time.time()

    def abort(self):
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self._set_state("inactive")
        self.interrupt_stop_event.set()
        self.was_interrupted.wait()
        self.was_interrupted.clear()

    def _db_reduce(self, audio, reduce_factor):

        # Apply decibel reduction
        dB_reduction = -reduce_factor  # Reduce by reduce_factor dB
        scaling_factor = 10 ** (dB_reduction / 20)  # Calculate the scaling factor
        audio *= scaling_factor  # Apply the scaling factor to reduce the decibel level

        return audio

    # def _pyannote(self, audio, sampling_rate):
    #     if isinstance(audio, np.ndarray):
    #         # (채널, 시간) 형식으로 맞추기 위해 차원을 추가
    #         audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, time) 형식
    #     else:
    #         raise TypeError("Audio data must be provided as a numpy array.")
    #     # 사전 훈련된 화자 분할 파이프라인을 로드합니다. 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
    #     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token="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")
    #     # print(f"pipeline {pipeline}")
    #     diarization = pipeline({"waveform": audio_tensor, "sample_rate": sampling_rate})
    #     # print(pipeline)
    #     # print(diarization)
    #     # 각 화자에 대해 발화 구간을 확인
    #     # 가장 많이 발화한 화자를 찾기 위한 작업
    #     speaker_durations = {}
    #     for turn, _, speaker in diarization.itertracks(yield_label=True):
    #         duration = turn.end - turn.start
    #         if speaker not in speaker_durations:
    #             speaker_durations[speaker] = duration
    #         else:
    #             speaker_durations[speaker] += duration

    #     # 가장 많이 말한 화자를 찾음
    #     main_speaker = max(speaker_durations, key=speaker_durations.get)

    #     # 메인 화자의 발화 구간 추출
    #     main_speaker_segments = [turn for turn, _, speaker in diarization.itertracks(yield_label=True) if speaker == main_speaker]

    #     # 메인 화자의 음성을 추출
    #     main_speaker_audio = []

    #     for segment in main_speaker_segments:
    #         start_sample = int(segment.start * sampling_rate)
    #         end_sample = int(segment.end * sampling_rate)
    #         main_speaker_audio.append(self.audio[start_sample:end_sample])

    #     # 메인 화자의 음성을 하나의 numpy 배열로 결합
    #     main_speaker_audio = np.concatenate(main_speaker_audio)
    #     return main_speaker_audio

    def wait_audio(self):
        """
        Waits for the start and completion of the audio recording process.

        This method is responsible for:
        - Waiting for voice activity to begin recording if not yet started.
        - Waiting for voice inactivity to complete the recording.
        - Setting the audio buffer from the recorded frames.
        - Resetting recording-related attributes.

        Side effects:
        - Updates the state of the instance.
        - Modifies the audio attribute to contain the processed audio data.
        전사 조건이 충족되면,
        wait_audio()가 호출되어 누적된 오디오 데이터를 처리
        """
        # 청취 시작 시간 저장 
        self.listen_start = time.time()

        # If not yet started recording, wait for voice activity to initiate.
        if not self.is_recording and not self.frames:
            self._set_state("listening")
            self.start_recording_on_voice_activity = True

            # Wait until recording starts
            while not self.interrupt_stop_event.is_set():
                if self.start_recording_event.wait(timeout=0.02):
                    break

        # If recording is ongoing, wait for voice inactivity
        # to finish recording.
        if self.is_recording:
            self.stop_recording_on_voice_deactivity = True

            # Wait until recording stops
            # self.interrupt_stop_event.is_set() 조건은 중단 신호가 수신되면 녹음을 즉시 중지할 수 있도록 하여 녹음 프로세스를 즉시 중단할 수 있는 방법을 제공
            while not self.interrupt_stop_event.is_set():
                if (self.stop_recording_event.wait(timeout=0.02)):
                    break

        # Convert recorded frames to the appropriate audio format.
        audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16) # byte형을 np.int16으로 변환. 배열의 각 요소는 특정 시간의 오디오 신호 진폭
        self.audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE # Whisper 모델에서 오디오 데이터의 format이 32비트 부동소수점을 사용해서 float32로 변환. 그리고 int16의 최대 값을 나누어 정규화.
        # print(f"self.audio.shape : {self.audio.shape}") # self.audio.shape은 음성 길이에 따라 가변. 일단 지금 buffer overflow queue max size가 10초 이기 때문에 최대 160,000 shape까지 쌓임.
        
        if self.reduce_noise_flag ==True:
            self.audio = nr.reduce_noise(y=self.audio, sr=self.sample_rate)

        # 코드수정 20240904 화자분리 -> 유료이며, 처리 속도 증대로 미적용
        # if self.pyannote_flag == True:
        #     # print("pyannote_flag\n")
        #     self.audio = self._pyannote(self.audio, self.sample_rate)
        
        # 코드수정 20240904 _db_reduce 적용
        if self.reduce_db_flag == True:
            # print("reduce_db_flag\n")
            self.audio = self._db_reduce(self.audio,reduce_factor = 0.1)

        self.frames.clear()

        # Reset recording-related timestamps
        self.recording_stop_time = 0
        self.listen_start = 0

        self._set_state("inactive")

    def transcribe(self):
        """
        Transcribes audio captured by this class instance using the
        `faster_whisper` model.

        Automatically starts recording upon voice activity if not manually
          started using `recorder.start()`.
        Automatically stops recording upon voice deactivity if not manually
          stopped with `recorder.stop()`.
        Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): Callback function
              to be executed when transcription is ready.
            If provided, transcription will be performed asynchronously,
              and the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously,
              and the result will be returned.

        Returns (if no callback is set):
            str: The transcription of the recorded audio.

        Raises:
            Exception: If there is an error during the transcription process.
        """
        
        self._set_state("transcribing")
        audio_copy = copy.deepcopy(self.audio)
        # print("transcribe self.parent_transcription_pipe.send((self.audio, self.language))")
        # print(f"\naudio_copy : {audio_copy}\n")
        # print(f"\nself.audio : {self.audio}\n")
        # print(f"\naudio_copy shape : {audio_copy.shape}\n")
        self.parent_transcription_pipe.send((self.audio, self.language))
        status, result = self.parent_transcription_pipe.recv()
        # print(f"\nresult : {result}\n")
        self._set_state("inactive")
        if status == 'success':
            self.last_transcription_bytes = audio_copy
            return self._preprocess_output(result)
        else:
            logging.error(result)
            raise Exception(result)

    def _process_wakeword(self, data):
        """
        Processes audio data to detect wake words.
        """
        if self.wakeword_backend in {'pvp', 'pvporcupine'}:
            pcm = struct.unpack_from(
                "h" * self.buffer_size,
                data
            )
            porcupine_index = self.porcupine.process(pcm)
            if self.debug_mode:
                print (f"wake words porcupine_index: {porcupine_index}")
            return self.porcupine.process(pcm)

        elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            pcm = np.frombuffer(data, dtype=np.int16)
            prediction = self.owwModel.predict(pcm)
            # print(f"prediction = {prediction}\n")
            max_score = -1
            max_index = -1
            wake_words_in_prediction = len(self.owwModel.prediction_buffer.keys())
            self.wake_words_sensitivities
            if wake_words_in_prediction:
                for idx, mdl in enumerate(self.owwModel.prediction_buffer.keys()):
                    scores = list(self.owwModel.prediction_buffer[mdl])
                    # print(f"idx = {idx}, scores[-1] = {scores[-1]}")
                    if scores[-1] >= self.wake_words_sensitivity and scores[-1] > max_score:
                        max_score = scores[-1]
                        max_index = idx
                if self.debug_mode:
                    print (f"wake words oww max_index, max_score: {max_index} {max_score}")
                # print(f"max_index = {max_index}\n")
                return max_index  
            else:
                if self.debug_mode:
                    print (f"wake words oww_index: -1")
                return -1

        if self.debug_mode:        
            print("wake words no match")
        return -1
    
    def text(self,
             on_transcription_finished=None,
             start_time=None,
             communicator=None,
             similarity_cal=None,
             similarity_config=None,
             ):
        """
        Transcribes audio captured by this class instance
        using the `faster_whisper` model.

        - Automatically starts recording upon voice activity if not manually
          started using `recorder.start()`.
        - Automatically stops recording upon voice deactivity if not manually
          stopped with `recorder.stop()`.
        - Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): Callback function
              to be executed when transcription is ready.
            If provided, transcription will be performed asynchronously, and
              the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously,
              and the result will be returned.

        Returns (if not callback is set):
            str: The transcription of the recorded audio
        """

        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
            
        self.wait_audio()
        
        if self.is_shut_down or self.interrupt_stop_event.is_set(): # is_shut_down 플래그는 전체 시스템이 종료되었는지, self.interrupt_stop_event.is_set()은 프로세스를 중단하는 이벤트가 설정되었는지
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return "" # 전사 종료 시 빈 문자열 반환

        inf_text = self.transcribe().rstrip(".?!")
        
        if on_transcription_finished:
            """

            on_transcription_finished 콜백 함수는 전사 후 수행할 사용자 지정 작업을 정의할 수 있는 강력한 기능입니다. 
            
            콜백을 사용하면 전사 프로세스가 메인 애플리케이션을 차단하지 않도록 할 수 있으므로 실시간 또는 대화형 애플리케이션 개발 가능.

            전사처리가 완료된 후에 결과를 처리하도록 설계됨. 즉, 시간 초과로 조기 중단을 구현하는 것과 다르다는 것을 인식.

            """
            thomas_serial_process_event = threading.Event() # set() 메서드로 참으로 설정하고 clear() 메서드로 거짓으로 재설정 할 수 있는 플래그를 관리
            
            def wrapper():
                self.thomas_event_state = on_transcription_finished(inf_text, start_time, communicator, similarity_cal, similarity_config, self.thomas_event_state)

                thomas_serial_process_event.set() # on_transcription_finished 콜백 함수가 처리가 완료되고 self.thomas_event_state가 업데이트 되면 set으로 event 변수 업데이트
                
            self.transcription_thread = threading.Thread(target=wrapper)
            
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
            thomas_serial_process_event.wait() # wrapper함수에서 on_transcription_finished 콜백 함수가 처리가 완료되고 self.thomas_event_state가 업데이트 될때까지 기다림.
            
            return inf_text, self.thomas_event_state
            
        else:
            # 콜백함수가 정의되지 않았으니 그냥 전사처리 완료 후 바로 출력 
            
            return inf_text, self.thomas_event_state

    def start(self):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        # Ensure there's a minimum interval
        # between stopping and starting recording
        # 녹음 사이에 충분한 시간이 지났는지 확인하는 역할
        # min_gap_between_recordings -> 0이니 절대 True가 될수 없다. 녹음이 시작되는 것을 방해하지 않으므로 녹음은 중지한 직후에 바로 시작.
        if (time.time() - self.recording_stop_time
                < self.min_gap_between_recordings):
            logging.info("Attempted to start recording "
                         "too soon after stopping."
                         )
            return self

        logging.info("recording started")
        self._set_state("recording")
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames = [] # recording start() self.frames가 지워져 녹음 세션 중에 캡처된 오디오 데이터를 저장할 새 버퍼가 준비
        self.is_recording = True # recording start flag
        self.recording_start_time = time.time() # 녹음이 시작된 시점 추적을 위한 time
        self.is_silero_speech_active = False # 거짓 양성을 방지합니다. 이러한 플래그를 재설정하지 않으면 새 녹음이 시작될 때 시스템이 음성이 여전히 활성 상태라고 잘못 가정하여 음성 감지에서 오작동이나 오류가 발생
        self.is_webrtc_speech_active = False # 거짓 양성을 방지합니다. 이러한 플래그를 재설정하지 않으면 새 녹음이 시작될 때 시스템이 음성이 여전히 활성 상태라고 잘못 가정하여 음성 감지에서 오작동이나 오류가 발생
        self.stop_recording_event.clear() # self.stop_recording_event 중지 이벤트를 지워 이전 녹음의 잔여 상태로 인해 현재 녹음이 제대로 중지되지 않도록 합니다.
        self.start_recording_event.set() # recording start event enable

        # None으로 무시 
        if self.on_recording_start:
            self.on_recording_start() # 초기화 중에 on_recording_start 콜백이 제공된 경우 여기에서 호출됩니다. 향후 이를 통해 녹음이 시작될 때 UI 업데이트나 로깅과 같은 외부 작업을 트리거할 수 있습니다.

        return self

    def stop(self):
        """
        Stops recording audio.
        """

        # Ensure there's a minimum interval
        # between starting and stopping recording
        # recording이 self.min_length_of_recording초 동안 지속되었는지 확인, self.min_length_of_recording 보다 작으면 멈추지 않음. 
        # self.min_length_of_recording = 0.5
        if (time.time() - self.recording_start_time
                < self.min_length_of_recording):
            logging.info("Attempted to stop recording "
                         "too soon after starting."
                         )
            return self
        # start(self) 함수 내용 참조 구현이 비슷함.
        logging.info("recording stopped")
        self.is_recording = False
        self.recording_stop_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        if self.on_recording_stop:
            self.on_recording_stop()

        return self

    def feed_audio(self, chunk, original_sample_rate=16000):
        """
        Feed an audio chunk into the processing pipeline. Chunks are
        accumulated until the buffer size is reached, and then the accumulated
        data is fed into the audio_queue.
        """
        # Check if the buffer attribute exists, if not, initialize it
        if not hasattr(self, 'buffer'):
            self.buffer = bytearray()

        # Check if input is a NumPy array
        if isinstance(chunk, np.ndarray):
            # Handle stereo to mono conversion if necessary
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            # Resample to 16000 Hz if necessary
            if original_sample_rate != 16000:
                num_samples = int(len(chunk) * 16000 / original_sample_rate)
                chunk = resample(chunk, num_samples)

            # Ensure data type is int16
            chunk = chunk.astype(np.int16)

            # Convert the NumPy array to bytes
            chunk = chunk.tobytes()

        # Append the chunk to the buffer
        self.buffer += chunk
        buf_size = 2 * self.buffer_size  # silero complains if too short

        # Check if the buffer has reached or exceeded the buffer_size
        while len(self.buffer) >= buf_size:
            # Extract self.buffer_size amount of data from the buffer
            to_process = self.buffer[:buf_size]
            self.buffer = self.buffer[buf_size:]

            # Feed the extracted data to the audio_queue
            self.audio_queue.put(to_process)

    def set_microphone(self, microphone_on=True):
        """
        Set the microphone on or off.
        """
        logging.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on

    def shutdown(self):
        """
        Safely shuts down the audio recording by stopping the
        recording worker and closing the audio stream.
        """
        
        with self.shutdown_lock:
            if self.is_shut_down:
                return
             
        # Force wait_audio() and text() to exit
        self.is_shut_down = True
        self.start_recording_event.set()
        self.stop_recording_event.set()

        self.shutdown_event.set()
        self.is_recording = False
        self.is_running = False

        logging.debug('Finishing recording thread')
        if self.recording_thread:
            self.recording_thread.join(timeout=5)
            if self.recording_thread.is_alive():
                logging.warning("Recording thread did not terminate in time. Terminating forcefully.")
                self.recording_thread = None

        logging.debug('Terminating reader process')

        # Give it some time to finish the loop and cleanup.
        if self.use_microphone:
            self.reader_process.join(timeout=5)

        if self.reader_process.is_alive():
            logging.warning("Reader process did not terminate "
                            "in time. Terminating forcefully."
                            )
            self.reader_process.terminate()

        logging.debug('Terminating transcription process')
        self.transcript_process.join(timeout=5)

        if self.transcript_process.is_alive():
            logging.warning("Transcript process did not terminate "
                            "in time. Terminating forcefully."
                            )
            self.transcript_process.terminate()

        self.parent_transcription_pipe.close()

        logging.debug('Finishing realtime thread')
        if self.realtime_thread:
            self.realtime_thread.join()

        if self.enable_realtime_transcription:
            if self.realtime_model_type:
                del self.realtime_model_type
                self.realtime_model_type = None
                
        """
        20250310 self.transcription_thread가 존재하고 실행 중인 경우 종료
        """
        # Terminate the transcription thread if it exists
        if hasattr(self, 'transcription_thread') and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=10)
            if self.transcription_thread.is_alive():
                logging.warning("Transcription thread did not terminate in time. Terminating forcefully.")
                # Forcefully terminate the thread if necessary
                self.transcription_thread = None
                
        gc.collect()

    def _recording_worker(self):
        """
        The main worker method which constantly monitors the audio
        input for voice activity and accordingly starts/stops the recording.

        음성 활동을 위해 오디오 입력을 지속적으로 모니터링하고 
        그에 따라 녹음을 시작/중지하는 주요 작업자 방식입니다.
        """

        logging.debug('Starting recording worker')

        try:
            was_recording = False # 이전에 녹음이 진행 중이었는지 추적
            delay_was_passed = False # 웨이크 워드 활성화 지연이 지났는지 여부

            # Continuously monitor audio for voice activity
            while self.is_running: # 오디오 입력을 지속적으로 확인
                
                try:

                    data = self.audio_queue.get() # audio_queue에 저장된 음성 chunk나 segment를 가져옴 
                                                  # -> 여기서 처리가 느려져 self.allowed_latency_limit 값보다 쌓인 청크의 수가 많으면 
                                                  # 워닝 발생 후 오래된 청크 삭제  
                    if self.on_recorded_chunk: # 이건 내가 callback 함수를 변수로 전달 안하니 none으로 들어옴
                        self.on_recorded_chunk(data) # audio chunk data 활용 callback 함수 처리 코드

                    if self.handle_buffer_overflow: # mac OS 아닌 경우 True
                        """
                        오디오 대기열 크기가 허용된 대기 시간 제한(self.allowed_latency_limit)을 초과하면 
                        경고가 기록되고 이전 오디오 청크는 대기열 크기를 관리하기 위해 삭제
                        현재는 cpu, gpu 처리속도가 더 빨라 잘 처리됨
                        """
                        # Handle queue overflow
                        # self.audio_queue.qsize() (return audio chunk num)
                        if (self.audio_queue.qsize() >
                                self.allowed_latency_limit):
                            logging.warning("Audio queue size exceeds "
                                            "latency limit. Current size: "
                                            f"{self.audio_queue.qsize()}. "
                                            "Discarding old audio chunks."
                                            )

                        while (self.audio_queue.qsize() >
                                self.allowed_latency_limit):
                            
                            """
                            while 루프에 들어가 대기열 크기가 허용된 제한으로 줄어들 때까지 
                            self.audio_queue.get()을 사용하여 대기열에서 청크를 반복적으로 제거(삭제)합니다.
                            """

                            data = self.audio_queue.get()
                except BrokenPipeError:
                    print("BrokenPipeError _recording_worker")
                    self.is_running = False
                    break

                if not self.is_recording:
                    # Handle not recording state
                    time_since_listen_start = (time.time() - self.listen_start
                                               if self.listen_start else 0)

                    wake_word_activation_delay_passed = (
                        time_since_listen_start >
                        self.wake_word_activation_delay
                    )

                    # Handle wake-word timeout callback
                    if wake_word_activation_delay_passed \
                            and not delay_was_passed:

                        if self.use_wake_words and self.wake_word_activation_delay:
                            if self.on_wakeword_timeout:
                                self.on_wakeword_timeout()
                    delay_was_passed = wake_word_activation_delay_passed

                    # Set state and spinner text
                    if not self.recording_stop_time:
                        if self.use_wake_words \
                                and wake_word_activation_delay_passed \
                                and not self.wakeword_detected:
                            self._set_state("wakeword")
                        else:
                            if self.listen_start:
                                self._set_state("listening")
                            else:
                                self._set_state("inactive")

                    #self.wake_word_detect_time = time.time()
                    if self.use_wake_words and wake_word_activation_delay_passed:
                        try:
                            wakeword_index = self._process_wakeword(data)
                            # print(f"wakeword_index: {wakeword_index}")

                        except struct.error:
                            logging.error("Error unpacking audio data "
                                          "for wake word processing.")
                            continue

                        except Exception as e:
                            logging.error(f"Wake word processing error: {e}")
                            continue

                        # If a wake word is detected                        
                        if wakeword_index >= 0:
                            
                            print("wakeword_detect\n")

                            # Removing the wake word from the recording
                            samples_time = int(self.sample_rate * self.wake_word_buffer_duration)
                            start_index = max(
                                0,
                                len(self.audio_buffer) - samples_time
                                )
                            temp_samples = collections.deque(
                                itertools.islice(
                                    self.audio_buffer,
                                    start_index,
                                    None)
                                )
                            self.audio_buffer.clear()
                            self.audio_buffer.extend(temp_samples)

                            self.wake_word_detect_time = time.time()
                            self.wakeword_detected = True
                            #self.wake_word_cooldown_time = time.time()
                            if self.on_wakeword_detected:
                                self.on_wakeword_detected()

                    # Check for voice activity to
                    # trigger the start of recording
                    # wake_word를 사용 안하니 그냥 self.start_recording_on_voice_activity == True가 되면 진입
                    if ((not self.use_wake_words
                         or not wake_word_activation_delay_passed)
                            and self.start_recording_on_voice_activity) \
                            or self.wakeword_detected:

                        if self._is_voice_active():
                            logging.info("voice activity detected")

                            # start() is called. -> recording start trigger!
                            self.start()

                            if self.is_recording:
                                self.start_recording_on_voice_activity = False

                                # Add the buffered audio
                                # to the recording frames
                                self.frames.extend(list(self.audio_buffer))
                                self.audio_buffer.clear()

                            self.silero_vad_model.reset_states()
                        else:
                            data_copy = data[:]
                            self._check_voice_activity(data_copy)

                    self.speech_end_silence_start = 0

                else:
                    # If we are currently recording -> self.is_recording == True

                    # Stop the recording if silence is detected after speech
                    # VAD CHECK!
                    if self.stop_recording_on_voice_deactivity:
                        is_speech = (
                            self._is_silero_speech(data) if self.silero_deactivity_detection
                            else self._is_webrtc_speech(data, True)
                        )
                        if not is_speech:
                            # Voice deactivity was detected, so we start
                            # measuring silence time before stopping recording
                            if self.speech_end_silence_start == 0:
                                self.speech_end_silence_start = time.time()
                        else:
                            self.speech_end_silence_start = 0

                        # Wait for silence to stop recording after speech
                        # post_speech_silence_duration = 0.2 -> 침묵시간이 post_speech_silence_duration보다 길면 STOP!
                        if self.speech_end_silence_start and time.time() - \
                                self.speech_end_silence_start > \
                                self.post_speech_silence_duration:
                            logging.info("voice deactivity detected")
                            # 즉, stop은 VAD CHECK, post_speech_silence_duration 침묵 시간 CHECK
                            # stop() is called. -> recording stop trigger!
                            self.stop()

                if not self.is_recording and was_recording:
                    # Reset after stopping recording to ensure clean state
                    self.stop_recording_on_voice_deactivity = False

                if time.time() - self.silero_check_time > 0.1:
                    self.silero_check_time = 0

                # Handle wake word timeout (waited to long initiating
                # speech after wake word detection)
                if self.wake_word_detect_time and time.time() - \
                        self.wake_word_detect_time > self.wake_word_timeout:

                    self.wake_word_detect_time = 0
                    if self.wakeword_detected and self.on_wakeword_timeout:
                        self.on_wakeword_timeout()
                    self.wakeword_detected = False

                was_recording = self.is_recording

                if self.is_recording:
                    # chunk data가 self.frames에 저장
                    self.frames.append(data)

                    # print(f"recording_audio_data len :{len(self.frames)}\n")

                if not self.is_recording or self.speech_end_silence_start:
                    """
                    매끄러운 전환: 최근 오디오 데이터의 버퍼를 유지함으로써 시스템은 녹음의 여러 상태 사이를 원활하게 전환하여 
                    녹음과 녹음하지 않는 사이를 전환하는 동안 중요한 오디오 정보가 손실되지 않도록 할 수 있습니다.

                    요약
                    이 코드는 음성 시작 시 또는 침묵 감지 시와 같이 녹음 세션의 가장자리 주변에서 오디오 데이터가 캡처되는 방식을 관리하는 데 중요합니다. 
                    이를 통해 시스템이 오디오 데이터를 효과적으로 캡처하고 음성 활동의 존재 여부에 따라 지능적인 결정을 내릴 수 있으며, 이는 실시간 음성-텍스트 시스템에 필수
                    """
                    self.audio_buffer.append(data)
            
        except Exception as e:
            if not self.interrupt_stop_event.is_set():
                logging.error(f"Unhandled exeption in _recording_worker: {e}")
                raise

    def _realtime_worker(self):
        """
        Performs real-time transcription if the feature is enabled.

        The method is responsible transcribing recorded audio frames
          in real-time based on the specified resolution interval.
        The transcribed text is stored in `self.realtime_transcription_text`
          and a callback
        function is invoked with this text if specified.
        """

        try:

            logging.debug('Starting realtime worker')

            # Return immediately if real-time transcription is not enabled
            if not self.enable_realtime_transcription:
                return

            # Continue running as long as the main process is active
            while self.is_running:

                # Check if the recording is active
                if self.is_recording:

                    # Sleep for the duration of the transcription resolution
                    time.sleep(self.realtime_processing_pause)

                    # Convert the buffer frames to a NumPy array
                    audio_array = np.frombuffer(
                        b''.join(self.frames),
                        dtype=np.int16
                        )

                    # Normalize the array to a [-1, 1] range
                    audio_array = audio_array.astype(np.float32) / \
                        INT16_MAX_ABS_VALUE

                    # Perform transcription and assemble the text
                    segments = self.realtime_model_type.transcribe(
                        audio_array,
                        language=self.language if self.language else None,
                        beam_size=self.beam_size_realtime,
                        initial_prompt=self.initial_prompt,
                        suppress_tokens=self.suppress_tokens,
                    )

                    # double check recording state
                    # because it could have changed mid-transcription
                    if self.is_recording and time.time() - \
                            self.recording_start_time > 0.5:

                        logging.debug('Starting realtime transcription')
                        self.realtime_transcription_text = " ".join(
                            seg.text for seg in segments[0]
                        )
                        self.realtime_transcription_text = \
                            self.realtime_transcription_text.strip()

                        self.text_storage.append(
                            self.realtime_transcription_text
                            )

                        # Take the last two texts in storage, if they exist
                        if len(self.text_storage) >= 2:
                            last_two_texts = self.text_storage[-2:]

                            # Find the longest common prefix
                            # between the two texts
                            prefix = os.path.commonprefix(
                                [last_two_texts[0], last_two_texts[1]]
                                )

                            # This prefix is the text that was transcripted
                            # two times in the same way
                            # Store as "safely detected text"
                            if len(prefix) >= \
                                    len(self.realtime_stabilized_safetext):

                                # Only store when longer than the previous
                                # as additional security
                                self.realtime_stabilized_safetext = prefix

                        # Find parts of the stabilized text
                        # in the freshly transcripted text
                        matching_pos = self._find_tail_match_in_text(
                            self.realtime_stabilized_safetext,
                            self.realtime_transcription_text
                            )

                        if matching_pos < 0:
                            if self.realtime_stabilized_safetext:
                                self._on_realtime_transcription_stabilized(
                                    self._preprocess_output(
                                        self.realtime_stabilized_safetext,
                                        True
                                    )
                                )
                            else:
                                self._on_realtime_transcription_stabilized(
                                    self._preprocess_output(
                                        self.realtime_transcription_text,
                                        True
                                    )
                                )
                        else:
                            # We found parts of the stabilized text
                            # in the transcripted text
                            # We now take the stabilized text
                            # and add only the freshly transcripted part to it
                            output_text = self.realtime_stabilized_safetext + \
                                self.realtime_transcription_text[matching_pos:]

                            # This yields us the "left" text part as stabilized
                            # AND at the same time delivers fresh detected
                            # parts on the first run without the need for
                            # two transcriptions
                            self._on_realtime_transcription_stabilized(
                                self._preprocess_output(output_text, True)
                                )

                        # Invoke the callback with the transcribed text
                        self._on_realtime_transcription_update(
                            self._preprocess_output(
                                self.realtime_transcription_text,
                                True
                            )
                        )

                # If not recording, sleep briefly before checking again
                else:
                    time.sleep(TIME_SLEEP)

        except Exception as e:
            logging.error(f"Unhandled exeption in _realtime_worker: {e}")
            raise

    def _is_silero_speech(self, chunk):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        self.silero_working = True
        audio_chunk = np.frombuffer(chunk, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
        vad_prob = self.silero_vad_model(
            torch.from_numpy(audio_chunk),
            SAMPLE_RATE).item()
        is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
        if is_silero_speech_active:
            self.is_silero_speech_active = True
        self.silero_working = False
        return is_silero_speech_active

    def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        # Number of audio frames per millisecond
        frame_length = int(16000 * 0.01)  # for 10ms frame
        num_frames = int(len(chunk) / (2 * frame_length))
        speech_frames = 0

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = chunk[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames_must_be_true:
                    if self.debug_mode:
                        print(f"Speech detected in frame {i + 1}"
                              f" of {num_frames}")
                    return True
        if all_frames_must_be_true:
            if self.debug_mode and speech_frames == num_frames:
                print(f"Speech detected in {speech_frames} of "
                      f"{num_frames} frames")
            elif self.debug_mode:
                print(f"Speech not detected in all {num_frames} frames")
            return speech_frames == num_frames
        else:
            if self.debug_mode:
                print(f"Speech not detected in any of {num_frames} frames")
            return False

    def _check_voice_activity(self, data):
        """
        Initiate check if voice is active based on the provided data.

        Args:
            data: The audio data to be checked for voice activity.
        """
        # 전체 chunk 프레임중에 10ms 프레임에서 음성이 디텍션 되면 True
        self.is_webrtc_speech_active = self._is_webrtc_speech(data)

        # First quick performing check for voice activity using WebRTC
        if self.is_webrtc_speech_active:

            if not self.silero_working:
                self.silero_working = True

                # Run the intensive check in a separate thread
                threading.Thread(
                    target=self._is_silero_speech,
                    args=(data,)).start()

    def _is_voice_active(self):
        """
        Determine if voice is active.

        Returns:
            bool: True if voice is active, False otherwise.
        """
        return self.is_webrtc_speech_active and self.is_silero_speech_active

    def _set_state(self, new_state):
        """
        Update the current state of the recorder and execute
        corresponding state-change callbacks.

        Args:
            new_state (str): The new state to set.

        """
        # Check if the state has actually changed
        if new_state == self.state:
            return

        # Store the current state for later comparison
        old_state = self.state

        # Update to the new state
        self.state = new_state

        # Execute callbacks based on transitioning FROM a particular state
        if old_state == "listening":
            if self.on_vad_detect_stop:
                self.on_vad_detect_stop()
        elif old_state == "wakeword":
            if self.on_wakeword_detection_end:
                self.on_wakeword_detection_end()

        # Execute callbacks based on transitioning TO a particular state
        if new_state == "listening":
            if self.on_vad_detect_start:
                self.on_vad_detect_start() # listening시 실행되는 콜백 함수 
            self._set_spinner("speak now")
            if self.spinner and self.halo:
                self.halo._interval = 250
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start: 
                self.on_wakeword_detection_start() # wakeword시 실행되는 콜백 함수 
            self._set_spinner(f"say {self.wake_words}")
            if self.spinner and self.halo:
                self.halo._interval = 500
        elif new_state == "transcribing":
            if self.on_transcription_start:
                self.on_transcription_start() # transcribing 실행되는 콜백 함수 
            self._set_spinner("transcribing")
            if self.spinner and self.halo:
                self.halo._interval = 50
        elif new_state == "recording":
            self._set_spinner("recording")
            if self.spinner and self.halo:
                self.halo._interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None

    def _set_spinner(self, text):
        """
        Update the spinner's text or create a new
        spinner with the provided text.

        Args:
            text (str): The text to be displayed alongside the spinner.
        """
        if self.spinner:
            # If the Halo spinner doesn't exist, create and start it
            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()
            # If the Halo spinner already exists, just update the text
            else:
                self.halo.text = text

    def _preprocess_output(self, text, preview=False):
        """
        Preprocesses the output text by removing any leading or trailing
        whitespace, converting all whitespace sequences to a single space
        character, and capitalizing the first character of the text.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = re.sub(r'\s+', ' ', text.strip())
        if self.ensure_sentence_starting_uppercase:
            if text:
                text = text[0].upper() + text[1:]

        # Ensure the text ends with a proper punctuation
        # if it ends with an alphanumeric character
        if not preview:
            if self.ensure_sentence_ends_with_period:
                if text and text[-1].isalnum():
                    text += '.'

        return text

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):
        """
        Find the position where the last 'n' characters of text1
        match with a substring in text2.

        This method takes two texts, extracts the last 'n' characters from
        text1 (where 'n' is determined by the variable 'length_of_match'), and
        searches for an occurrence of this substring in text2, starting from
        the end of text2 and moving towards the beginning.

        Parameters:
        - text1 (str): The text containing the substring that we want to find
          in text2.
        - text2 (str): The text in which we want to find the matching
          substring.
        - length_of_match(int): The length of the matching string that we are
          looking for

        Returns:
        int: The position (0-based index) in text2 where the matching
          substring starts. If no match is found or either of the texts is
          too short, returns -1.
        """

        # Check if either of the texts is too short
        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1

        # The end portion of the first text that we want to compare
        target_substring = text1[-length_of_match:]

        # Loop through text2 from right to left
        for i in range(len(text2) - length_of_match + 1):
            # Extract the substring from text2
            # to compare with the target_substring
            current_substring = text2[len(text2) - i - length_of_match:
                                      len(text2) - i]

            # Compare the current_substring with the target_substring
            if current_substring == target_substring:
                # Position in text2 where the match starts
                return len(text2) - i

        return -1

    def _on_realtime_transcription_stabilized(self, text):
        """
        Callback method invoked when the real-time transcription stabilizes.

        This method is called internally when the transcription text is
        considered "stable" meaning it's less likely to change significantly
        with additional audio input. It notifies any registered external
        listener about the stabilized text if recording is still ongoing.
        This is particularly useful for applications that need to display
        live transcription results to users and want to highlight parts of the
        transcription that are less likely to change.

        Args:
            text (str): The stabilized transcription text.
        """
        if self.on_realtime_transcription_stabilized:
            if self.is_recording:
                self.on_realtime_transcription_stabilized(text)

    def _on_realtime_transcription_update(self, text):
        """
        Callback method invoked when there's an update in the real-time
        transcription.

        This method is called internally whenever there's a change in the
        transcription text, notifying any registered external listener about
        the update if recording is still ongoing. This provides a mechanism
        for applications to receive and possibly display live transcription
        updates, which could be partial and still subject to change.

        Args:
            text (str): The updated transcription text.
        """
        if self.on_realtime_transcription_update:
            if self.is_recording:
                self.on_realtime_transcription_update(text)

    def __enter__(self):
        """
        Method to setup the context manager protocol.

        This enables the instance to be used in a `with` statement, ensuring
        proper resource management. When the `with` block is entered, this
        method is automatically called.

        Returns:
            self: The current instance of the class.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Method to define behavior when the context manager protocol exits.

        This is called when exiting the `with` block and ensures that any
        necessary cleanup or resource release processes are executed, such as
        shutting down the system properly.

        Args:
            exc_type (Exception or None): The type of the exception that
              caused the context to be exited, if any.
            exc_value (Exception or None): The exception instance that caused
              the context to be exited, if any.
            traceback (Traceback or None): The traceback corresponding to the
              exception, if any.
        """
        self.shutdown()
