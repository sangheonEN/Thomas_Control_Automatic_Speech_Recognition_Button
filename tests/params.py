import os
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recorder_config.yaml')
base_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    
# config load
with open(CONFIG_FILE, "r", encoding='UTF8') as file:
    configs = yaml.safe_load(file)
    
recorder_config = configs['recorder_config']
recorder_config['model'] = os.path.join(base_dir_path, 'faster_whisper_model', recorder_config['model'])
recorder_config['silero_model_path'] = os.path.join(base_dir_path, recorder_config['silero_model_path'])
communicator_config = configs['communicator_config']
similarity_config = configs['similarity_config']
gui_config = configs['gui_default_config']
gui_config['icon_path'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), gui_config['icon_path'])
gui_config['ui_file_path'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), gui_config['ui_file_path'])
event_flag = configs['event_flag']

total_cer = 0.
total_process_time = 0.
total_similarity = 0.
max_similarity_threshold = 0.7

core_text_list = ["환자분", "성함이", "어떻게", "되세요.", "되세요?",
                  "안녕하세요.",
                  "입을", "벌려보세요.", "벌려", "여세요.", "열어주세요.", "아", "하세요.", "벌려보시겠어요?", "벌려보시겠어요.", "더", "크게", "아하세요.",
                  "불편하세요.", "불편하세요?", "어디가",
                  "진료", "시작하겠습니다.",
                  "통증이", "동증이", "느껴진다면", "왼쪽", "팔을", "들어주세요.",
                  "느껴지시나요.", "느껴지시나요?",
                  "다시", "시작해도", "될까요.",
                  "끝났습니다.", "수고하셨습니다.",
                  "네", "알겠습니다."]

reference_texts = [
    "환자분 안녕하세요",
    "환자분 안녕하세요",
    "환자분 안녕하세요",
    "환자분 성함이 어떻게 되세요",
    "환자분 성함이 어떻게 되세요",
    "환자분 성함이 어떻게 되세요",
    "어디가 불편하세요",
    "어디가 불편하세요",
    "어디가 불편하세요",
    "환자분 진료 시작하겠습니다",
    "환자분 진료 시작하겠습니다",
    "환자분 진료 시작하겠습니다",
    "환자분 아 하세요",
    "환자분 아 하세요",
    "환자분 아 하세요",
    "환자분 입을 더 크게 벌려보시겠어요",
    "환자분 입을 더 크게 벌려보시겠어요",
    "환자분 입을 더 크게 벌려보시겠어요",
    "통증이 느껴진다면 왼쪽 팔을 들어주세요",
    "통증이 느껴진다면 왼쪽 팔을 들어주세요",
    "통증이 느껴진다면 왼쪽 팔을 들어주세요",
    "환자분 통증이 느껴지시나요",
    "환자분 통증이 느껴지시나요",
    "환자분 통증이 느껴지시나요",
    "환자분 다시 진료 시작해도 될까요",
    "환자분 다시 진료 시작해도 될까요",
    "환자분 다시 진료 시작해도 될까요",
    "환자분 진료 끝났습니다. 수고하셨습니다",
    "환자분 진료 끝났습니다. 수고하셨습니다",
    "환자분 진료 끝났습니다. 수고하셨습니다"
]