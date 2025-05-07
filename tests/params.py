import os
import sys
import yaml

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

CONFIG_FILE = resource_path('recorder_config.yaml')

with open(CONFIG_FILE, "r", encoding='UTF8') as file:
    configs = yaml.safe_load(file)

recorder_config = configs['recorder_config']

if recorder_config.get('model'):
    recorder_config['model'] = resource_path(os.path.join('faster_whisper_model', recorder_config['model']))
else:
    raise ValueError("recorder_config['model'] 값이 없습니다.")

if recorder_config.get('silero_model_path'):
    recorder_config['silero_model_path'] = resource_path(recorder_config['silero_model_path'])
else:
    raise ValueError("recorder_config['silero_model_path'] 값이 없습니다.")

communicator_config = configs['communicator_config']

similarity_config = configs['similarity_config']
if similarity_config.get('model_path'):
    similarity_config['model_path'] = resource_path(similarity_config['model_path'])
else:
    print("⚠️ similarity_config['model_path'] 값이 None입니다. 모델 로딩을 건너뜁니다.")

gui_config = configs['gui_default_config']
gui_config['icon_path'] = resource_path(gui_config['icon_path'])
gui_config['ui_file_path'] = resource_path(gui_config['ui_file_path'])

event_flag = configs['event_flag']



"""
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

"""