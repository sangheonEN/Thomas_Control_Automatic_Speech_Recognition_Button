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