# -*- mode: python ; coding: utf-8 -*-

#from PyInstaller.utils.hooks import collect_all

# Collect all files needed by torch
#torch_binaries, torch_datas, torch_hiddenimports = collect_all('torch')

# Add torch-specific collected data to the main analysis
a = Analysis(
    ['Thomas_audio_control_src.py'],
    pathex=[],
    binaries=[],  # Add torch binaries
    datas=[
        ('..\\RealTimeSTT_LEE', 'RealTimeSTT_LEE'), 
        ('..\\realtimestt_gpu_env\\Lib\\site-packages\\pvporcupine\\resources\\keyword_files\\windows', 'pvporcupine/resources/keyword_files/windows'),
        ('..\\tests\\silero_model', 'silero_model'),
        ('..\\tests\\faster_whisper_model', 'faster_whisper_model'),
        ('..\\tests\\recorder_config.yaml', '.'),
        ('..\\tests\\icon', 'icon')
    ],  # Add torch datas
    hiddenimports=[
        'torch', 'openwakeword', 'faster_whisper', 'pvporcupine', 
        'webrtcvad', 'pyaudio', 'halo', 'noisereduce'
    ],  # Add torch hidden imports
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RealTimeSTT_Button_Mode',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='..\\tests\\icon\\megagen_icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RealTimeSTT_Button_Mode',
)
