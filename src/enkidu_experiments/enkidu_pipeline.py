import sys
from pathlib import Path #because os.path is old
# needed for Enkidu
import torch
import torchaudio
from speechbrain.inference import SpeakerRecognition

# GOAL : to be able to import Enkidu we go back to the root of the project
#need to make it more generalize (this is only for my personal)
# ENKIDU_PATH = "/home/student/workspace/nomades_project/dependencies/Enkidu" 
ENKIDU_PATH = Path(__file__).parent.parent.parent / 'dependencies' / 'Enkidu' #but the file should follow the same structure
sys.path.append(str(ENKIDU_PATH)) #add the enkidu path so that I can import it or else it will only have the default
#python prend le premier qu'il prend, il faut mettre en priorite le ENKIDU_PATH
from core import Enkidu
from speechbrain.inference import SpeakerRecognition


class EnkiduPipeline:

    DEFAULT_CONFIG = {
        'device' : 'cpu',
        'sample_rate' : 16000,
        'steps' : 40,
        'alpha' : 0.1,
        'mask_ratio' : 0.3,
        'frame_length' : 120,
        'noise_level' : 0.4,
    }

    def __init__(self):
        self.device = ('cpu')
        self.sample_rate = (16000)

    def prepare_speech:
    
