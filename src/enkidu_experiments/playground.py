# import os
# import sys
# from pathlib import Path

# import sys
# from pathlib import Path #because os.path is old
# # needed for Enkidu
# import torch
# import torchaudio
# from speechbrain.inference import SpeakerRecognition

# #print(Path(__file__).parent.parent.parent / 'dependencies' / 'Enkidu')

# #print(sys.path)

# ENKIDU_PATH = Path(__file__).parent.parent.parent / 'dependencies' / 'Enkidu' #but the file should follow the same structure
# #print(ENKIDU_PATH)
# #print(sys.path.append(str(ENKIDU_PATH))) #need to do this to be able to import Enkidu

# sys.path.append(str(ENKIDU_PATH))
# #print(len(sys.path))
# for path in sys.path:
#     print(path)

# from core import enkidu

# print(dir(enkidu))

# Test 1: WITHOUT .copy()
DEFAULT_CONFIG = {'device': 'cpu', 'noise_level': 0.4}

config1 = DEFAULT_CONFIG  # No copy!
config1['device'] = 'cuda'

print(DEFAULT_CONFIG['device'])  # What do you think this prints?