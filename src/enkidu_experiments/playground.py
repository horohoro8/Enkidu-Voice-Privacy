import os
import sys
from pathlib import Path

#print(Path(__file__).parent.parent.parent / 'dependencies' / 'Enkidu')

#print(sys.path)

ENKIDU_PATH = Path(__file__).parent.parent.parent / 'dependencies' / 'Enkidu' #but the file should follow the same structure
#print(ENKIDU_PATH)
#print(sys.path.append(str(ENKIDU_PATH))) #need to do this to be able to import Enkidu

sys.path.append(str(ENKIDU_PATH))
#print(len(sys.path))
for path in sys.path:
    print(path)

from core import enkidu

print(dir(enkidu))