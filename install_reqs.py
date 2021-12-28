import subprocess
import sys
import os

exec(open("./venv/Scripts/activate_this.py").read(),
     {'__file__': "./venv/Scripts/activate_this.py"})

os.system('pip install -q -e TTS/')
os.system('pip install -q torchaudio==0.9.0')
