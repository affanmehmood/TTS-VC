exec(open("./venv/Scripts/activate_this.py").read(),
     {'__file__': "./venv/Scripts/activate_this.py"})

import sys
TTS_PATH = "TTS/"

# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally

import os
import string
import time
import argparse
import json
import scipy.io.wavfile as savwav
import numpy as np
import torch

from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor


from os import listdir
from os.path import isfile, join
from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *

from TTS.tts.utils.speakers import SpeakerManager
from pydub import AudioSegment
import librosa

def compute_spec(ref_file):
    print("ref_file", ref_file)
    y, sr = librosa.load(ref_file, sr=ap.sample_rate)
    spec = ap.spectrogram(y)
    spec = torch.FloatTensor(spec).unsqueeze(0)
    return spec


OUT_PATH = 'out/'

# create output path
os.makedirs(OUT_PATH, exist_ok=True)

# model vars
MODEL_PATH = 'best_model.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = "language_ids.json"
TTS_SPEAKERS = "speakers.json"
USE_CUDA = torch.cuda.is_available()

# load the config
C = load_config(CONFIG_PATH)


# load the audio processor
ap = AudioProcessor(**C.audio)

speaker_embedding = None

C.model_args['d_vector_file'] = TTS_SPEAKERS
C.model_args['use_speaker_encoder_as_loss'] = False

model = setup_model(C)
model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)

cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# remove speaker encoder
model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
  if "speaker_encoder" in key:
    del model_weights[key]

model.load_state_dict(model_weights)


model.eval()

if USE_CUDA:
    model = model.cuda()

# synthesize voice
use_griffin_lim = False

CONFIG_SE_PATH = "config_se.json"
CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"



def main(target_file, driving_file, output_base):
    SE_speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH, encoder_config_path=CONFIG_SE_PATH,
                                        use_cuda=USE_CUDA)

    #target_speaker_base = "Test-Dataset/VCTK/VCTK-Ground-Truth/p347/"
    # target_files = [os.path.join(target_speaker_base, f)  # Convert to
    #                 for f in listdir(target_speaker_base)
    #                 if isfile(join(target_speaker_base, f))]
    #
    # driving_files = list(['p221_005.wav'])  # Convert from
    # driving_file = list(['p221_005.wav'])[0]  # Actual voice + words to convert

    # subprocess.call(['ffmpeg-normalize','-m','-l','-0.1',file])


    target_files = [target_file]
    print("target_files", target_files)
    target_emb = SE_speaker_manager.compute_d_vector_from_clip(target_files)
    target_emb = torch.FloatTensor(target_emb).unsqueeze(0)

    driving_files = [driving_file]
    print("driving_files", driving_files)
    driving_emb = SE_speaker_manager.compute_d_vector_from_clip(driving_files)
    driving_emb = torch.FloatTensor(driving_emb).unsqueeze(0)

    driving_spec = compute_spec(driving_file)
    y_lengths = torch.tensor([driving_spec.size(-1)])
    if USE_CUDA:
        ref_wav_voc, _, _ = model.voice_conversion(driving_spec.cuda(), y_lengths.cuda(), driving_emb.cuda(),
                                                   target_emb.cuda())
        ref_wav_voc = ref_wav_voc.squeeze().cpu().detach().numpy()
    else:
        ref_wav_voc, _, _ = model.voice_conversion(driving_spec, y_lengths, driving_emb, target_emb)
        ref_wav_voc = ref_wav_voc.squeeze().detach().numpy()

    print("Saving Audio after decoder:")
    savwav.write( output_base , ap.sample_rate, ref_wav_voc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple models.')
    parser.add_argument('-input',
                        type=str,
                        help='Input folder path e.g. path/path/', required=True)
    parser.add_argument('-output',
                        type=str,
                        help='Output folder path e.g. path/path/', required=True)

    args = parser.parse_args()
    while True:
        potential_input_files = [f # [os.path.join(args.input, f)
                        for f in listdir(args.input)
                        if isfile(join(args.input, f))]

        stripped_filenames = [filename.split("-")[0] for filename in potential_input_files if "source" in filename]
        for stripped_filename in stripped_filenames:
            main(os.path.join(args.input, stripped_filename + "-target.wav"),
                 os.path.join(args.input, stripped_filename + "-source.wav"),
                 os.path.join(args.output, stripped_filename + "-output.wav"))

        import time
        print("sleeping")
        time.sleep(5)

