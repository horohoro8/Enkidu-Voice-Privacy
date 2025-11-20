"""
Test script for Enkidu
"""
import torch
import torchaudio
from speechbrain.inference import SpeakerRecognition

from wavdataset import WaveformPrivacyDataset
from core import Enkidu

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

wave_dataset = WaveformPrivacyDataset(
    dataset_dir='/home/student/workspace/nomades_project/data/LibriSpeech/train-clean-100/19',
    sample_rate=16000,
    mono=True,
    wav_format='flac',
    transforms=None,
)

# just test with 40 samples
sample_list = wave_dataset.get_speaker_samples(0)[:20]

waveform_encrypted_model = Enkidu(
    model=SpeakerRecognition.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={"device": device}),
    steps=40,
    alpha=0.1,
    mask_ratio=0.3,
    frame_length=120,
    noise_level=0.4,
    device=device
)

noise_real, noise_imag = waveform_encrypted_model(sample_list)

# # the voice you want to encrypt
# # assume a 10s voice
# benign_voice = torch.randn(1, 160000)
# torchaudio.save('/home/student/workspace/nomades_project/dependencies/Enkidu/benign_voice.wav', benign_voice, 16000)

# the voice you want to encrypt: use a LibriSpeech .flac
benign_voice_path = "/home/student/workspace/nomades_project/data/LibriSpeech/train-clean-100/2391/145015/2391-145015-0001.flac"  # <-- change to an existing file

benign_voice, sr = torchaudio.load(benign_voice_path)   # shape: (channels, num_frames)

# ensure mono (LibriSpeech is mono, but this is safe)
if benign_voice.size(0) > 1:
    benign_voice = benign_voice.mean(dim=0, keepdim=True)

# ensure 16 kHz (LibriSpeech is 16k already, but this keeps things robust)
target_sr = 16000
if sr != target_sr:
    benign_voice = torchaudio.functional.resample(benign_voice, sr, target_sr)
    sr = target_sr

# # optionally limit to 10 seconds (like the comment suggests)
# max_len = target_sr * 10
# benign_voice = benign_voice[..., :max_len]

# move to same device as the model
benign_voice = benign_voice.to(device)

# save the benign (unencrypted) voice as flac for reference
torchaudio.save('/home/student/workspace/nomades_project/dependencies/Enkidu/benign_voice.flac', benign_voice.cpu(), sr)


encrypted_voice = waveform_encrypted_model.add_noise(
    benign_voice,
    noise_real,
    noise_imag,
    mask_ratio=1,
    random_offset=False,
    noise_smooth=True
)

print(encrypted_voice)
print('done')