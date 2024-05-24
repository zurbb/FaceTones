from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
import torchaudio
from pydub import AudioSegment
import os
import torch
import io

os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')

class VoiceToVec:
    def __init__(self):
        self.encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")

    def get_embedding(self, mp3_path: str, new_freq: int =16000) -> torch.Tensor:
        audio = AudioSegment.from_mp3(mp3_path)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        signal, fs = torchaudio.load(wav_io)
        signal = signal.mean(dim=0, keepdim=True) # Convert stereo to mono
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=new_freq)(signal)
        embedding = self.encoder.encode_batch(signal)
        return embedding