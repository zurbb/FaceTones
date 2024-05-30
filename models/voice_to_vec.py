import os

os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), '.cache')
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio
from pydub import AudioSegment
import torch
import io
import coloredlogs, logging
from transformers import AutoProcessor, WavLMModel
import torch
logger = logging.getLogger()
coloredlogs.install()


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
        embedding = embedding.squeeze()
        return embedding

class WavLM:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = WavLMModel.from_pretrained("facebook/wav2vec2-base-960h")
    def get_embedding(self, mp3_path: str,new_freq: int =16000) -> torch.Tensor:
        audio = AudioSegment.from_mp3(mp3_path)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        signal, fs = torchaudio.load(wav_io)
        signal = signal.mean(dim=0, keepdim=True) # Convert stereo to mono
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=new_freq)(signal)
        input_values = self.processor(signal, return_tensors="pt", sampling_rate=new_freq).input_values
        with torch.no_grad():
            hidden_states = self.model(**input_values).last_hidden_state
        embedding = hidden_states.mean(dim=1).squeeze()
        return embedding
    
    # Create instances of the classes

wav_lm = WavLM()

# Get the embedding using WavLM
mp3_path = "data/test/audio/-H9Ab6pveJU_82005.mp3"
embedding = wav_lm.get_embedding(mp3_path)

# Print the embedding
print(embedding)