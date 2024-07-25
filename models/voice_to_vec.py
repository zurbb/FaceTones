import os

# os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
# os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), '.cache')
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio
from pydub import AudioSegment
import torch
import io
import coloredlogs, logging
from transformers import AutoProcessor, Wav2Vec2Model
import torch
logger = logging.getLogger()
coloredlogs.install()


class AbstractAudioEmbedding:
    """
    Abstract class for audio embeddings, to be used with the ImagesVoicesDataset.
    This way we can easily switch between different audio embedding models.
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def get_embedding(self, signals: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("get_embedding method must be implemented")
    
    def get_signals(self, mp3_path: str) -> torch.Tensor:
        """
        Load an mp3 file and convert it to a tensor.
        """
        audio = AudioSegment.from_mp3(mp3_path)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        signal, fs = torchaudio.load(wav_io)
        signal = signal.mean(dim=0, keepdim=True)
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=self.sample_rate)(signal)
        return signal

class VoiceToVec(AbstractAudioEmbedding):
    """
    Tool for extracting audio embeddings from voice clips.
    Uses the speechbrain library, and the pretrained x-vector model.
    """
    def __init__(self):
        super().__init__()
        self.encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")

    def get_embedding(self,  signals: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder.encode_batch(signals)
        embedding = embedding.squeeze()
        return embedding

class WavLM(AbstractAudioEmbedding):
    """
    wavlm model for audio embeddings.
    Was not used in the project.
    """
    def __init__(self):
        super().__init__()
        # self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        # self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        self.model = Wav2Vec2Model.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")

    def get_embedding(self,  signals: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(signals.squeeze(), return_tensors="pt", sampling_rate=self.sample_rate)
        with torch.no_grad():
            hidden_states = self.model(**inputs).last_hidden_state
        embedding = hidden_states.mean(dim=1).squeeze()
        return embedding
