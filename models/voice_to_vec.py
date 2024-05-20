from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
import torchaudio
from pydub import AudioSegment
import os
import torch

# plsease make sure to install ffmpeg
# sudo apt-get install ffmpeg -y

class Voice2Vec:
    def __init__(self):
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

    def embed(self, voice_path) -> torch.Tensor:
        """
        Embeds the given voice file into a torch.Size([1, 512]) embedding

        Parameters:
            voice_path (str): The path to the voice file.

        Returns:
            torch.Tensor: torch.Size([1, 512]) embedding.
        """
        if not os.path.exists(voice_path):
            raise ValueError(f"The provided voice path: {voice_path} does not exist.")
        signal, fs = torchaudio.load(voice_path)
        embeddings = self.classifier.encode_batch(signal)
        return embeddings
