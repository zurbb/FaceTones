from speechbrain.inference.speaker import EncoderClassifier


classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
signal, fs = torchaudio.load(dst)
embeddings = classifier.encode_batch(signal)

