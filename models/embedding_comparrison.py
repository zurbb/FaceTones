import os
from voice_to_vec import VoiceToVec
import torch
import tqdm
from torch import nn
import eval_lib as lib

def compare_embed_audio_files(model, audio_path, N=100):
    # Load the VoiceToVec model
    audio_files = list(os.listdir(audio_path))[:N]
    signals = []
    # for audio_file in tqdm.tqdm(audio_files, desc="getting signals"):
    #     signals.append(model.get_signals(os.path.join(audio_path, audio_file)))
        
    # longest_signal = max(signals, key=lambda x: x.size(1))
    cosine_sim = lib.cosine_similarity
    prev_voice = None
    success = 0
    for audio_file in audio_files:
        
        audio_file_path = os.path.join(audio_path, audio_file)
        signals = model.get_signals(audio_file_path)
        print(signals.size())
        # reg_embedding = model.get_embedding(signals)
        # print(reg_embedding.size())
        # continue
        # print(f"regular embedding: max - {reg_embedding.max()}, min - {reg_embedding.min()}, mean - {reg_embedding.mean()}")
        # padded_signal = torch.cat([signals, torch.zeros(1, signals.size(1))], dim=0)
        # padded_embedding = model.get_embedding(padded_signal).mean()
        # print(padded_embedding.size())
        # print(f"padded embedding: max - {padded_embedding.max()}, min - {padded_embedding.min()}, mean - {padded_embedding.mean()}")
        # print(cosine_sim(reg_embedding, padded_embedding))
        first_half_embedding = model.get_embedding(signals[:, :signals.size(1)//2]).unsqueeze(0)
        # print(first_half_embedding.size())
        second_half_embedding = model.get_embedding(signals[:, signals.size(1)//2:]).unsqueeze(0)
        # print(second_half_embedding.size())
        two_halves_sim = cosine_sim(first_half_embedding, second_half_embedding)
        print(f"two halves similarity: {two_halves_sim}")

        # reg_embedding = model.get_embedding(signals).unsqueeze(0)
        if prev_voice is not None:
            different_halves_sim = cosine_sim(first_half_embedding, prev_voice)
            print(f"different voices similarity: {different_halves_sim}")
            if two_halves_sim > different_halves_sim:
                success += 1
        prev_voice = first_half_embedding
    print(f"Success rate: {success}/{N}")
        # print(reg_embedding.size())
        # print(f"regular embedding: max - {reg_embedding.max()}, min - {reg_embedding.min()}, mean - {reg_embedding.mean()}")
        # padded_signal = torch.cat([signals, torch.zeros(1, signals.size(1))], dim=0)
        # padded_embedding = model.get_embedding(padded_signal).mean()
        # print(padded_embedding.size())
        # print(f"padded embedding: max - {padded_embedding.max()}, min - {padded_embedding.min()}, mean - {padded_embedding.mean()}")
        # print(cosine_sim(reg_embedding, padded_embedding))
        # if torch.allclose(reg_embedding, padded_embedding, atol=1e-2):
        #     print("True")
        # else:
        #     print("False")

def main():
    model = VoiceToVec()
    AUDIO_DIR = "data/test/audio/"
    compare_embed_audio_files(model, AUDIO_DIR, N=100)

if __name__ == "__main__":
    main()

