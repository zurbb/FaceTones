import os
import random
from PIL import Image
import torch
import numpy as np
from voice_to_vec import VoiceToVec, AbstractAudioEmbedding
from training import cosine_similarity_loss, ImageVoiceClassifier
from data_loader import VOICE_SUFFIX, IMAGE_SUFFIX, ImagesVoicesDataset, transform

def load_image(image_path, dino=True, dataset=None):
    image = Image.open(image_path)
    image = transform(image)
    if dino:
        image = dataset.dino.get_embedding(image)
    return image


def match_voice_to_image(image_path, voice1_path, voice2_path, dataset=None):
    # Load the image and voices
    image = load_image(image_path, dino=True, dataset=dataset)
    voice_to_vec = VoiceToVec()
    # Perform inference:
    # model_path = "image_voice_classifier.pth"
    model_path = "trained_models/dino_31_05/checkpoint_3.pth"

    model = ImageVoiceClassifier(dino=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    with torch.inference_mode():
        model.eval()  # Set the model to evaluation mode
        image_features = model(image.unsqueeze(0))  # Pass the image through the model
    voice_1_siganls = voice_to_vec.get_signals(voice1_path)
    voice_2_siganls = voice_to_vec.get_signals(voice2_path)
    voice_1_features = voice_to_vec.get_embedding(voice_1_siganls).unsqueeze(0)
    voice_2_features = voice_to_vec.get_embedding(voice_2_siganls).unsqueeze(0)
    voice_1_match = cosine_similarity_loss(image_features, voice_1_features)
    print(f"Voice 1 match: {voice_1_match}")
    voice_2_match = cosine_similarity_loss(image_features, voice_2_features)
    print(f"Voice 2 match: {voice_2_match}")

    # Compare the model output and determine the matching voice
    if voice_1_match < voice_2_match:
        matching_voice = 0
    else:
        matching_voice = 1

    return matching_voice

def run_tests(n):
    images_dir = "data/test/images"
    voices_dir = "data/test/audio"
    dataset = ImagesVoicesDataset(
        images_dir, voices_dir, voice_transformer=AbstractAudioEmbedding(), transform=transform, dino_embedding=True)
    true_results = []
    false_results = []
    for i in range(n):
        # Get random image and voices
        audio_filename_true, audio_filename_false = random.sample(dataset.voices, 2)
        true_sample_id = audio_filename_true.split(VOICE_SUFFIX)[0]
        false_sample_id = audio_filename_false.split(VOICE_SUFFIX)[0]
        print(f"Running test {i+1}/{n}. Sample ID: {true_sample_id}, False ID: {false_sample_id}")
        image_filename = f"{true_sample_id}{IMAGE_SUFFIX}"
        image_path = os.path.join(images_dir, image_filename)
        voice1_path = os.path.join(voices_dir, audio_filename_true)
        voice2_path = os.path.join(voices_dir, audio_filename_false)

        # Test the model
        matching_voice = match_voice_to_image(image_path, voice1_path, voice2_path, dataset=dataset)
        if matching_voice == 0:
            true_results.append(true_sample_id)
        else:
            false_results.append(true_sample_id)
        print(f"True: {len(true_results)}, False: {len(false_results)}")
    return true_results, false_results

n = 1000  # Number of tests to run
results = run_tests(n)
print(f"Results: \nTrue: {len(results[0])}\nFalse: {len(results[1])}")