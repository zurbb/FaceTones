import os
import numpy as np
from pydub import AudioSegment
from torchaudio.transforms import Resample
import io
import torchaudio
import threading


def save_raw_data(input_folder, output_file):

    audio_files = os.listdir(input_folder)

    # Initialize an empty list to store the audio data
    audio_data = []

    # Iterate over each audio file
    for i, file in enumerate(audio_files):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(audio_files)}")
        new_freq = 16000
        file_path = os.path.join(input_folder, file)
        audio = AudioSegment.from_mp3(file_path)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        signal, fs = torchaudio.load(wav_io)
        signal = signal.mean(dim=0, keepdim=True) # Convert stereo to mono
        signal = Resample(orig_freq=fs, new_freq=new_freq)(signal)
        audio_data.append(signal)

    # Convert the audio data list to a numpy array
    audio_array = np.array(audio_data)

    # Save the audio array to a file

    np.save(output_file, audio_array)

if __name__ == '__main__':
    input_folder = "data/test/audio"
    output_file = "data/raw_data/test.npy"
    audio_files = os.listdir(input_folder)
    # Create a list to store the threads
    threads = []

    # Define the number of threads
    num_threads = 4

    # Calculate the number of files per thread
    files_per_thread = len(audio_files) // num_threads

    # Iterate over the number of threads
    for i in range(num_threads):
        # Calculate the start and end indices for each thread
        start_index = i * files_per_thread
        end_index = start_index + files_per_thread

        # Create a thread for each range of files
        thread = threading.Thread(target=save_raw_data, args=(input_folder, output_file, start_index, end_index))
        threads.append(thread)

    # Start all the threads
    for thread in threads:
        thread.start()

    # Wait for all the threads to finish
    for thread in threads:
        thread.join()
    save_thread = threading.Thread(target=save_raw_data, args=(input_folder, output_file))

    # Start the thread
    save_thread.start()

    # Wait for the thread to finish
    save_thread.join()

    # Load the saved audio data
    audio_data = np.load(output_file)
    print(audio_data.shape)


