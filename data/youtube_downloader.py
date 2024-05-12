from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import pandas as pd
import os
import tqdm
import shutil
import coloredlogs, logging
import os
from moviepy.editor import VideoFileClip
import os

logger = logging.getLogger()
coloredlogs.install()

SUBCLIP_DIR = "data/subclip"
TEMP_VIDEO_DIR = "data/video"
ERROR_FILE = "errors.txt"
AUDIO_DIR_TARGET = "data/audio"
DOWNLOAD_CHUNK_SIZE = 10

dataset_info = pd.read_csv("data/avspeech_train.csv", sep=",")
dataset_info = dataset_info[:2]
num_of_chunks = np.ceil(len(dataset_info) / DOWNLOAD_CHUNK_SIZE)
dataset_info_chunks = np.array_split(dataset_info, num_of_chunks)


def _download_and_trim_youtube_video(url: str, full_video_dir: str, full_video_name: str, subclip_dir: str,
                                            subclip_name: str, start_time: float, end_time: float):
    full_video_path = YouTube(url).streams.first().download(full_video_dir, filename=f"{full_video_name}.mp4")
    target_name = os.path.join(subclip_dir, subclip_name)
    ffmpeg_extract_subclip(full_video_path, start_time, end_time, targetname=f"{target_name}.mp4")


def download_and_save_video_subclips():
    """
    Downloads and saves subclips of YouTube videos based on the provided dataset information.

    This function iterates over the dataset information chunks and downloads subclips of YouTube videos
    based on the start and end segments specified in the dataset. The downloaded subclips are saved in
    the specified subclip directory.

    Any errors encountered during the download process are logged in an error file.

    Returns:
        None
    """
    error_file = open(ERROR_FILE, "w")
    for info_chunk in tqdm.tqdm(dataset_info_chunks,
                            desc=f"Downloading and saving subclips in chunks of {DOWNLOAD_CHUNK_SIZE}",
                            colour="green"):
        logger.info("Deleting all temp files")
        shutil.rmtree(TEMP_VIDEO_DIR)
        os.mkdir(TEMP_VIDEO_DIR)
        for _, row in info_chunk.iterrows():
            try:
                youtube_id = row["YouTubeID"]
                start_time = row["startSegment"]
                end_time = row["endSegment"]
                url = f"https://www.youtube.com/watch?v={youtube_id}"
                _download_and_trim_youtube_video(url=url, full_video_dir=TEMP_VIDEO_DIR,
                                                    full_video_name=youtube_id, subclip_dir=SUBCLIP_DIR,
                                                    subclip_name=f"{youtube_id}_subclip", start_time=start_time,
                                                    end_time=end_time)
            except Exception as e:
                error_file.write(str(e) + "\n")
                continue
    error_file.close()

    num_errors = len(open("errors.txt").readlines())
    num_success = len(os.listdir(SUBCLIP_DIR))
    logger.error(f"Number of errors encountered: {num_errors}")
    logger.error(f"see {ERROR_FILE} for more details")
    logger.info(f"Number of successful subclips: {num_success}")

download_and_save_video_subclips()

def extract_audio(video_path: str, audio_dir_target: str):
    """
    Extracts the audio from a video file and saves it as a WAV file.

    Args:
        video_path (str): Path to the video file.
        audio_dir_target (str): Directory to save the extracted audio file.

    Returns:
        None
    """

    video_name = os.path.basename(video_path)
    audio_name = os.path.splitext(video_name)[0] + ".wav"
    audio_path = os.path.join(audio_dir_target, audio_name)

    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, codec="pcm_s16le")

    video_clip.close()
    audio_clip.close()

# Get the list of files in the subclip directory
subclip_files = os.listdir(SUBCLIP_DIR)

# Iterate over each file in the subclip directory
for file in tqdm.tqdm(subclip_files, desc="Extracting audio from subclips", colour="green"):
    # Get the full path of the file
    file_path = os.path.join(SUBCLIP_DIR, file)
    
    # Call the extract_audio function on the file
    extract_audio(file_path, AUDIO_DIR_TARGET)
