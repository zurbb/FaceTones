from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip , ffmpeg_extract_audio
import numpy as np
import pandas as pd
import os
import tqdm
import shutil
import coloredlogs, logging
from moviepy.editor import VideoFileClip
from PIL import Image
import threading
import time

SUBCLIP_DIR = "data/subclip"
VIDEO_DIR = "data/video"
ERROR_FILE = "data/errors.txt"
SUCCESS_FILE= "data/success.txt"
AUDIO_DIR = "data/audio"
DATASET_CSV = "data/avspeech_test.csv"
IMAGE_DIR = "data/images"
DOWNLOAD_CHUNK_SIZE = 10
NUM_OF_THREADS = 20
SEMAPHORE = threading.Semaphore(1)

YOUTUBE_ID= "YouTubeID" 
START_SEGMENT = "startSegment"
END_SEGMENT = "endSegment"
X_COORDINATE = "Xcoordinate"
Y_COORDINATE = "Ycoordinate"
UNIQUE_ID = "unique_id"


logger = logging.getLogger()
coloredlogs.install()

def _download_and_trim_youtube_video(url: str, full_video_dir: str, full_video_name: str, subclip_dir: str,
                                            subclip_name: str, start_time: float, end_time: float):
    full_video_path = YouTube(url).streams.first().download(full_video_dir, filename=f"{full_video_name}.mp4")
    target_name = os.path.join(subclip_dir, subclip_name)
    ffmpeg_extract_subclip(full_video_path, start_time, end_time, targetname=f"{target_name}.mp4")


def download_and_save_video_subclips(youtube_id: str, start_time: float, end_time: float, name_to_save:str)->bool:
    """
    
    Downloads and saves  YouTube video.

    Args:
        youtube_id (str): The YouTube video ID.
        start_time (float): The start time of the subclip in seconds.
        end_time (float): The end time of the subclip in seconds.
    Args:
        youtube_id (str): The YouTube video ID.
        start_time (float): The start time of the subclip in seconds.
        end_time (float): The end time of the subclip in seconds.
    
    Returns: bool : True if the video was downloaded successfully, False otherwise.
    """
    try:
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        _download_and_trim_youtube_video(url=url, full_video_dir=VIDEO_DIR,
                                            full_video_name=name_to_save, subclip_dir=SUBCLIP_DIR,
                                            subclip_name=name_to_save, start_time=start_time,
                                            end_time=end_time)
        return True
    except Exception as e:
        logger.error(f"Error downloading video {youtube_id}. error details: {e}")
        return False



def extract_audio(video_name: str)->bool:
    """
    Extracts the audio from a given mp4 file and saves it in the target directory.

    Args:
        file_path (str): The path of the mp4 file.
        target_dir (str): The directory where the extracted audio will be saved.

    Returns:
        bool : True if the audio was extracted successfully, False otherwise.
    """
    video_path = os.path.join(SUBCLIP_DIR, f"{video_name}.mp4")
   
    target_path = os.path.join(AUDIO_DIR, f"{video_name}.mp3")
    ffmpeg_extract_audio(video_path, target_path)
    return os.path.exists(target_path)
    

    

        



def extract_images_from_subclip(video_id, x_coord, y_coord, number_of_images=5)->bool:
    """
    Extracts images from a subclip and saves them in the target directory.

    Args:
        clip_id (str): The id of the subclip.

    Returns:
        None
    """
    subclip_path = os.path.join(SUBCLIP_DIR, f"{video_id}.mp4")
    target_dir = os.path.join(IMAGE_DIR, video_id)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    clip = VideoFileClip(subclip_path)
    duration = clip.duration
    for i in range(number_of_images):
        frame = clip.get_frame(t=i * duration / number_of_images)
        image_path = os.path.join(target_dir, f"{video_id}_{i}.jpg")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        # crop the image around the face given by x and y coordinates.
        cropped_frame = crop_image(frame, x_coord, y_coord)
        img = Image.fromarray(cropped_frame)
        img.save(image_path)
    return os.path.exists(target_dir)


def crop_image(image: np.ndarray, x_coord: float, y_coord: float):
    width, height = image.shape[1], image.shape[0]
    x_coord = int(x_coord * width)
    y_coord = int(y_coord * height)
    closest_edge_distance = min(x_coord, y_coord, width - x_coord, height - y_coord)
    cropped_image = image[y_coord - closest_edge_distance:y_coord + closest_edge_distance,
                          x_coord - closest_edge_distance:x_coord + closest_edge_distance]
    return cropped_image
    
def get_relative_path(path: str):
    return os.path.join(os.getcwd(), path)


def preprocess_video_chunks(thread_dataset_info_chunks, thread_id):
    proccessed = 0
    for info_chunk in thread_dataset_info_chunks:
        if thread_id == 0:
            SEMAPHORE.acquire()
            logger.info("Deleting all temp files")
            shutil.rmtree(VIDEO_DIR, ignore_errors=True)
            os.mkdir(VIDEO_DIR) if not os.path.exists(VIDEO_DIR) else None

            shutil.rmtree(SUBCLIP_DIR, ignore_errors=True)
            os.mkdir(SUBCLIP_DIR) if not os.path.exists(SUBCLIP_DIR) else None
            logger.info("Deleted all temp files")
            SEMAPHORE.release()
            
        success_ids = []
        fail_ids = []
        
        for index, row in info_chunk.iterrows():
            
            youtube_id, start_time, end_time   = row[YOUTUBE_ID], row[START_SEGMENT], row[END_SEGMENT]
            x_coord, y_coord = row[X_COORDINATE], row[Y_COORDINATE]
            
            name_to_save = f"{youtube_id}_{index}"
            if not download_and_save_video_subclips(youtube_id, start_time, end_time, name_to_save):
                logger.warning(f"Failed to download video {youtube_id}")
                fail_ids.append(name_to_save)
                continue
            if not extract_audio(name_to_save):
                logger.warning(f"Failed to extract audio from video {youtube_id}")
                fail_ids.append(name_to_save)
                continue
            if not  extract_images_from_subclip(name_to_save, x_coord, y_coord, number_of_images=1):
                logger.warning(f"Failed to extract images from video {youtube_id}")
                fail_ids.append(name_to_save)
                continue
            success_ids.append(name_to_save)
        
        SEMAPHORE.acquire()    
        with open(SUCCESS_FILE, "a") as f:
            for success_id in success_ids:
                f.write(f"{success_id}\n")
                
        with open(ERROR_FILE, "a") as f:
            for fail_id in fail_ids:
                f.write(f"{fail_id}\n")
        SEMAPHORE.release()

        proccessed += DOWNLOAD_CHUNK_SIZE
        logger.info(f"Thread {thread_id} processing {proccessed} videos at {time.time() - time_zero}")


if __name__ == "__main__":
    
    time_zero = time.time()

    
    dataset_info = pd.read_csv(get_relative_path(DATASET_CSV), sep=",")
    logger.info(f"Loaded dataset with {len(dataset_info)} videos. time taken {time.time() - time_zero}")
    dataset_info[UNIQUE_ID] = dataset_info[YOUTUBE_ID] + "_" + dataset_info.index.astype(str)
    
    seen = set()
    
    with open(SUCCESS_FILE, "r") as f:
        success = [i.strip() for i in f]
        seen.update(success)
        
    with open(ERROR_FILE, "r") as f:
        errors = [i.strip() for i in f]
        seen.update(errors)
    
    before = len(dataset_info)
    dataset_info = dataset_info[~dataset_info[UNIQUE_ID].isin(seen)]    
    
    logger.info(f"Removed {before - len(dataset_info)} already processed videos")
    
    dataset_info = dataset_info[:200]
    num_of_chunks = np.ceil(len(dataset_info) / DOWNLOAD_CHUNK_SIZE)
    dataset_info_chunks = np.array_split(dataset_info, num_of_chunks)
    
    threads = []
    # Calculate the number of chunks each thread will handle
    chunks_per_thread = len(dataset_info_chunks) // NUM_OF_THREADS 
    for i in range(NUM_OF_THREADS):
        start_index = i * chunks_per_thread
        end_index = start_index + chunks_per_thread  if i < NUM_OF_THREADS - 1 else len(dataset_info_chunks)
        thread = threading.Thread(target=preprocess_video_chunks, args=(dataset_info_chunks[start_index:end_index],i))
        threads.append(thread)
        thread.start() 

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    
    
    logger.info(f"Finished processing {len(dataset_info)} videos.\n Time taken {time.time() - time_zero}. average time per video {(time.time() - time_zero) / len(dataset_info)}")
        
