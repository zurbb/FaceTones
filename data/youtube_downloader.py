from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip , ffmpeg_extract_audio
import numpy as np
import pandas as pd
import os
import sys
import tqdm
import shutil
import coloredlogs, logging
from moviepy.editor import VideoFileClip
from PIL import Image
import threading
import time


TEST_OR_TRAIN = "test" # Change between "train" and "test"
SUBCLIP_DIR = f"data/{TEST_OR_TRAIN}/subclip"
VIDEO_DIR =  f"data/{TEST_OR_TRAIN}/video"
ERROR_FILE =  f"data/{TEST_OR_TRAIN}/errors.txt"
SUCCESS_FILE=  f"data/{TEST_OR_TRAIN}/success.txt"
AUDIO_DIR =  f"data/{TEST_OR_TRAIN}/audio"
DATASET_CSV =  f"data/avspeech_{TEST_OR_TRAIN}.csv"
IMAGE_DIR =  f"data/{TEST_OR_TRAIN}/images"

DOWNLOAD_CHUNK_SIZE = 50
NUM_OF_THREADS = 16
DATA_LIMIT_FOR_TEST = 640
SEMAPHORE = threading.Semaphore(1)

YOUTUBE_ID= "YouTubeID" 
START_SEGMENT = "startSegment"
END_SEGMENT = "endSegment"
X_COORDINATE = "Xcoordinate"
Y_COORDINATE = "Ycoordinate"
UNIQUE_ID = "unique_id"


LOG_EVARY_N = 20
PROCCESSED_COUNTER = 0
COUNTER_LOCK = threading.Lock()
TIME_ZERO = time.time()
# TIME_START_CHUNK

# Disable logging from other libraries
for logger_name in logging.Logger.manager.loggerDict:
    logger2 = logging.getLogger(logger_name)
    logger2.propagate = False

logger = logging.getLogger()
coloredlogs.install()

def _download_and_trim_youtube_video(url: str, full_video_dir: str, full_video_name: str, subclip_dir: str,
                                            subclip_name: str, start_time: float, end_time: float):
    full_video_path = YouTube(url).streams.first().download(full_video_dir, filename=f"{full_video_name}.mp4")
    target_name = os.path.join(subclip_dir, subclip_name)
    # for _ in range(10):
    #     if os.path.exists(full_video_path):
    #         break
    #     time.sleep(1)

    if not os.path.exists(full_video_path):
        raise Exception(f"path not exsit {full_video_path}")
    
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    ffmpeg_extract_subclip(full_video_path, start_time, end_time, targetname=f"{target_name}.mp4")

    # Restore stdout
    sys.stdout = original_stdout
    # ffmpeg_extract_subclip(full_video_path, start_time, end_time, targetname=f"{target_name}.mp4")


def _download_and_save_video_subclips(youtube_id: str, start_time: float, end_time: float, name_to_save:str)->bool:
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



def _extract_audio(video_name: str)->bool:
    """
    Extracts the audio from a given mp4 file and saves it in the target directory.

    Args:
        file_path (str): The path of the mp4 file.
        target_dir (str): The directory where the extracted audio will be saved.

    Returns:
        bool : True if the audio was extracted successfully, False otherwise.
    """
    try:
        video_path = os.path.join(SUBCLIP_DIR, f"{video_name}.mp4")
    
        target_path = os.path.join(AUDIO_DIR, f"{video_name}.mp3")
        # Redirect stdout to null
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        ffmpeg_extract_audio(video_path, target_path)
        # Restore stdout
        sys.stdout = original_stdout
    except Exception as e:
        logger.error(f"Error extracting audio from video {video_name}. error details: {e}")
        return False
    return os.path.exists(target_path)
    

def _extract_images_from_subclip(video_id, x_coord, y_coord, number_of_images=5)->bool:
    """
    Extracts images from a subclip and saves them in the target directory.

    Args:
        clip_id (str): The id of the subclip.

    Returns:
        None
    """
    try:
        subclip_path = os.path.join(SUBCLIP_DIR, f"{video_id}.mp4")
        # target_dir = os.path.join(IMAGE_DIR, video_id)
        # if not os.path.exists(target_dir):
        #     os.mkdir(target_dir)
        clip = VideoFileClip(subclip_path)
        duration = clip.duration
        for i in range(number_of_images):
            frame = clip.get_frame(t=i * duration / number_of_images)
            image_path = os.path.join(IMAGE_DIR, f"{video_id}_{i}.jpg")
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            # crop the image around the face given by x and y coordinates.
            cropped_frame = _crop_image(frame, x_coord, y_coord)
            img = Image.fromarray(cropped_frame)
            img.save(image_path)
    except Exception as e:
        logger.error(f"Error extracting images from video {video_id}. error details: {e}")
        return False
    return os.path.exists(target_dir)


def _crop_image(image: np.ndarray, x_coord: float, y_coord: float):
    width, height = image.shape[1], image.shape[0]
    x_coord = int(x_coord * width)
    y_coord = int(y_coord * height)
    closest_edge_distance = min(x_coord, y_coord, width - x_coord, height - y_coord)
    cropped_image = image[y_coord - closest_edge_distance:y_coord + closest_edge_distance,
                          x_coord - closest_edge_distance:x_coord + closest_edge_distance]
    return cropped_image
    

# def _clean_directories():
#     for file in os.listdir(VIDEO_DIR):
#         if file.endswith(".mp4"):
#             os.remove(os.path.join(VIDEO_DIR, file))
#     logger.info("Clear video directory")

#     for file in os.listdir(SUBCLIP_DIR):
#         if file.endswith(".mp4"):
#             os.remove(os.path.join(SUBCLIP_DIR, file))
#     logger.info("Clear subclip directory")

def clean_file_from_dirs(file_name):
    dirs = [VIDEO_DIR, SUBCLIP_DIR]
    for dir in dirs:
        if os.path.exists(os.path.join(dir, f"{file_name}.mp4")):
            os.remove(os.path.join(dir, f"{file_name}.mp4"))
        
def preprocess_video_chunks(thread_dataset_info_chunks, thread_id):
    for info_chunk in thread_dataset_info_chunks:
        
        success_ids = set()
        fail_ids = set()
        
        for index, row in info_chunk.iterrows():
         
            global PROCCESSED_COUNTER
            with COUNTER_LOCK:
                PROCCESSED_COUNTER += 1
            
                # if PROCCESSED_COUNTER % 20==0:
                #     _clean_directories()
            
            
            youtube_id, start_time, end_time   = row[YOUTUBE_ID], row[START_SEGMENT], row[END_SEGMENT]
            x_coord, y_coord = row[X_COORDINATE], row[Y_COORDINATE]
            
            name_to_save = f"{youtube_id}_{index}"
            if not _download_and_save_video_subclips(youtube_id, start_time, end_time, name_to_save):
                logger.warning(f"Failed to download video {youtube_id}")
                fail_ids.add(name_to_save)
                continue
            if not _extract_audio(name_to_save):
                logger.warning(f"Failed to extract audio from video {youtube_id}")
                fail_ids.add(name_to_save)
                continue
            if not  _extract_images_from_subclip(name_to_save, x_coord, y_coord, number_of_images=1):
                logger.warning(f"Failed to extract images from video {youtube_id}")
                fail_ids.add(name_to_save)
                continue
            success_ids.add(name_to_save)
            clean_file_from_dirs(name_to_save)
            
            with COUNTER_LOCK:
                if PROCCESSED_COUNTER % LOG_EVARY_N == 0:
                    # global TIME_START_CHUNK
                    # logger.info(f"proccessed {LOG_EVARY_N} videos at {round(time.time() - TIME_START_CHUNK,2)}")
                    logger.info(f"Total proccessed {PROCCESSED_COUNTER} videos at {round(time.time() - TIME_ZERO,2)}")
        
        SEMAPHORE.acquire()    
        with open(SUCCESS_FILE, "a") as f:
            for success_id in success_ids:
                f.write(f"{success_id}\n")
                
        with open(ERROR_FILE, "a") as f:
            for fail_id in fail_ids:
                f.write(f"{fail_id}\n")
        SEMAPHORE.release()

            
        
def open_dirs_and_files_if_not_exists():
    dirs = [VIDEO_DIR, SUBCLIP_DIR, AUDIO_DIR, IMAGE_DIR]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    for file in [ERROR_FILE, SUCCESS_FILE]:
        if not os.path.exists(file):
            with open(file, "w") as f:
                pass

if __name__ == "__main__":
    
    open_dirs_and_files_if_not_exists()
    
    dataset_info = pd.read_csv(DATASET_CSV, sep=",")
    logger.info(f"Loaded dataset with {len(dataset_info)} videos. time taken {round(time.time() - TIME_ZERO,2)}")
    dataset_info[UNIQUE_ID] = dataset_info[YOUTUBE_ID] + "_" + dataset_info.index.astype(str)
    
    
    seen = set()
    
    with open(SUCCESS_FILE, "r") as f:
        success = [i.strip() for i in f]
        seen.update(success)
        
    with open(ERROR_FILE, "r") as f:
        errors = [i.strip() for i in f]
        seen.update(errors)
    
    before = len(dataset_info)
    unique_videos = dataset_info.drop_duplicates(subset=[YOUTUBE_ID], keep="first")
    logger.info(f"unique videos: {len(unique_videos)}")
    dataset_info = unique_videos[~unique_videos[UNIQUE_ID].isin(seen)]    
    
    logger.info(f"Removed {len(unique_videos) - len(dataset_info)} already processed videos")
    logger.info(f"Remaining videos: {len(dataset_info)}")
    
    # dataset_info = dataset_info[:DATA_LIMIT_FOR_TEST] # TODO: remove for actual run
    num_of_chunks = np.ceil(len(dataset_info) / DOWNLOAD_CHUNK_SIZE)
    dataset_info_chunks = np.array_split(dataset_info, num_of_chunks)
    
   
    
    threads = []
    
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
    
    
    logger.info(f"Finished processing {len(dataset_info)} videos.\n Time taken {round(time.time() - TIME_ZERO,2)}. average time per video {(time.time() - TIME_ZERO) / len(dataset_info)}")
        
