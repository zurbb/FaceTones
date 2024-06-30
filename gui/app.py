import base64
import time
import streamlit as st
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play
import random
import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(os.path.abspath(ROOT_DIR))

from gui.gui_api import GuiBackend



# Game State Management
if 'score' not in st.session_state:
    st.session_state['score'] = {'player': 0, 'model': 0, 'turn': 0}

def reset_game():
    st.session_state['score'] = {'player': 0, 'model': 0, 'turn': 0}
    st.session_state['gui_backend'] = load_model_and_data()
    if 'game_data' in st.session_state:
        del st.session_state['game_data']

def play_audio(file_path):
    # Check if audio file exists and then play it
    try:
        audio_file = open(file_path, 'rb')
        audio_bytes = audio_file.read()
        audio_file.close()
    except FileNotFoundError:
        st.error("Audio file not found.") 
    except Exception as e:
        st.error(f"An error occurred while trying to read the audio file:\n {e}")

    # Create a base64 audio player and embed it in HTML
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    

# GUI Components
st.title("FaceTones Game")

# Caching the model instance
@st.cache_resource
def load_model_and_data():
    print("Loading the model and data...")
    x = GuiBackend()
    data_generator = x.getImagesAndVoice()
    if 'gui_backend' not in st.session_state:
        st.session_state['gui_backend'] = data_generator
    if 'game_data' in st.session_state:
        del st.session_state['game_data']
    print("Model and data loaded.")
load_model_and_data()


# Initialize game-related states
if 'true_image_path' not in st.session_state:
    st.session_state.true_image_path = ""
if 'false_image_path' not in st.session_state:
    st.session_state.false_image_path = ""
if 'true_voice_path' not in st.session_state:
    st.session_state.true_voice_path = ""
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = ""
if 'true_first' not in st.session_state:
    st.session_state.true_first = False

def next_turn():
    if 'gui_backend' in st.session_state:
        gui_backend = st.session_state['gui_backend']
        try:
            true_image, false_image, true_voice, true_similarity, false_similarity = next(gui_backend)
        except StopIteration:
            load_model_and_data()
        st.session_state['game_data'] = {
            'true_image_path': true_image,
            'false_image_path': false_image,
            'true_voice_path': true_voice,
            'true_similarity': true_similarity,
            'false_similarity': false_similarity,
            'model_choice': true_image if true_similarity > false_similarity else false_image
        }
        if 'player_choice' in st.session_state:
            del st.session_state['player_choice']
        st.session_state['score']['turn'] += 1
        st.session_state['true_first'] = random.choice([True, False])
        st.session_state.reveal = False

if st.button("Start Game"):
    load_model_and_data()
    next_turn()

if 'game_data' in st.session_state:
    print("started game")
    st.write(f"Turn: {st.session_state['score']['turn']}")
    true_image = st.session_state['game_data']['true_image_path']
    false_image = st.session_state['game_data']['false_image_path']
    true_voice = st.session_state['game_data']['true_voice_path']
    true_similarity = st.session_state['game_data']['true_similarity']
    false_similarity = st.session_state['game_data']['false_similarity']
    model_choice = st.session_state['game_data']['model_choice']
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state['true_first']:
            st.image(true_image, caption='Image 1', use_column_width=True)
            st.session_state['other_image'] = false_image
            if st.button("Choose Image 1"):
                st.session_state['player_choice'] = true_image
                st.session_state['player_button_press'] = 1

        else:
            st.image(false_image, caption='Image 1', use_column_width=True)
            st.session_state['other_image'] = true_image
            if st.button("Choose Image 1"):
                st.session_state['player_choice'] = false_image
                st.session_state['player_button_press'] = 1

    with col2:
        other_image = st.session_state['other_image']
        st.image(other_image, caption='Image 2', use_column_width=True)
        if st.button("Choose Image 2"):
            st.session_state['player_choice'] = other_image
            st.session_state['player_button_press'] = 2
    
    if st.button("Play Voice"):
        play_audio(true_voice)

    if 'player_choice' in st.session_state:
        player_choice = st.session_state['player_choice']
        player_button_press = st.session_state['player_button_press']
        st.markdown(  # player's choice
        f"""
        <div style="text-align: center;">
            <strong>Your choice:</strong> {'Image 1' if player_button_press == 1 else 'Image 2'}
        </div>
        """,
        unsafe_allow_html=True
        )
        models_choice_text = 'Image 1' if (model_choice == true_image and \
                                                   st.session_state.true_first) or \
                                                     (model_choice != other_image and \
                                                       not st.session_state.true_first) else 'Image 2'
        st.markdown(  # model's choice
        f"""
        <div style="text-align: center;">
            <strong>Model's choice:</strong> {models_choice_text}
        </div>
        """,
        unsafe_allow_html=True
        )
        if st.button("Reveal Answer", use_container_width=True):
            st.markdown(  # Answer
            f"""
            <div style="text-align: center;">
                <strong>Correct Image:</strong> {'Image 1' if st.session_state['true_first'] else 'Image 2'}
            </div>
            """,
            unsafe_allow_html=True
            )
            if player_choice == true_image:
                st.session_state['score']['player'] += 1
            if model_choice == true_image:
                st.session_state['score']['model'] += 1
            st.write(f"Scores - Player: {st.session_state['score']['player']}, FaceTones: {st.session_state['score']['model']}")
            # Automatically proceed to the next turn after 2 seconds
            time.sleep(3)
            next_turn()
            st.rerun()
    
    if st.button("Reset Game"):
        reset_game()

st.write(f"Current Scores - Player: {st.session_state['score']['player']}, FaceTones: {st.session_state['score']['model']}")
