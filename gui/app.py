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
st.markdown("<h1 class='title'>FaceTones Game</h1>", unsafe_allow_html=True)

# Caching the model instance
@st.cache_resource
def load_model_and_data(same_gender):
    print("Loading the model and data...")
    x = GuiBackend()
    data_generator = x.getImagesAndVoice(same_gender=same_gender)
    data_generator = x.getImagesAndVoice()
    if 'gui_backend' not in st.session_state:
        st.session_state['gui_backend'] = data_generator
    if 'game_data' in st.session_state:
        del st.session_state['game_data']
    print("Model and data loaded.")
    return data_generator
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
            if st.session_state['difficulty_level'] == "Easy":
                same_gender = False
            else:
                same_gender= True
            st.session_state['gui_backend'] = load_model_and_data(same_gender)
            true_image, false_image, true_voice, true_similarity, false_similarity = next(gui_backend)
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

if 'game_started' not in st.session_state:
    st.session_state['game_started'] = False

if not st.session_state['game_started']:
    st.session_state['difficulty_level'] = st.radio("Select Difficulty Level:", ("Easy", "Hard"))
    # BUG: The start button disappears after only after the first turn
    if st.button("Start Game", key='start_button'):
        st.session_state['game_started'] = True
        if st.session_state['difficulty_level'] == "Easy":
            same_gender = False
        else:
            same_gender= True
        st.session_state['gui_backend'] = load_model_and_data(same_gender=same_gender)
        next_turn()

else: # st.session_state['game_started']:
    with st.sidebar:
        st.write(f"Turn: {st.session_state['score']['turn']} 🔄")
        st.write(f"Player Score: {st.session_state['score']['player']} 🎯")
        st.write(f"Model Score: {st.session_state['score']['model']} 🤖") 
    if 'game_data' in st.session_state:
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
            if st.button("Reveal Answer", use_container_width=True, key='reveal_button'):
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

                if player_choice == true_image and model_choice == true_image:
                    result_message = "<div class='result' style='color: green; font-size: 20px;'><strong>🎉🎉 Both Won!</strong></div>"
                elif player_choice == true_image and model_choice != true_image:
                    result_message = "<div class='result' style='color: blue; font-size: 20px;'><strong>🎉 You Won!</strong></div>"
                elif player_choice != true_image and model_choice == true_image:
                    result_message = "<div class='result' style='color: red; font-size: 20px;'><strong>😢 Model Won!</strong></div>"
                else:
                    result_message = "<div class='result' style='color: gray; font-size: 20px;'><strong>😢 Both Wrong!</strong></div>"

                st.markdown(result_message, unsafe_allow_html=True)

                # Automatically proceed to the next turn after 2 seconds
                time.sleep(2)
                next_turn()
                st.rerun()    

    if st.button("End Game"):
        reset_game()
        st.session_state['game_started'] = False

    if st.button("Reset Game"):
        reset_game()