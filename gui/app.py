import base64
import datetime
import time
import streamlit as st
import random
import os
import sys
import gspread
from google.oauth2.service_account import Credentials
import uuid


ROOT_DIR = os.getcwd()
sys.path.append(os.path.abspath(ROOT_DIR))

from gui.gui_api import GuiBackend

# Google Sheets API
def log_results_to_google_sheet(data):
        client = st.session_state["client"]
        # Open the Google Sheet by its name
        sheet = client.open('FaceTones Log').sheet1

        # Append data to the sheet
        sheet.append_row(data)


# Game State Management
if 'score' not in st.session_state:
    st.session_state['score'] = {'player': 0, 'model': 0, 'turn': 0}

# Define the scope for Google Sheets API
scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

# Authenticate using the credentials file
credentials = Credentials.from_service_account_info(st.secrets['google_auth'], scopes=scope)
st.session_state["client"] = gspread.authorize(credentials)
if 'uuid' not in st.session_state:
    st.session_state['uuid'] = str(uuid.uuid4())

def reset_game():
    """
    Resets the game score and turn count to zero and reruns the app.
    """
    st.session_state['score'] = {'player': 0, 'model': 0, 'turn': 0}
    st.rerun()

def play_audio(file_path):
    """
    Check if audio file exists and then play it.
    """
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
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 class='title'>FaceTones Game</h1>", unsafe_allow_html=True)

# Caching the model instance
@st.cache_resource
def load_model_and_data(dificulty_level):
    print("Loading the model and data...")
    x = GuiBackend()
    data_generator = x.getImagesAndVoice(dificulty_level)
    print("Model and data loaded.")
    return data_generator


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
    """
    Prepares and advances the game to the next turn.

    This function is responsible for incrementing the turn count in the session state and selecting the next set 
    of items (images and voice clip) for the current turn. It also resets the player's choice and the reveal state.
    """
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
        st.rerun()

if 'game_started' not in st.session_state:
    st.session_state['game_started'] = False


def play_turn():
    """
    Executes the logic for a single turn in the game.

    During a turn, this function displays the current items (images and voice clip) to the user and collects their input (choice).
    It then evaluates the user's choice against the correct answer, updates the score accordingly,
    and provides feedback to the user about the correctness of their guess.
    It does the same for the model's choice, and displays the model's choice to the user.
    """
    with st.sidebar:
        st.write(f"Turn: {st.session_state['score']['turn']} 🔄")
        st.write(f"Player Score: {st.session_state['score']['player']} 🎯")
        st.write(f"Model Score: {st.session_state['score']['model']} 🤖") 
    if 'game_data' in st.session_state:
        true_image = st.session_state['game_data']['true_image_path']
        false_image = st.session_state['game_data']['false_image_path']
        true_voice = st.session_state['game_data']['true_voice_path']
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
                if not st.session_state['reveal']:
                    if player_choice == true_image:
                        st.session_state['score']['player'] += 1
                    if model_choice == true_image:
                        st.session_state['score']['model'] += 1
                st.session_state['reveal'] = True

                if player_choice == true_image and model_choice == true_image:
                    result_message = "<div class='result' style='color: green; font-size: 20px;'><strong>🎉🎉 Both Won!</strong></div>"
                elif player_choice == true_image and model_choice != true_image:
                    result_message = "<div class='result' style='color: blue; font-size: 20px;'><strong>🎉 You Won!</strong></div>"
                elif player_choice != true_image and model_choice == true_image:
                    result_message = "<div class='result' style='color: red; font-size: 20px;'><strong>😢 Model Won!</strong></div>"
                else:
                    result_message = "<div class='result' style='color: gray; font-size: 20px;'><strong>😢 Both Wrong!</strong></div>"

                st.markdown(result_message, unsafe_allow_html=True)
                true_image_id = true_image.split("/")[-1].split(".")[0][:-2]
                false_image_id = false_image.split("/")[-1].split(".")[0][:-2]
                player_choice_id = player_choice.split("/")[-1].split(".")[0][:-2]
                model_choice_id = model_choice.split("/")[-1].split(".")[0][:-2]
                log_results_to_google_sheet([st.session_state['uuid'], true_image_id, false_image_id, model_choice_id, player_choice_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

                # Automatically proceed to the next turn after 2 seconds
                time.sleep(2)
                next_turn()
                st.rerun()    

    if st.button("Reset Game"):
        st.session_state['game_started'] = False
        reset_game()

if not st.session_state['game_started']:
    st.session_state['difficulty_level'] = st.slider(label="Select Difficulty Level:", min_value=1, max_value=5, value=3)
    if st.button("Start Game", key='start_button'):
        st.session_state['gui_backend'] = load_model_and_data(st.session_state['difficulty_level'])
        st.session_state['game_started'] = True
        print("starting game...")
        next_turn()

else:
    play_turn()