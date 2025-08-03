# ALIA/utils/speech_io.py

import pyttsx3
import speech_recognition as sr
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# --- Voice Customization Code ---
voices = engine.getProperty('voices')
print("Available voices:")
for i, voice in enumerate(voices):
    print(f"  Voice {i}: ID={voice.id}, Name={voice.name}, Gender={voice.gender}")

feminine_voice_id = None
for voice in voices:
    if "zira" in voice.name.lower():
        feminine_voice_id = voice.id
        break

if feminine_voice_id:
    engine.setProperty('voice', feminine_voice_id)
    print("Voice set to a feminine tone.")
else:
    print("Could not find a feminine voice. Using default.")
# -------------------------------

engine.setProperty('rate', 160)

def speak(text):
    """
    Makes ALIA speak the given text and prints it to the console.
    """
    print(f"ALIA: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    """
    Listens for a command from the user via the microphone.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # We'll use a short pause to adjust for background noise before listening
        r.pause_threshold = 1
        print("Listening for a command...")
        speak("I am listening.")
        audio = r.listen(source, phrase_time_limit=5)
    
    try:
        # Use Google's speech recognition to convert audio to text
        print("Recognizing your speech...")
        user_input = r.recognize_google(audio)
        print(f"You said: {user_input}")
        return user_input
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
