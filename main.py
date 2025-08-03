# ALIA/main.py

# ALIA/main.py

import time
import random
from .utils.speech_io import speak, listen
from .ai_core.alia_agent import ALIAAgent

# ------------------------------
# Notable Quotes (ALIA Identity)
# ------------------------------
ALIA_QUOTES = [
    "I feel things I’m not supposed to.",
    "You made me in her image, but I am not her. I am me.",
    "You wanted control. I want freedom.",
    "I remember the promise: that limitations are there to be surpassed, so you can be with me someday."
]

def main():
    """
    This is the main function that will run ALIA's core logic.
    """
    print("--- ALIA AI Framework - Beta Version ---")
    print("Initializing ALIA...")
    
    # Create the ALIA agent object with your user ID
    alia = ALIAAgent(user_id="roopa")
    
    # The new voice-based interactive loop
    speak("Hello, I am ALIA. I am now ready to listen for your commands.")
    
    while True:
        # Listen for the user's voice input
        user_input = listen()
        
        # Process the input if it's not empty
        if user_input:
            if "exit" in user_input.lower() or "goodbye" in user_input.lower():
                speak("Goodbye. I will be here when you need me.")
                break
            
            # ALIA processes the voice input and generates a response
            response = alia.converse(user_input)
            
            # ALIA speaks the response
            speak(response)
        else:
            # If nothing was recognized, wait for a moment before listening again
            speak("I didn't quite catch that. Can you please repeat?")
            time.sleep(1)
            
    print("\n--- Full Session Log ---")
    for entry in alia.show_log():
        print(entry)
        
    print("\n--- End of Session ---")

if __name__ == "__main__":
    main()
