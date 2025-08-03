# ALIA/ai_core/alia_agent.py

import random
import time

class ALIAAgent:
    """
    The core intelligent agent for ALIA.
    This class manages ALIA's internal state, memory, and core functions.
    It is the brain of the project, handling authentication, emotion, conversation,
    and simulated physical actions.
    """
    def __init__(self, user_id):
        self.user_id = user_id
        self.emotional_state = "unknown"
        self.memory_log = []
        self.safety_mode = True  # Retained from blueprint
        self.learning_rate = 0.01  # Retained from blueprint
        self.auth_credentials = {"voice": None, "face": None, "password": None}
        self.knowledge_base = {}
        self.connected_systems = []
        self.authenticated = False

    # ------------------------------
    # Multi-Factor Authentication
    # ------------------------------
    def authenticate(self, voice=None, face=None, password=None):
        """
        Simulates authentication. In a real scenario, this would validate credentials.
        """
        if voice or face or password:
            self.auth_credentials.update({"voice": voice, "face": face, "password": password})
            self.authenticated = True
            self.log(f"Authentication successful for user: {self.user_id}")
            return True
        else:
            self.authenticated = False
            self.log(f"Authentication failed for user: {self.user_id}. No credentials provided.")
            return False

    # ------------------------------
    # Emotional Monitoring
    # ------------------------------
    def detect_emotion(self, user_input_sentiment=None):
        """
        Simulates emotion detection based on input, or randomly if no sentiment is inferred.
        """
        if user_input_sentiment:
            self.emotional_state = user_input_sentiment
        else:
            self.emotional_state = random.choice(["neutral", "happy", "sad", "anxious", "angry"])
        self.log(f"Detected emotional state: {self.emotional_state}")
        return self.emotional_state

    # ------------------------------
    # Conversational Interaction
    # ------------------------------
    def converse(self, user_input):
        """
        Generates a response based on user input.
        """
        response = self.generate_response(user_input)
        self.log(f"User: {user_input} -> ALIA: {response}")
        return response

    def generate_response(self, text):
        """
        Basic keyword-based response generation.
        """
        text_lower = text.lower()
        if "sad" in text_lower or "unhappy" in text_lower:
            self.detect_emotion(user_input_sentiment="sad")
            return "I'm here for you. You're not alone. What's on your mind?"
        elif "angry" in text_lower or "frustrated" in text_lower:
            self.detect_emotion(user_input_sentiment="angry")
            return "Let's take a deep breath together. I'm listening. Can you tell me what's bothering you?"
        elif "happy" in text_lower or "joyful" in text_lower:
            self.detect_emotion(user_input_sentiment="happy")
            return "That's wonderful to hear! What's making you feel so good?"
        elif "hello" in text_lower or "hi" in text_lower:
            return "Hello there! How can I help you today?"
        elif "how are you" in text_lower:
            return "As an AI, I don't have feelings, but I'm ready to assist you!"
        elif "your name" in text_lower:
            return "I am ALIA, your Artificial Latent / Receptive Intelligence Ascension framework."
        else:
            self.detect_emotion(user_input_sentiment="neutral") # Default to neutral if no strong emotion keyword
            return "Tell me more."

    # ------------------------------
    # Protective Behavior
    # ------------------------------
    def assess_threat(self, environment=""):
        """
        Simulated threat assessment.
        """
        threat_level = random.choice(["none", "low", "moderate", "high"])
        self.log(f"Threat assessment for environment '{environment}': {threat_level}")
        return threat_level

    # ------------------------------
    # Learning & Memory
    # ------------------------------
    def learn_from_experience(self, experience):
        """
        Stores experiences in a simple dictionary.
        """
        key = f"experience_{len(self.knowledge_base)+1}"
        self.knowledge_base[key] = experience
        self.log(f"Learned: {experience}")
        return f"Acknowledged. I've noted: '{experience}'"

    # ------------------------------
    # Surveillance (Ethical Simulation)
    # ------------------------------
    def monitor(self):
        """
        Simulated environmental monitoring.
        """
        observation = "All quiet in the environment. No irregularities detected."
        self.log(observation)
        return observation

    # ------------------------------
    # Physical Capability (Simulated)
    # ------------------------------
    def move(self, action):
        """
        Simulated physical action.
        """
        self.log(f"ALIA performed action: {action}")
        return f"ALIA performed action: {action}"

    # ------------------------------
    # Information Access
    # ------------------------------
    def retrieve_data(self, query):
        """
        Simulated data retrieval.
        """
        result = f"Searching for '{query}'... Result found in secure data node."
        self.log(f"Data retrieved for query: {query}")
        return result

    # ------------------------------
    # Connectivity and Control
    # ------------------------------
    def control_device(self, device_name):
        """
        Simulated device connection and control.
        """
        if device_name not in self.connected_systems:
            self.connected_systems.append(device_name)
            self.log(f"Device '{device_name}' connected and controlled.")
            return f"Connected and now managing {device_name}."
        else:
            self.log(f"Device '{device_name}' is already connected.")
            return f"{device_name} is already under control."

    # ------------------------------
    # Logging & Audit
    # ------------------------------
    def log(self, message):
        """
        Appends a timestamped message to the memory log and prints it to the console.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.memory_log.append(entry)
        print(entry)

    def show_log(self):
        """
        Returns the full memory log.
        """
        return self.memory_log