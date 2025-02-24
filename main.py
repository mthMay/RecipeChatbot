import aiml
import nltk
import pyttsx3
import pandas as pd
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

qa = pd.read_csv("q&a.csv", header=None, names=["question", "answer"])
nltk.download('punkt')

vectorizer = TfidfVectorizer()
q_vectors = vectorizer.fit_transform(qa["question"])

kern = aiml.Kernel()
kern.verbose(False)
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="pattern.xml")

# code referenced from https://www.geeksforgeeks.org/convert-text-speech-python/ (GeeksforGeeks, 2024)
engine = pyttsx3.init()
engine.setProperty("rate", 140)

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# code referenced from https://www.geeksforgeeks.org/python-speech-recognition-module/ (GeeksforGeeks, 2024)
recognizer = sr.Recognizer()
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5)
            user_input = recognizer.recognize_google(audio).lower()
            print("You (Voice):", user_input)
            return user_input.lower()
        except sr.UnknownValueError:
            print("I did not catch that. Can you repeat?")
            return None
        except sr.RequestError:
            print("Speech recognition service is unavailable. Please try again.")
            return None
        except sr.WaitTimeoutError:
            print("You took too long to respond. Please try again.")
            return None


def find_best_match(user_input):
    input_vectors = vectorizer.transform([user_input])
    cosine_similarity_tfidf = cosine_similarity(input_vectors, q_vectors)
    max_index = cosine_similarity_tfidf.argmax(axis=1)[0]
    # best_score = cosine_similarity_tfidf[0, max_index]
    return qa.iloc[max_index]["answer"]

print("Recipe Chatbot: Hello! I am recipe chatbot. Would you like to use (1) Text or (2) Voice? ")

while True:
    mode = input("Recipe Chatbot: Enter choice (1 or 2): ").strip()
    if mode == "1":
        print("Recipe Chatbot: You have chosen text mode. Please type 'change mode' to change chatbot mode.")
        break
    elif mode == "2":
        print("Recipe Chatbot (Voice): You have chosen voice mode. Please say change voice to change chatbot mode.")
        text_to_speech("You have chosen voice mode. Please say change voice to change chatbot mode.")
        break
    else:
        print("Recipe Chatbot: Invalid choice. Please enter 1 for Text or 2 for Voice.")

while True:
    try:
        if mode == "1":
            user_input = input("You: ")
            if user_input.lower() == "exit" or user_input.lower() == "stop":
                print("Recipe Chatbot: Goodbye! Have fun cooking.")
                break
            if user_input.lower() == "change mode":
                print("Recipe Chatbot: Switching input mode. Would you like (1) Text or (2) Voice?")
                while True:
                    mode = input("Recipe Chatbot: Enter choice (1 or 2): ").strip()
                    if mode == "1":
                        print("Recipe Chatbot: You have chosen text mode. Please type 'change mode' to change chatbot mode.")
                        break
                    elif mode == "2":
                        print("Recipe Chatbot (Voice): You have chosen voice mode. Please say change voice to change chatbot mode.")
                        text_to_speech("You have chosen voice mode. Please say change voice to change chatbot mode.")
                        break
                    else:
                        print("Recipe Chatbot: Invalid choice. Please enter 1 for Text or 2 for Voice.")

                continue
            response = kern.respond(user_input.upper())

            if not response or response.startswith("WARNING"):
                response = find_best_match(user_input)
            print("Recipe Chatbot:", response)

        elif mode == "2":
            user_input = listen()
            if user_input.lower() == "exit" or user_input.lower() == "stop":
                print("Recipe Chatbot (Voice): Goodbye! Have fun cooking.")
                text_to_speech("Goodbye! Have fun cooking.")
                break
            if user_input.lower() == "change mode":
                print("Recipe Chatbot (Voice): Switching input mode. Would you like (1) Text or (2) Voice?")
                text_to_speech("Switching input mode. Would you like text or voice? Please enter one for text or two for voice.")
                while True:
                    mode = input("Recipe Chatbot: Enter choice (1 or 2): ").strip()
                    if mode == "1":
                        print("Recipe Chatbot: You have chosen text mode. Please type 'change mode' to change chatbot mode.")
                        break
                    elif mode == "2":
                        print("Recipe Chatbot (Voice): You have chosen voice mode. Please say change voice to change chatbot mode.")
                        text_to_speech("You have chosen voice mode. Please say change voice to change chatbot mode.")
                        break
                    else:
                        print("Recipe Chatbot: Invalid choice. Please enter 1 for Text or 2 for Voice.")

                continue
            response = kern.respond(user_input.upper())

            if not response or response.startswith("WARNING"):
                response = find_best_match(user_input)
            print("Recipe Chatbot (Voice):", response)
            text_to_speech(response)

    except(KeyboardInterrupt, EOFError):
        print("Recipe Chatbot: Goodbye! Have fun cooking.")
        text_to_speech("Goodbye! Have fun cooking.")
        break
