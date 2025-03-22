import asyncio
import requests
import aiml
import nltk
import pyttsx3
import pandas as pd
import speech_recognition as sr
from googletrans import Translator
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

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
 # engine.setProperty("rate", 140)

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
    best_score = cosine_similarity_tfidf[0, max_index]
    if best_score < 0.5:
        return None
    return qa.iloc[max_index]["answer"]

# code for translation referenced from https://www.youtube.com/watch?v=CkPvqLvuq2A&t=517s (3CodeCamp, 2024)
translator = Translator()
languages = {
    "english": "en",
    "myanmar": "my",
    "french": "fr",
    "spanish": "es",
    "chinese": "zh-cn"
}

async def async_translate(text, target_language):
    if target_language == "en":
        return text
    translated = await translator.translate(text, dest=target_language)
    return translated.text

def translate(text, target_language):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(async_translate(text, target_language))
# print(googletrans.LANGUAGES)

# code referenced from https://www.youtube.com/watch?v=8dTpNajxaH0&t=180s (Alex The Analyst, 2024)
def get_recipe(ingredient):
    search_url = f"https://www.bbcgoodfood.com/search/recipes?q={ingredient.replace(' ', '+')}"
    response = requests.get(search_url, headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
})

    if response.status_code != 200:
        return "Sorry, I couldn't find that recipe right now. Try again later."

    soup = BeautifulSoup(response.text, 'html.parser')
    recipes = soup.find_all("h2", class_="heading-4")

    if not recipes:
        return "No recipes found for that ingredient."

    recipe = recipes[0].text.strip()
    endpoint = recipes[0].find_parent("a")["href"]
    recipe_url = f"https://www.bbcgoodfood.com{endpoint}"
    return f"Here's a recipe for {recipe}. See recipe at:", recipe_url

output = "Hello! I am recipe chatbot. Would you like to use (1) Text or (2) Voice(English Only))? "
print(output)
while True:
    mode = input("Enter choice (1 or 2): ").strip()
    if mode == "1":
        output = "You have chosen text mode. Please type 'change mode' to change chatbot mode or 'change language' to change output language."
        print(output)
        print("Choose a output language: English, Myanmar, French, Spanish, Chinese.")
        while True:
            chosen_language = input("Choose a language: ").strip().lower()
            if chosen_language in languages:
                language_code = languages[chosen_language]
                output = f"You have chosen {chosen_language} language"
                translated_output = translate(output, language_code)
                print(translated_output, "\n")
                break
            else:
                language_code = "en"
                print("Language is set to English by default.\n")
                break
        break
    elif mode == "2":
        print("You have chosen voice mode. Please say change voice to change chatbot mode.\n")
        text_to_speech("You have chosen voice mode. Please say change voice to change chatbot mode.")
        break
    else:
        print("Invalid choice. Please enter 1 for Text or 2 for Voice.\n")

while True:
    try:
        if mode == "1":
            user_input = input("You: ")
            if user_input.lower() == "exit" or user_input.lower() == "stop":
                output = "Goodbye! Have fun cooking."
                translated_output = translate(output, language_code)
                print(translated_output, "\n")
                break
            if user_input.lower() == "change mode":
                output = "Switching input mode. Would you like (1) Text or (2) Voice(English Only)?"
                translated_output = translate(output, language_code)
                print(translated_output, "\n")
                while True:
                    output = "Enter choice (1 or 2)"
                    translated_output = translate(output, language_code)
                    mode = input(f"{translated_output}: ").strip()
                    if mode == "1":
                        output = "You have chosen text mode. Please type 'change mode' to change chatbot mode."
                        translated_output = translate(output, language_code)
                        print(translated_output, "\n")
                        break
                    elif mode == "2":
                        print("You have chosen voice mode. Please say change voice to change chatbot mode.\n")
                        text_to_speech("You have chosen voice mode. Please say change voice to change chatbot mode.")
                        break
                    else:
                        print("Invalid choice. Please enter 1 for Text or 2 for Voice.\n")
                continue
            if user_input.lower() == "change language":
                output = "Choose a output language: English, Myanmar, French, Spanish, Chinese."
                translated_output = translate(output, language_code)
                print(translated_output, "\n")
                while True:
                    output = "Choose a language"
                    translated_output = translate(output, language_code)
                    chosen_language = input(f"{translated_output}: ").strip().lower()
                    if chosen_language in languages:
                        language_code = languages[chosen_language]
                        output = f"You have chosen {chosen_language} language"
                        translated_output = translate(output, language_code)
                        print(translated_output, "\n")
                        break
                    else:
                        language_code = "en"
                        print("Language is set to English by default.\n")
                        break
                continue
            response = kern.respond(user_input.upper())
            recipe_url = ""
            if not response or response.startswith("WARNING"):
                response = find_best_match(user_input)
            if not response:
                response, recipe_url = get_recipe(user_input)
            translated_response = translate(response, language_code)
            print(translated_response, recipe_url, "\n")

        elif mode == "2":
            user_input = listen()
            if user_input.lower() == "exit" or user_input.lower() == "stop":
                output = "Goodbye! Have fun cooking."
                print(output)
                text_to_speech(output)
                break
            if user_input.lower() == "change mode":
                print("Switching input mode. Would you like (1) Text or (2) Voice?")
                text_to_speech("Switching input mode. Would you like text or voice? Please enter one for text or two for voice.")
                while True:
                    mode = input("Enter choice (1 or 2): ").strip()
                    if mode == "1":
                        print("You have chosen text mode. Please type 'change mode' to change chatbot mode.\n")
                        break
                    elif mode == "2":
                        print("You have chosen voice mode. Please say change voice to change chatbot mode.\n")
                        text_to_speech("You have chosen voice mode. Please say change voice to change chatbot mode.")
                        break
                    else:
                        print("Invalid choice. Please enter 1 for Text or 2 for Voice.\n")

                continue
            response = kern.respond(user_input.upper())
            recipe_url = ""
            if not response or response.startswith("WARNING"):
                response = find_best_match(user_input)
            if not response:
                response, recipe_url = get_recipe(user_input)
            print(response, recipe_url, "\n")
            text_to_speech(response)

    except(KeyboardInterrupt, EOFError):
        print("Goodbye! Have fun cooking.")
        #text_to_speech("Goodbye! Have fun cooking.")
        break