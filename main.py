import asyncio
import requests
import aiml
import nltk
import pyttsx3
import pandas as pd
import speech_recognition as sr
import wikipedia
from googletrans import Translator
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer, Expression, ResolutionProver
from nltk.corpus import wordnet
from nltk.sem.logic import NegatedExpression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import random
import csv
import re
warnings.filterwarnings("ignore")

# Only needs to be run once
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

kern = aiml.Kernel()
kern.verbose(False)
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="pattern.xml")

# code referenced from https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#wordnetlemmatizerwithappropriatepostag (Machine Learning Plus, 2018)
lemmatizer = WordNetLemmatizer()

def nltk_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    else:
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk_wordnet_tag(x[1])), nltk_tagged)
    words = []
    for word, tag in wn_tagged:
        if tag is None:
            words.append(word)
        else:
            words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(words)

def custom_tokenizer(text):
    return lemmatize_sentence(text).split()

qa = pd.read_csv("q&a.csv", header=None, names=["question", "answer"])

vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
q_vectors = vectorizer.fit_transform(qa["question"])

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
    if best_score < 0.6:
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

# LOGIC KB
read_expr = Expression.fromstring

def load_logic_kb():
    logic_kb = []
    df = pd.read_csv("logic_kb.csv", header=None)
    for _, row in df.iterrows():
        logic_kb.append(read_expr(row[0]))
    return logic_kb

# def add_logic_fact(added_fact_expr, logic_kb):
#     added_fact = read_expr(added_fact_expr)
#     subject, category = added_fact.argument.args
#     for fact in logic_kb:
#         existing_subject, existing_category = fact.argument.args
#         if str(existing_subject) == str(subject) and str(existing_category) != str(category):
#             return f"Contradiction already known. Thus, {subject} will not be remembered as {category} but as {existing_category}."
#         logic_kb.append(added_fact)
#         with open("logic_kb.csv", 'a') as file:
#             file.write(f"{added_fact_expr}\n")
#         return f"OK, I will remember that {subject} is {category}."
def add_logic_fact(expr_str, logic_kb):
    expr = read_expr(expr_str)
    neg_expr = read_expr(f"-{expr}")
    if ResolutionProver().prove(neg_expr, logic_kb):
        return "That contradicts what I already know! I won’t remember it."
    else:
        with open("logic_kb.csv", 'a') as file:
            file.write(f"{expr_str}\n")
        if expr_str.startswith("-"):
            expr_str = expr_str[1:]
            return f"OK, I will remember that {expr_str.split('(')[1].split(')')[0]} is not {expr_str.split('(')[0]}."
        else:
            return f"OK, I will remember that {expr_str.split('(')[1].split(')')[0]} is {expr_str.split('(')[0]}."

def check_logic_fact(check_fact_expr, logic_kb):
    check_fact_expr = check_fact_expr.strip().lower()
    fact = read_expr(check_fact_expr)
    negated_fact = NegatedExpression(fact)
    if ResolutionProver().prove(fact, logic_kb):
        return "It is CORRECT!"
    elif ResolutionProver().prove(negated_fact, logic_kb):
        return "It is INCORRECT!"
    else:
        return "Sorry, I don't know."

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
            translated_input = translate(user_input, "en")
            if translated_input.lower() == "exit" or translated_input.lower() == "stop":
                output = "Goodbye! Have fun cooking."
                translated_output = translate(output, language_code)
                print(translated_output, "\n")
                break
            if translated_input.lower() == "change mode":
                output = "Switching input mode. Would you like (1) Text or (2) Voice(English Only)?"
                translated_output = translate(output, language_code)
                print(translated_output, "\n")
                while True:
                    output = "Enter choice (1 or 2)"
                    translated_output = translate(output, language_code)
                    mode = input(f"{translated_output}: ").strip()
                    translated_mode = translate(mode, "en").lower()
                    if translated_mode == "1":
                        output = "You have chosen text mode. Please type 'change mode' to change chatbot mode."
                        translated_output = translate(output, language_code)
                        print(translated_output, "\n")
                        break
                    elif translated_mode == "2":
                        print("You have chosen voice mode. Please say change voice to change chatbot mode.\n")
                        text_to_speech("You have chosen voice mode. Please say change voice to change chatbot mode.")
                        break
                    else:
                        print("Invalid choice. Please enter 1 for Text or 2 for Voice.\n")
                continue
            if translated_input.lower() == "change language":
                output = "Choose a output language: English, Myanmar, French, Spanish, Chinese."
                translated_output = translate(output, language_code)
                print(translated_output, "\n")
                while True:
                    output = "Choose a language"
                    translated_output = translate(output, language_code)
                    chosen_language = input(f"{translated_output}: ").strip().lower()
                    translated_chosen_language = translate(chosen_language, "en").lower()
                    if translated_chosen_language in languages:
                        language_code = languages[translated_chosen_language]
                        output = f"You have chosen {translated_chosen_language} language"
                        translated_output = translate(output, language_code)
                        print(translated_output, "\n")
                        break
                    else:
                        language_code = "en"
                        print("Language is set to English by default.\n")
                        break
                continue
            response = kern.respond(translated_input.upper())
            recipe_url = ""
            if response.startswith("#"):
                cmd, value = response[1:].split("$", 1)
                if cmd == "0":
                    response = value
                    translated_response = translate(response, language_code)
                    print(translated_response, "\n")
                    break
                elif cmd == "1":
                    try:
                        response = wikipedia.summary(value, sentences=2, auto_suggest=False)
                    except:
                        response = "Sorry, can you please be more specific?"
                elif cmd == "31":
                    subject, category = map(str.strip, value.split(" is "))
                    subject = subject.lower()
                    category = category.lower()
                    if "not " in category:
                        logic_expr = f"-{category[4:]}({subject})"
                    else:
                        logic_expr = f"{category}({subject})"
                    print(logic_expr)
                    response = add_logic_fact(logic_expr, load_logic_kb())
                elif cmd == "32":
                    subject, category = map(str.strip, value.split(" is "))
                    subject = subject.lower()
                    category = category.lower()
                    if "not " in category:
                        logic_expr = f"-{category[4:]}({subject})"
                    else:
                        logic_expr = f"{category}({subject})"
                    response = check_logic_fact(logic_expr, load_logic_kb())
            else:
                if not response or response.startswith("WARNING"):
                    response = find_best_match(translated_input)
                if not response:
                    response, recipe_url = get_recipe(translated_input)
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
            if response.startswith("#"):
                cmd, value = response[1:].split("$", 1)
                if cmd == "0":
                    response = value
                    print(response, "\n")
                    break
                elif cmd == "1":
                    try:
                        response = wikipedia.summary(value, sentences=2, auto_suggest=False)
                    except:
                        response = "Sorry, can you please be more specific?"
                elif cmd == "31":
                    subject, category = map(str.strip, value.split(" is "))
                    subject = subject.lower()
                    category = category.lower()
                    if "not " in category:
                        logic_expr = f"-{category[4:]}({subject})"
                    else:
                        logic_expr = f"{category}({subject})"
                    response = add_logic_fact(logic_expr, load_logic_kb())
                elif cmd == "32":
                    subject, category = map(str.strip, value.split(" is "))
                    subject = subject.lower()
                    category = category.lower()
                    if "not " in category:
                        logic_expr = f"-{category[4:]}({subject})"
                    else:
                        logic_expr = f"{category}({subject})"
                    response = check_logic_fact(logic_expr, load_logic_kb())
            else:
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