import asyncio
import requests
import aiml
import nltk
import wikipedia
import pandas as pd
import numpy as np
from googletrans import Translator
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer, Expression, ResolutionProver
from nltk.corpus import wordnet
from nltk.sem.logic import NegatedExpression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import warnings
import random
import csv
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from simpful import *
from tensorflow.keras.applications.efficientnet import preprocess_input

warnings.filterwarnings("ignore")

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

model = load_model("eff_model.keras")

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

def add_logic_fact(expr_str, logic_kb):
    expr = read_expr(expr_str)
    neg_expr = read_expr(f"-{expr}")
    if expr in logic_kb:
        return "Right! I know that as well."
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

# GAME
game_kb = "game_kb.csv"
def get_random():
    with open(game_kb, newline="") as file:
        return random.choice(list(csv.reader(file)))

def guess_game(language_code):
    entry = get_random()
    hidden, traits = entry[0], entry[1:]
    output = "Guess the food! You have 5 lives."
    translated_output = translate(output, language_code)
    print(translated_output)
    lives = 5
    while lives > 0:
        output = "Enter 'guess' to guess the food or trait of the food to check."
        translated_output = translate(output, language_code)
        print(translated_output)
        user_input = input("You: ").strip()
        translated_input = translate(user_input, "en").lower()
        if translated_input == "guess":
            user_guess = input("Your guess:").strip()
            translated_guess = translate(user_guess, "en").lower()
            if translated_guess == hidden:
                output = f"Congratulations! You are correct. The food was {hidden}."
                translated_output = translate(output, language_code)
                print(translated_output)
                return
            else:
                lives -= 1
                output = f"Wrong guess!"
                translated_output = translate(output, language_code)
                print(translated_output)
        elif translated_input in map(str.lower, traits):
            output = "That's a trait of the food. Getting close."
            translated_output = translate(output, language_code)
            print(translated_output)
        else:
            lives -= 1
            output = "Oops! It is not a trait of the food."
            translated_output = translate(output, language_code)
            print(translated_output)
        output = f"You have {lives} lives."
        translated_output = translate(output, language_code)
        print(translated_output, "\n")
    output = f"Game Over. The food was {hidden}."
    translated_output = translate(output, language_code)
    print(translated_output)

# FUZZY LOGIC
FS = FuzzySystem()

FS.add_linguistic_variable("Spiciness", LinguisticVariable([
    FuzzySet(function=Trapezoidal_MF(a=0, b=2, c=4), term="low"),
    FuzzySet(function=Triangular_MF(a=3, b=4, c=7), term="medium"),
    FuzzySet(function=Trapezoidal_MF(a=6, b=8, c=10), term="high")
], universe_of_discourse=[0,10]))

FS.add_linguistic_variable("Healthiness", LinguisticVariable([
    FuzzySet(function=Trapezoidal_MF(a=0, b=2, c=4), term="low"),
    FuzzySet(function=Triangular_MF(a=3, b=5, c=7), term="medium"),
    FuzzySet(function=Trapezoidal_MF(a=6, b=8, c=10), term="high")
], universe_of_discourse=[0,10]))

FS.set_crisp_output_value("low", 0)
FS.set_crisp_output_value("medium", 10)
FS.set_crisp_output_value("high", 20)

FS.add_rules([
    "IF (Spiciness IS low) AND (Healthiness IS high) THEN (Recommendation IS high)",
    "IF (Spiciness IS medium) AND (Healthiness IS medium) THEN (Recommendation IS medium)",
    "IF (Spiciness IS high) AND (Healthiness IS low) THEN (Recommendation IS low)"
])

def get_recommendation(spice_score, health_score):
    FS.set_variable("Spiciness", spice_score)
    FS.set_variable("Healthiness", health_score)
    result = FS.Sugeno_inference(["Recommendation"])
    level = float(result["Recommendation"])
    if level >= 15:
        return "Highly recommended for you! It is considered healthy."
    elif level >= 7:
        return "Moderately recommended. It is not bad for your health."
    else:
        return "Not recommended. It is not good for your stomach."

# IMAGE CLASSIFICATION
class_labels = {
    0: "bibimbap",
    1: "fried rice",
    2: "grilled cheese sandwich",
    3: "hamburger",
    4: "pad thai",
    5: "pancakes",
    6: "pizza",
    7: "spring rolls",
    8: "steak",
    9: "sushi"
}

def classify_food(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        classifications = model.predict(img_array)
        class_index = np.argmax(classifications[0])
        label = class_labels[class_index]
        confidence = classifications[0][class_index] * 100
        return f"I think this is {label} (confidence: {confidence:.2f}%)"
    except Exception as e:
        return f"Error loading image: {str(e)}"


print("Hello! I am May's recipe chatbot. Please type 'change language' to change the language of the chatbot.")
print("Choose chatbot language: English, Myanmar, French, Spanish, Chinese.")
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

while True:
    try:
        user_input = input("You: ")
        translated_input = translate(user_input, "en")
        if translated_input.lower() == "exit" or translated_input.lower() == "stop":
            output = "Goodbye! Have fun cooking."
            translated_output = translate(output, language_code)
            print(translated_output, "\n")
            break
        if translated_input.lower() == "change language":
            output = "Choose chatbot language: English, Myanmar, French, Spanish, Chinese."
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
                subject = subject.lower().replace("the ", "").strip()
                category = category.lower()
                if "not " in category:
                    logic_expr = f"-{category[4:]}({subject})"
                else:
                    logic_expr = f"{category}({subject})"
                response = add_logic_fact(logic_expr, load_logic_kb())
            elif cmd == "32":
                subject, category = map(str.strip, value.split(" is "))
                subject = subject.lower().replace("the ", "").strip()
                category = category.lower()
                if "not " in category:
                    logic_expr = f"-{category[4:]}({subject})"
                else:
                    logic_expr = f"{category}({subject})"
                response = check_logic_fact(logic_expr, load_logic_kb())
            elif cmd == "33":
                guess_game(language_code)
                response = ""
            elif cmd == "34":
                # code referenced form https://en.ittrip.xyz/python/python-file-dialog (IT trip, 2024)
                root = tk.Tk()
                root.withdraw()
                image_path = filedialog.askopenfilename(
                    title = "Select an image for classification",
                    filetypes = (("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
                )

                if image_path:
                    img = Image.open(image_path)
                    img.show()

                    try:
                        response = classify_food(image_path)
                    except Exception as e:
                        response = "Sorry, I could not identify this image."
                else:
                    output = "No image selected."
            elif cmd == "35":
                output = "On a scale of 0–10, how spicy is the food? "
                translated_output = translate(output, language_code)
                spice = float(input(translated_output))
                output = "On a scale of 0–10, how healthy is the food? "
                translated_output = translate(output, language_code)
                health = float(input(translated_output))
                response = get_recommendation(spice, health)

        else:
            if not response or response.startswith("WARNING"):
                response = find_best_match(translated_input)
            if not response:
                response, recipe_url = get_recipe(translated_input)
        translated_response = translate(response, language_code)
        print(translated_response, recipe_url, "\n")

    except(KeyboardInterrupt, EOFError):
        print("Goodbye! Have fun cooking.")
        break