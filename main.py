import aiml
import pandas as pd
import nltk
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

print("Recipe Chatbot: Hello! I am recipe chatbot. Type 'exit' to stop.")

def find_best_match(user_input):
    input_vectors = vectorizer.transform([user_input])
    cosine_similarity_tfidf = cosine_similarity(input_vectors, q_vectors)
    max_index = cosine_similarity_tfidf.argmax(axis=1)[0]
    # best_score = cosine_similarity_tfidf[0, max_index]
    return qa.iloc[max_index]["answer"]

while True:
    try:
        user_input = input("You: ")
    except(KeyboardInterrupt, EOFError):
        print("Recipe Chatbot: Goodbye! Have fun cooking.")
        break
    if user_input == "exit":
        print("Recipe Chatbot: Goodbye! Have fun cooking.")
        break
    response = kern.respond(user_input.upper())
    if not response or response.startswith("WARNING"):
        response = find_best_match(user_input)
    print("Recipe Chatbot:", response)