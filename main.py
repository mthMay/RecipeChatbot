import aiml
import pandas as pd

kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="pattern.xml")

print("Recipe Chatbot: Hello! I can suggest recipes. Type 'exit' to stop.")

while True:
    try:
        userInput = input("You: ")
    except(KeyboardInterrupt, EOFError):
        print("Recipe Chatbot: Goodbye! Have fun cooking.")
        break
    if userInput == "exit":
        print("Recipe Chatbot: Goodbye! Have fun cooking.")
        break
    response = kern.respond(userInput.upper())
    print("Recipe Chatbot:", response)